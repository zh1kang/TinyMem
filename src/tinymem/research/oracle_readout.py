"""Known variant states calibrate the read path without a learned writer."""
from collections.abc import Sequence

import torch
from torch import nn

from tinymem.memory.readout_interface import ReadoutBridge, check_readout_state
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.pretrained import PretrainedReader
from tinymem.research.prefix_reader import prefix_answer_loss
from tinymem.research.readout_runner import ReadoutQuery
from tinymem.research.paired_readout_evaluation import summarize_pairs


def oracle_variant_state(variant: str, device: torch.device) -> LatentSlotState:
    if variant not in ('a', 'b'):
        raise ValueError('oracle variant must be a or b')
    values = torch.zeros(1, 2, 8, device=device, dtype=torch.float32)
    values[0, 0, 0] = -1 if variant == 'a' else 1
    return LatentSlotState(values, torch.ones(1, 2, device=device, dtype=torch.bool))


def check_oracle_state(state: LatentSlotState) -> None:
    check_readout_state(state)
    if state.values.requires_grad or state.values.grad_fn is not None or not state.valid.all():
        raise ValueError('oracle values must be fixed and both slots occupied')
    value = float(state.values[0, 0, 0])
    expected = torch.zeros_like(state.values)
    expected[0, 0, 0] = value
    if value not in (-1, 1) or not torch.equal(state.values, expected):
        raise ValueError('oracle state must contain exactly the declared variant bit')


def train_oracle_step(reader: PretrainedReader, bridge: ReadoutBridge, state: LatentSlotState,
                      before_ids: tuple[int, ...], queries: Sequence[ReadoutQuery],
                      optimizer: torch.optim.Optimizer, *, adapter_parameters: tuple[nn.Parameter, ...]) -> dict:
    check_oracle_state(state)
    if any(m.training for m in reader.model.modules()):
        raise ValueError('read-side model must be in evaluation mode')
    reader_owned = {id(p) for p in adapter_parameters}
    if len(reader_owned) != len(adapter_parameters) or reader_owned != {id(p) for p in reader.model.parameters() if p.requires_grad}:
        raise ValueError('reader adapter ownership changed')
    parameters = [*bridge.parameters(), *adapter_parameters]
    owned = {id(p) for p in parameters}
    optimized = [p for group in optimizer.param_groups for p in group['params']]
    if (len(owned) != len(parameters) or len(optimized) != len(owned)
            or {id(p) for p in optimized} != owned or any(not p.requires_grad for p in parameters)):
        raise ValueError('optimizer must own exactly bridge and declared read adapter')
    if not queries:
        raise ValueError('queries are required')
    if any(p.grad is not None for p in reader.model.parameters() if id(p) not in reader_owned):
        raise ValueError('frozen reader has unexpected gradients')
    original_values, original_valid = state.values.clone(), state.valid.clone()
    optimizer.zero_grad(set_to_none=True)
    device = reader.model.device
    memory = bridge(state)
    before = torch.tensor(before_ids, device=device)
    loss = torch.stack([
        prefix_answer_loss(reader, before, memory, torch.tensor(q.after_ids, device=device),
                           torch.tensor(q.answer_ids, device=device)) for q in queries
    ]).mean()
    if not torch.isfinite(loss):
        raise ValueError('nonfinite answer loss')
    loss.backward()
    if any(p.grad is None for p in parameters):
        raise ValueError('all declared parameters must receive gradients')
    if any(p.grad is not None for p in reader.model.parameters() if id(p) not in reader_owned):
        raise ValueError('frozen reader gradient ownership violation')
    gradient_norms = {}
    for name, group in (('bridge', tuple(bridge.parameters())), ('adapter', adapter_parameters)):
        value = torch.stack([p.grad.float().square().sum() for p in group]).sum().sqrt() if group else loss.new_zeros(())
        if group and (not torch.isfinite(value) or value <= 0):
            raise ValueError(f'{name} must receive finite nonzero gradients')
        gradient_norms[name + '_gradient_norm'] = float(value)
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    result = {'answer_ce': float(loss.detach()), 'gradient_norm': float(norm), 'persistent_bytes': 66,
              'write_states': 0, 'oracle_value': float(state.values[0, 0, 0]),
              'supervised_tokens': sum(len(q.answer_ids) for q in queries), **gradient_norms}
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in parameters):
        raise ValueError('optimizer produced nonfinite parameters')
    if not torch.equal(state.values, original_values) or not torch.equal(state.valid, original_valid):
        raise ValueError('fixed oracle state changed during training')
    return result


def summarize_oracle(records, rows) -> dict:
    summary = summarize_pairs(records, rows)
    summary.pop('binding_passed')
    scores, gaps = summary['scores'], summary['known_accuracy_gaps']
    known = (scores['normal']['known_accuracy'] >= .95 and summary['swapped_donor_known_accuracy'] >= .95
             and min(gaps[c] for c in ('zero', 'no_memory')) >= .4)
    return {**summary, 'known_bit_readout_passed': known,
            'binary_qa_passed': known and scores['normal']['missing_accuracy'] >= .95,
            'evidence_kind': 'privileged_variant_bit_training_fit_only'}
