"""Normalized correction-target answer loss with a fixed learned-state auxiliary."""
from collections.abc import Sequence
import math

import torch

from tinymem.memory.query_pool_slots import QueryPoolSlotWriter
from tinymem.memory.readout_interface import ReadoutBridge
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.independent_fact_answer_training import (
    _validate_modules, _validate_rows,
)
from tinymem.research.independent_fact_data import FactPrompt
from tinymem.research.independent_fact_learned_state import validate_learned_targets
from tinymem.research.independent_fact_recurrent_training import RecurrentBatch, _check_state, _validate_batch
from tinymem.research.pretrained import PretrainedReader
from tinymem.research.prefix_reader import prefix_answer_losses


def correction_query(before: int, after: int) -> int | None:
    """Map a valid one-fact correction to its known-answer query index."""
    if any(type(code) is not int or not 0 <= code < 16 for code in (before, after)):
        raise ValueError('world codes must be integers from zero through fifteen')
    changed = before ^ after
    if changed == 0:
        return None
    if changed & (changed - 1):
        raise ValueError('a correction must change exactly one fact')
    return changed.bit_length() - 1


def _weighted_answer_gradient(reader, bridge, values, valid, index, row, device, query, multiplier):
    single = values[index:index + 1].detach().to(device).clone().requires_grad_(True)
    memory = bridge(LatentSlotState(single, valid[index:index + 1].detach().to(device).clone()))
    before = torch.tensor(row.before_ids, dtype=torch.long, device=device)
    examples = [(before, memory,
                 torch.tensor(q.after_ids, dtype=torch.long, device=device),
                 torch.tensor(q.answer_ids, dtype=torch.long, device=device)) for q in row.queries]
    losses = prefix_answer_losses(reader, examples)
    if query is None or multiplier == 1:
        loss = losses.sum() / 6
    else:
        loss = (losses.sum() + (multiplier - 1) * losses[query]) / (5 + multiplier)
    if not torch.isfinite(loss):
        raise ValueError('nonfinite weighted answer loss')
    single_gradient = torch.autograd.grad(loss, single)[0]
    if not torch.isfinite(single_gradient).all():
        raise ValueError('nonfinite answer-state gradient')
    gradient = torch.zeros_like(values)
    gradient[index:index + 1] = single_gradient.to(values.device)
    return float(loss.detach()), losses.detach().cpu().tolist(), gradient


def train_correction_weight_batch(
    writer: QueryPoolSlotWriter,
    batch: RecurrentBatch,
    optimizer: torch.optim.Optimizer,
    *,
    reader: PretrainedReader,
    bridge: ReadoutBridge,
    worlds: Sequence[FactPrompt],
    targets: LatentSlotState,
    state_weight: float,
    correction_multiplier: float = 1.0,
    measure_only: bool = False,
) -> dict[str, float | int | bool]:
    """Use full recurrent gradients; measurement leaves weights and optimizer unchanged."""
    if type(state_weight) not in (int, float) or not math.isfinite(state_weight) or state_weight < 0:
        raise ValueError('state_weight must be finite and nonnegative')
    if type(measure_only) is not bool:
        raise TypeError('measure_only must be boolean')
    if (type(correction_multiplier) not in (int, float)
            or not math.isfinite(correction_multiplier) or correction_multiplier <= 0):
        raise ValueError('correction_multiplier must be finite and positive')
    _validate_batch(batch, writer)
    changed_queries = [[correction_query(int(before), int(after)) for before, after in zip(bs, ats, strict=True)]
                       for bs, ats in zip(batch.before_codes, batch.after_codes, strict=True)]
    validate_learned_targets(targets)
    rows = _validate_rows(worlds)
    parameters, device = _validate_modules(reader, bridge, writer, optimizer)
    snapshots = {k: v.clone() for k, v in batch.__dict__.items() if isinstance(v, torch.Tensor)}
    teacher_values, teacher_valid = targets.values.clone(), targets.valid.clone()
    optimizer.zero_grad(set_to_none=True)
    state = LatentSlotState(targets.values[batch.initial_codes], targets.valid[batch.initial_codes])
    states = []
    for step in range(4):
        state = writer(state, batch.events[:, step], batch.event_valid[:, step])
        _check_state(state, batch.events.shape[0], writer)
        states.append(state)
    ce_gradients = []
    losses = []
    unweighted_losses = []
    diagnostics = {
        'correction_event_count': 0, 'correction_target_ce_sum': 0.,
        'correction_untouched_ce_sum': 0., 'correction_absent_ce_sum': 0.,
        'repetition_event_count': 0, 'repetition_known_ce_sum': 0., 'repetition_absent_ce_sum': 0.,
    }
    count = batch.events.shape[0] * 4
    for step, state in enumerate(states):
        accumulated = torch.zeros_like(state.values)
        for row in range(batch.events.shape[0]):
            code = int(batch.after_codes[row, step])
            query = changed_queries[row][step]
            loss, per_query, gradient = _weighted_answer_gradient(
                reader, bridge, state.values, state.valid, row, rows[code], device, query, correction_multiplier,
            )
            losses.append(loss)
            unweighted_losses.append(sum(per_query) / 6)
            if query is None:
                diagnostics['repetition_event_count'] += 1
                diagnostics['repetition_known_ce_sum'] += sum(per_query[:4])
                diagnostics['repetition_absent_ce_sum'] += sum(per_query[4:])
            else:
                diagnostics['correction_event_count'] += 1
                diagnostics['correction_target_ce_sum'] += per_query[query]
                diagnostics['correction_untouched_ce_sum'] += sum(v for q, v in enumerate(per_query[:4]) if q != query)
                diagnostics['correction_absent_ce_sum'] += sum(per_query[4:])
            accumulated.add_(gradient, alpha=1 / count)
        ce_gradients.append(accumulated)
    mse = (torch.stack([s.values for s in states], dim=1) - targets.values[batch.after_codes]).square().mean()
    ce_parameters = torch.autograd.grad(
        tuple(s.values for s in states), parameters, grad_outputs=tuple(ce_gradients), retain_graph=True,
    )
    mse_parameters = torch.autograd.grad(mse, parameters)
    if any(not torch.isfinite(g).all() for g in (*ce_parameters, *mse_parameters)):
        raise ValueError('nonfinite updater gradient')
    ce_norm = torch.stack([g.square().sum() for g in ce_parameters]).sum().sqrt()
    mse_norm = torch.stack([g.square().sum() for g in mse_parameters]).sum().sqrt()
    dot = torch.stack([(a*b).sum() for a,b in zip(ce_parameters, mse_parameters, strict=True)]).sum()
    for parameter, ce_gradient, mse_gradient in zip(parameters, ce_parameters, mse_parameters, strict=True):
        parameter.grad = ce_gradient + state_weight * mse_gradient
    ce = sum(losses) / count
    objective = ce + state_weight * float(mse.detach())
    if not math.isfinite(objective):
        raise ValueError('nonfinite joint objective')
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    if not measure_only:
        optimizer.step()
    if any(not torch.isfinite(p).all() for p in parameters):
        raise ValueError('optimizer produced nonfinite parameters')
    if (not torch.equal(targets.values, teacher_values) or not torch.equal(targets.valid, teacher_valid)
            or any(not torch.equal(getattr(batch,k),v) for k,v in snapshots.items())):
        raise ValueError('training input or frozen teacher mutated')
    if any(p.grad is not None for p in (*reader.model.parameters(), *bridge.parameters())):
        raise ValueError('frozen reader or bridge received gradients')
    return {
        **diagnostics, 'correction_multiplier': float(correction_multiplier),
        'unweighted_answer_ce': sum(unweighted_losses) / count,
        'answer_ce': ce, 'learned_state_mse': float(mse.detach()), 'state_weight': float(state_weight),
        'objective': objective, 'answer_parameter_gradient_norm': float(ce_norm),
        'state_parameter_gradient_norm': float(mse_norm), 'parameter_gradient_dot': float(dot),
        'gradient_norm': float(norm), 'gradient_clipped': bool(float(norm)>1),
        'initial_encoder_examples': 0, 'event_examples': count, 'answer_sequences': count*6,
        'target_values': count*16, 'optimizer_steps': int(not measure_only),
    }
