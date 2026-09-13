"""Bounded training for the fixed independent-fact state readout."""

from collections.abc import Sequence

import torch
from torch import nn

from tinymem.memory.readout_interface import STATE_BYTES, ReadoutBridge, check_readout_state
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.independent_fact_data import oracle_state
from tinymem.research.prefix_reader import prefix_answer_loss
from tinymem.research.pretrained import PretrainedReader
from tinymem.research.readout_runner import ReadoutQuery


def check_fact_state(state: LatentSlotState, code: int) -> None:
    """Require the immutable four-bit state for one independent-fact code."""
    if type(code) is not int or not 0 <= code < 16:
        raise ValueError("code must be an integer from zero through fifteen")
    check_readout_state(state)
    if state.values.requires_grad or state.values.grad_fn is not None or not state.valid.all():
        raise ValueError("independent fact state must be fixed and both slots occupied")
    expected = oracle_state(code, state.values.device)
    if not torch.equal(state.values, expected.values) or not torch.equal(state.valid, expected.valid):
        raise ValueError("state does not equal the declared independent fact code")


def train_fact_step(
    reader: PretrainedReader,
    bridge: ReadoutBridge,
    state: LatentSlotState,
    code: int,
    before_ids: tuple[int, ...],
    queries: Sequence[ReadoutQuery],
    optimizer: torch.optim.Optimizer,
    *,
    adapter_parameters: tuple[nn.Parameter, ...],
) -> dict[str, float | int]:
    """Train one fixed independent-fact payload against its six answer queries."""
    check_fact_state(state, code)
    if any(module.training for module in reader.model.modules()):
        raise ValueError("read-side model must be in evaluation mode")
    reader_owned = {id(parameter) for parameter in adapter_parameters}
    if (len(reader_owned) != len(adapter_parameters)
            or reader_owned != {id(parameter) for parameter in reader.model.parameters()
                                if parameter.requires_grad}):
        raise ValueError("reader adapter ownership changed")
    bridge_parameters = tuple(bridge.parameters())
    parameters = [*bridge_parameters, *adapter_parameters]
    owned = {id(parameter) for parameter in parameters}
    optimized = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if (len(owned) != len(parameters) or len(optimized) != len(owned)
            or {id(parameter) for parameter in optimized} != owned
            or any(not parameter.requires_grad for parameter in parameters)):
        raise ValueError("optimizer must own exactly bridge and declared read adapter")
    if len(queries) != 6:
        raise ValueError("exactly six independent fact queries are required")
    if any(parameter.grad is not None for parameter in reader.model.parameters()
           if id(parameter) not in reader_owned):
        raise ValueError("frozen reader has unexpected gradients")

    original_values = state.values.clone()
    original_valid = state.valid.clone()
    optimizer.zero_grad(set_to_none=True)
    device = reader.model.device
    memory = bridge(state)
    before = torch.tensor(before_ids, device=device)
    loss = torch.stack([
        prefix_answer_loss(reader, before, memory, torch.tensor(query.after_ids, device=device),
                           torch.tensor(query.answer_ids, device=device))
        for query in queries
    ]).mean()
    if not torch.isfinite(loss):
        raise ValueError("nonfinite answer loss")
    loss.backward()
    if any(parameter.grad is None for parameter in parameters):
        raise ValueError("all declared parameters must receive gradients")
    if any(parameter.grad is not None for parameter in reader.model.parameters()
           if id(parameter) not in reader_owned):
        raise ValueError("frozen reader gradient ownership violation")

    gradient_norms = {}
    for name, group in (("bridge", bridge_parameters), ("adapter", adapter_parameters)):
        value = (torch.stack([parameter.grad.float().square().sum() for parameter in group]).sum().sqrt()
                 if group else loss.new_zeros(()))
        if group and (not torch.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must receive finite nonzero gradients")
        gradient_norms[name + "_gradient_norm"] = float(value)
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    if not torch.isfinite(norm):
        raise ValueError("nonfinite gradient norm")
    result = {
        "answer_ce": float(loss.detach()), "gradient_norm": float(norm),
        "persistent_bytes": STATE_BYTES, "write_states": 0,
        "supervised_tokens": sum(len(query.answer_ids) for query in queries),
        "code": code, **gradient_norms,
    }
    optimizer.step()
    if any(not torch.isfinite(parameter).all() for parameter in parameters):
        raise ValueError("optimizer produced nonfinite parameters")
    if not torch.equal(state.values, original_values) or not torch.equal(state.valid, original_valid):
        raise ValueError("fixed independent fact state changed during training")
    return result
