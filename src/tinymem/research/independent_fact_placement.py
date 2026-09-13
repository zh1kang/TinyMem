"""Placement variants for the fixed independent-fact readout."""

from collections.abc import Sequence

import torch
from torch import nn

from tinymem.memory.readout_interface import STATE_BYTES, ReadoutBridge, check_readout_state
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.independent_fact_training import train_fact_step
from tinymem.research.prefix_reader import prefix_answer_loss
from tinymem.research.pretrained import PretrainedReader
from tinymem.research.readout_runner import ReadoutQuery


_LAYOUTS = ("control", "separate_fact0")


def _check_layout(layout: str) -> None:
    if type(layout) is not str or layout not in _LAYOUTS:
        raise ValueError("layout must be control or separate_fact0")


def _check_code(code: int) -> None:
    if type(code) is not int or not 0 <= code < 16:
        raise ValueError("code must be an integer from zero through fifteen")


def placement_state(code: int, device: torch.device, layout: str) -> LatentSlotState:
    """Construct the fixed 66-byte state for one placement layout."""
    _check_code(code)
    _check_layout(layout)
    values = torch.zeros(1, 2, 8, dtype=torch.float32, device=device)
    if layout == "control":
        values[0, 0, :4] = torch.tensor(
            [1 if code & (1 << bit) else -1 for bit in range(4)],
            dtype=torch.float32,
            device=device,
        )
    else:
        values[0, 0, 0] = 1 if code & 1 else -1
        values[0, 1, 1:4] = torch.tensor(
            [1 if code & (1 << bit) else -1 for bit in range(1, 4)],
            dtype=torch.float32,
            device=device,
        )
    return LatentSlotState(values, torch.ones(1, 2, dtype=torch.bool, device=device))


def check_placement_state(state: LatentSlotState, code: int, layout: str) -> None:
    """Require the exact immutable payload for a placement and code."""
    _check_code(code)
    _check_layout(layout)
    check_readout_state(state)
    if state.values.requires_grad or state.values.grad_fn is not None or not state.valid.all():
        raise ValueError("placement state must be fixed and both slots occupied")
    expected = placement_state(code, state.values.device, layout)
    if not torch.equal(state.values, expected.values) or not torch.equal(state.valid, expected.valid):
        raise ValueError("state does not equal the declared placement and code")


def _validate_training_inputs(
    reader: PretrainedReader,
    bridge: ReadoutBridge,
    state: LatentSlotState,
    code: int,
    queries: Sequence[ReadoutQuery],
    optimizer: torch.optim.Optimizer,
    adapter_parameters: tuple[nn.Parameter, ...],
    layout: str,
) -> tuple[tuple[nn.Parameter, ...], tuple[nn.Parameter, ...], torch.Tensor, torch.Tensor]:
    check_placement_state(state, code, layout)
    if any(module.training for module in reader.model.modules()):
        raise ValueError("read-side model must be in evaluation mode")
    reader_owned = {id(parameter) for parameter in adapter_parameters}
    if (
        len(reader_owned) != len(adapter_parameters)
        or reader_owned != {id(parameter) for parameter in reader.model.parameters() if parameter.requires_grad}
    ):
        raise ValueError("reader adapter ownership changed")
    bridge_parameters = tuple(bridge.parameters())
    parameters = (*bridge_parameters, *adapter_parameters)
    owned = {id(parameter) for parameter in parameters}
    optimized = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if (
        len(owned) != len(parameters)
        or len(optimized) != len(owned)
        or {id(parameter) for parameter in optimized} != owned
        or any(not parameter.requires_grad for parameter in parameters)
    ):
        raise ValueError("optimizer must own exactly bridge and declared read adapter")
    if len(queries) != 6:
        raise ValueError("exactly six independent fact queries are required")
    if any(
        parameter.grad is not None
        for parameter in reader.model.parameters()
        if id(parameter) not in reader_owned
    ):
        raise ValueError("frozen reader has unexpected gradients")
    original_values = state.values.clone()
    original_valid = state.valid.clone()
    return bridge_parameters, parameters, original_values, original_valid


def train_placement_step(
    reader: PretrainedReader,
    bridge: ReadoutBridge,
    state: LatentSlotState,
    code: int,
    before_ids: tuple[int, ...],
    queries: Sequence[ReadoutQuery],
    optimizer: torch.optim.Optimizer,
    *,
    adapter_parameters: tuple[nn.Parameter, ...],
    layout: str,
) -> dict[str, float | int]:
    """Train one placement state using the fixed independent-fact objective."""
    check_placement_state(state, code, layout)
    if layout == "control":
        return train_fact_step(
            reader,
            bridge,
            state,
            code,
            before_ids,
            queries,
            optimizer,
            adapter_parameters=adapter_parameters,
        )

    bridge_parameters, parameters, original_values, original_valid = _validate_training_inputs(
        reader, bridge, state, code, queries, optimizer, adapter_parameters, layout,
    )
    optimizer.zero_grad(set_to_none=True)
    device = reader.model.device
    memory = bridge(state)
    before = torch.tensor(before_ids, device=device)
    loss = torch.stack([
        prefix_answer_loss(
            reader,
            before,
            memory,
            torch.tensor(query.after_ids, device=device),
            torch.tensor(query.answer_ids, device=device),
        )
        for query in queries
    ]).mean()
    if not torch.isfinite(loss):
        raise ValueError("nonfinite answer loss")
    loss.backward()
    if any(parameter.grad is None for parameter in parameters):
        raise ValueError("all declared parameters must receive gradients")
    if any(
        parameter.grad is not None
        for parameter in reader.model.parameters()
        if id(parameter) not in {id(item) for item in adapter_parameters}
    ):
        raise ValueError("frozen reader gradient ownership violation")

    gradient_norms = {}
    for name, group in (("bridge", bridge_parameters), ("adapter", adapter_parameters)):
        value = (
            torch.stack([parameter.grad.float().square().sum() for parameter in group]).sum().sqrt()
            if group
            else loss.new_zeros(())
        )
        if group and (not torch.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must receive finite nonzero gradients")
        gradient_norms[name + "_gradient_norm"] = float(value)
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    if not torch.isfinite(norm):
        raise ValueError("nonfinite gradient norm")
    result = {
        "answer_ce": float(loss.detach()),
        "gradient_norm": float(norm),
        "persistent_bytes": STATE_BYTES,
        "write_states": 0,
        "supervised_tokens": sum(len(query.answer_ids) for query in queries),
        "code": code,
        **gradient_norms,
    }
    optimizer.step()
    if any(not torch.isfinite(parameter).all() for parameter in parameters):
        raise ValueError("optimizer produced nonfinite parameters")
    if not torch.equal(state.values, original_values) or not torch.equal(state.valid, original_valid):
        raise ValueError("fixed placement state changed during training")
    return result
