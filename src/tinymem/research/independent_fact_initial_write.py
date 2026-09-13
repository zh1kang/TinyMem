"""Joint initial-write and conditional-update training for fixed fact states."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from tinymem.memory.query_pool_slots import QueryPoolSlotWriter
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.independent_fact_updates import UpdateCase, pack_update_batch


@dataclass(frozen=True)
class InitialUpdateBatch:
    """Detached CPU inputs and coordinate targets for one initial/update batch."""

    histories: torch.Tensor
    history_valid: torch.Tensor
    events: torch.Tensor
    event_valid: torch.Tensor
    before_targets: torch.Tensor
    after_targets: torch.Tensor


def _validate_history_features(
    features: Mapping[int, torch.Tensor], reader_width: int,
) -> dict[int, torch.Tensor]:
    if not isinstance(features, Mapping):
        raise TypeError("history_features must be a mapping")
    if any(type(code) is not int for code in features) or set(features) != set(range(16)):
        raise ValueError("history features must cover codes zero through fifteen")
    result: dict[int, torch.Tensor] = {}
    for code, value in features.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 2
            or value.shape[0] == 0
            or value.shape[1] != reader_width
            or value.device.type != "cpu"
            or value.dtype != torch.float32
            or value.requires_grad
            or not torch.isfinite(value).all()
        ):
            raise ValueError("history features must be detached CPU FP32 token matrices")
        result[code] = value.detach().clone()
    return result


def pack_initial_update_batch(
    cases: Sequence[UpdateCase],
    history_features: Mapping[int, torch.Tensor],
    event_features: Mapping[tuple[int, int], torch.Tensor],
) -> InitialUpdateBatch:
    """Pack selected histories and events with detached coordinate targets."""
    old, events, event_valid, after_targets = pack_update_batch(cases, event_features)
    width = events.shape[-1]
    history_copy = _validate_history_features(history_features, width)
    maximum = max(history_copy[case.before_code].shape[0] for case in cases)
    histories = torch.zeros(len(cases), maximum, width, dtype=torch.float32)
    history_valid = torch.zeros(len(cases), maximum, dtype=torch.bool)
    for row, case in enumerate(cases):
        history = history_copy[case.before_code]
        histories[row, : history.shape[0]] = history
        history_valid[row, : history.shape[0]] = True
    return InitialUpdateBatch(
        histories=histories,
        history_valid=history_valid,
        events=events.detach().clone(),
        event_valid=event_valid.detach().clone(),
        before_targets=old.values.detach().clone(),
        after_targets=after_targets.detach().clone(),
    )


def _validate_batch(batch: InitialUpdateBatch, writer: QueryPoolSlotWriter) -> None:
    if not isinstance(batch, InitialUpdateBatch):
        raise TypeError("batch must be an InitialUpdateBatch")
    tensors = (
        batch.histories, batch.history_valid, batch.events, batch.event_valid,
        batch.before_targets, batch.after_targets,
    )
    if (
        batch.histories.ndim != 3
        or batch.events.ndim != 3
        or batch.histories.shape[0] == 0
        or batch.events.shape[0] != batch.histories.shape[0]
        or batch.histories.shape[-1] != writer.reader_width
        or batch.events.shape[-1] != writer.reader_width
        or batch.history_valid.shape != batch.histories.shape[:2]
        or batch.event_valid.shape != batch.events.shape[:2]
        or batch.history_valid.dtype != torch.bool
        or batch.event_valid.dtype != torch.bool
        or batch.before_targets.shape != (
            batch.histories.shape[0], writer.slots, writer.memory_width,
        )
        or batch.after_targets.shape != batch.before_targets.shape
        or any(tensor.device.type != "cpu" for tensor in tensors)
        or any(tensor.dtype != torch.float32 for tensor in (batch.histories, batch.events,
                                                             batch.before_targets, batch.after_targets))
        or any(tensor.requires_grad for tensor in (batch.histories, batch.events,
                                                   batch.before_targets, batch.after_targets))
        or not batch.history_valid.any(dim=1).all()
        or not batch.event_valid.any(dim=1).all()
        or not torch.isfinite(batch.histories).all()
        or not torch.isfinite(batch.events).all()
        or not torch.isfinite(batch.before_targets).all()
        or not torch.isfinite(batch.after_targets).all()
    ):
        raise ValueError("initial update batch tensors are invalid")


def _validate_writer_ownership(writer: QueryPoolSlotWriter, optimizer: torch.optim.Optimizer) -> tuple:
    parameters = tuple(writer.parameters())
    optimized = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    owned = {id(parameter) for parameter in parameters}
    if (
        len(owned) != len(parameters)
        or len(optimized) != len(owned)
        or {id(parameter) for parameter in optimized} != owned
        or any(not parameter.requires_grad for parameter in parameters)
    ):
        raise ValueError("optimizer must own exactly all writer parameters")
    return parameters


def _check_state(state: LatentSlotState, batch_size: int) -> None:
    if (
        not state.valid.all()
        or state.values.shape != (batch_size, 2, 8)
        or state.values.dtype != torch.float32
        or not torch.isfinite(state.values).all()
        or bool((state.values.abs() > 1).any())
        or state.nbytes != batch_size * 66
    ):
        raise ValueError("writer output must retain finite two-slot 66-byte states")


def train_initial_update_batch(
    writer: QueryPoolSlotWriter,
    batch: InitialUpdateBatch,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Train one writer step through an initial write followed by an update."""
    _validate_batch(batch, writer)
    parameters = _validate_writer_ownership(writer, optimizer)
    snapshots = tuple(tensor.clone() for tensor in (
        batch.histories, batch.history_valid, batch.events, batch.event_valid,
        batch.before_targets, batch.after_targets,
    ))
    optimizer.zero_grad(set_to_none=True)
    initial = writer(writer.empty(batch.histories.shape[0]), batch.histories, batch.history_valid)
    updated = writer(initial, batch.events, batch.event_valid)
    _check_state(initial, batch.histories.shape[0])
    _check_state(updated, batch.histories.shape[0])
    initial_loss = F.mse_loss(initial.values, batch.before_targets)
    updated_loss = F.mse_loss(updated.values, batch.after_targets)
    loss = 0.5 * initial_loss + 0.5 * updated_loss
    if not all(torch.isfinite(value) for value in (initial_loss, updated_loss, loss)):
        raise ValueError("nonfinite initial/update state MSE")
    loss.backward()
    if any(parameter.grad is None for parameter in parameters):
        raise ValueError("all writer parameters must receive gradients")
    if any(not torch.isfinite(parameter.grad).all() for parameter in parameters):
        raise ValueError("nonfinite writer gradients")
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    if not torch.isfinite(norm):
        raise ValueError("nonfinite writer gradient norm")
    optimizer.step()
    if any(not torch.isfinite(parameter).all() for parameter in parameters):
        raise ValueError("optimizer produced nonfinite writer parameters")
    if any(not torch.equal(actual, snapshot) for actual, snapshot in zip(
        (batch.histories, batch.history_valid, batch.events, batch.event_valid,
         batch.before_targets, batch.after_targets), snapshots, strict=True,
    )):
        raise ValueError("initial update batch was mutated")
    return {
        "initial_state_mse": float(initial_loss.detach()),
        "updated_state_mse": float(updated_loss.detach()),
        "state_mse": float(loss.detach()),
        "gradient_norm": float(norm),
    }
