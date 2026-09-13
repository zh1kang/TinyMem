"""Matched reset and recurrent training for fixed fact update streams."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from tinymem.memory.query_pool_slots import QueryPoolSlotWriter
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.independent_fact_data import ENTITIES, ROOM_PAIRS


@dataclass(frozen=True)
class RecurrentBatch:
    """Detached CPU inputs for one matched stream batch."""

    histories: torch.Tensor
    history_valid: torch.Tensor
    initial_codes: torch.Tensor
    initial_targets: torch.Tensor
    events: torch.Tensor
    event_valid: torch.Tensor
    before_codes: torch.Tensor
    after_codes: torch.Tensor
    before_targets: torch.Tensor
    after_targets: torch.Tensor
    stream_ids: tuple[str, ...]


def _literal_values(code: int) -> torch.Tensor:
    values = torch.zeros(2, 8, dtype=torch.float32)
    values[0, 0] = 1.0 if code & 1 else -1.0
    values[1, 1:4] = torch.tensor(
        [1.0 if code & (1 << fact) else -1.0 for fact in range(1, 4)],
    )
    return values


def _copy_features(
    features: Mapping[int, torch.Tensor], width: int,
) -> dict[int, torch.Tensor]:
    if not isinstance(features, Mapping) or set(features) != set(range(16)):
        raise ValueError("history features must cover codes zero through fifteen")
    copied: dict[int, torch.Tensor] = {}
    for code, feature in features.items():
        if (
            type(code) is not int
            or not isinstance(feature, torch.Tensor)
            or feature.ndim != 2
            or feature.shape[0] == 0
            or feature.shape[1] != width
            or feature.device.type != "cpu"
            or feature.dtype != torch.float32
            or feature.requires_grad
            or not torch.isfinite(feature).all()
        ):
            raise ValueError("history features must be detached CPU FP32 token matrices")
        copied[code] = feature.detach().clone()
    return copied


def _copy_event_features(
    features: Mapping[tuple[int, int], torch.Tensor], width: int,
) -> dict[tuple[int, int], torch.Tensor]:
    expected = {(fact, bit) for fact in range(4) for bit in range(2)}
    if not isinstance(features, Mapping) or set(features) != expected:
        raise ValueError("event features must cover every fact and value")
    copied: dict[tuple[int, int], torch.Tensor] = {}
    for key, feature in features.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or type(key[0]) is not int
            or type(key[1]) is not int
            or not isinstance(feature, torch.Tensor)
            or feature.ndim != 2
            or feature.shape[0] == 0
            or feature.shape[1] != width
            or feature.device.type != "cpu"
            or feature.dtype != torch.float32
            or feature.requires_grad
            or not torch.isfinite(feature).all()
        ):
            raise ValueError("event features must be detached CPU FP32 token matrices")
        copied[key] = feature.detach().clone()
    return copied


def _validate_stream(stream: dict, index: int) -> tuple[str, int, list[dict]]:
    if not isinstance(stream, dict):
        raise ValueError(f"stream {index} must be a mapping")
    stream_id = stream.get("id")
    initial_code = stream.get("initial_code")
    events = stream.get("events")
    if not isinstance(stream_id, str) or not stream_id:
        raise ValueError("stream id must be a nonempty string")
    if type(initial_code) is not int or not 0 <= initial_code < 16:
        raise ValueError("stream initial_code must be an integer from zero through fifteen")
    if not isinstance(events, Sequence) or len(events) != 4:
        raise ValueError("stream must contain exactly four events")
    current = initial_code
    for position, event in enumerate(events, start=1):
        if not isinstance(event, Mapping):
            raise ValueError("stream event must be a mapping")
        required = {"step", "before_code", "after_code", "target_fact", "new_bit", "action", "text"}
        if not required.issubset(event):
            raise ValueError("stream event is missing required fields")
        step = event["step"]
        before = event["before_code"]
        fact = event["target_fact"]
        new_bit = event["new_bit"]
        action = event["action"]
        if (
            type(step) is not int
            or step != position
            or type(before) is not int
            or before != current
            or type(event["after_code"]) is not int
            or type(fact) is not int
            or fact not in range(4)
            or type(new_bit) is not int
            or new_bit not in (0, 1)
            or action not in ("C", "R")
            or not isinstance(event["text"], str)
        ):
            raise ValueError("stream event fields are inconsistent")
        old_bit = (current >> fact) & 1
        expected_bit = old_bit if action == "R" else 1 - old_bit
        expected_after = (current & ~(1 << fact)) | (expected_bit << fact)
        expected_text = f"{ENTITIES[fact]} moved to the {ROOM_PAIRS[fact][expected_bit]}."
        if (
            new_bit != expected_bit
            or event["after_code"] != expected_after
            or event["text"] != expected_text
        ):
            raise ValueError("stream event truth or text is inconsistent")
        current = expected_after
    return stream_id, initial_code, list(events)


def pack_recurrent_batch(
    streams: list[dict],
    history_features: Mapping[int, torch.Tensor],
    event_features: Mapping[tuple[int, int], torch.Tensor],
) -> RecurrentBatch:
    """Pack four-event streams with detached CPU FP32 features and literal targets."""
    if not isinstance(streams, list) or not streams:
        raise ValueError("streams must be a nonempty list")
    parsed = [_validate_stream(stream, index) for index, stream in enumerate(streams)]
    stream_ids = tuple(item[0] for item in parsed)
    if len(set(stream_ids)) != len(stream_ids):
        raise ValueError("stream ids must be unique")

    if not isinstance(history_features, Mapping) or not history_features:
        raise ValueError("history features must be nonempty")
    first_history = next(iter(history_features.values()))
    if not isinstance(first_history, torch.Tensor) or first_history.ndim != 2 or first_history.shape[1] == 0:
        raise ValueError("history features must be token matrices")
    width = first_history.shape[1]
    histories_copy = _copy_features(history_features, width)
    events_copy = _copy_event_features(event_features, width)
    max_history = max(feature.shape[0] for feature in histories_copy.values())
    max_event = max(feature.shape[0] for feature in events_copy.values())
    histories = torch.zeros(16, max_history, width, dtype=torch.float32)
    history_valid = torch.zeros(16, max_history, dtype=torch.bool)
    for code, feature in histories_copy.items():
        histories[code, : feature.shape[0]] = feature
        history_valid[code, : feature.shape[0]] = True

    batch_size = len(parsed)
    events = torch.zeros(batch_size, 4, max_event, width, dtype=torch.float32)
    event_valid = torch.zeros(batch_size, 4, max_event, dtype=torch.bool)
    initial_codes = torch.tensor([item[1] for item in parsed], dtype=torch.long)
    before_codes = torch.empty(batch_size, 4, dtype=torch.long)
    after_codes = torch.empty(batch_size, 4, dtype=torch.long)
    before_targets = torch.empty(batch_size, 4, 2, 8, dtype=torch.float32)
    after_targets = torch.empty_like(before_targets)
    for row, (_, _, stream_events) in enumerate(parsed):
        for step, event in enumerate(stream_events):
            fact, bit = event["target_fact"], event["new_bit"]
            feature = events_copy[(fact, bit)]
            events[row, step, : feature.shape[0]] = feature
            event_valid[row, step, : feature.shape[0]] = True
            before = event["before_code"]
            after = event["after_code"]
            before_codes[row, step] = before
            after_codes[row, step] = after
            before_targets[row, step] = _literal_values(before)
            after_targets[row, step] = _literal_values(after)

    initial_targets = torch.stack([_literal_values(code) for code in range(16)])
    return RecurrentBatch(
        histories=histories,
        history_valid=history_valid,
        initial_codes=initial_codes,
        initial_targets=initial_targets,
        events=events,
        event_valid=event_valid,
        before_codes=before_codes,
        after_codes=after_codes,
        before_targets=before_targets,
        after_targets=after_targets,
        stream_ids=stream_ids,
    )


def _validate_batch(batch: RecurrentBatch, writer: QueryPoolSlotWriter) -> None:
    if not isinstance(batch, RecurrentBatch):
        raise TypeError("batch must be a RecurrentBatch")
    if batch.histories.shape[0] != 16 or batch.histories.ndim != 3:
        raise ValueError("batch histories must contain sixteen worlds")
    batch_size = batch.events.shape[0]
    if batch_size == 0 or batch.events.ndim != 4 or batch.events.shape[1] != 4:
        raise ValueError("batch events must have shape [batch, four, tokens, width]")
    if len(batch.stream_ids) != batch_size or len(set(batch.stream_ids)) != batch_size:
        raise ValueError("batch stream ids must match the batch")
    expected = (
        batch.history_valid.shape == batch.histories.shape[:2]
        and batch.event_valid.shape == batch.events.shape[:3]
        and batch.initial_codes.shape == (batch_size,)
        and batch.before_codes.shape == (batch_size, 4)
        and batch.after_codes.shape == (batch_size, 4)
        and batch.initial_targets.shape == (16, 2, writer.memory_width)
        and batch.before_targets.shape == (batch_size, 4, 2, writer.memory_width)
        and batch.after_targets.shape == batch.before_targets.shape
        and batch.histories.shape[-1] == writer.reader_width
        and batch.events.shape[-1] == writer.reader_width
        and batch.history_valid.dtype == torch.bool
        and batch.event_valid.dtype == torch.bool
        and batch.initial_codes.dtype == torch.long
        and batch.before_codes.dtype == torch.long
        and batch.after_codes.dtype == torch.long
        and all(tensor.device.type == "cpu" for tensor in batch.__dict__.values() if isinstance(tensor, torch.Tensor))
        and all(tensor.dtype == torch.float32 for tensor in (
            batch.histories, batch.events, batch.initial_targets,
            batch.before_targets, batch.after_targets,
        ))
        and all(not tensor.requires_grad for tensor in batch.__dict__.values() if isinstance(tensor, torch.Tensor))
        and batch.history_valid.any(dim=1).all()
        and batch.event_valid.any(dim=2).all()
        and bool(((batch.initial_codes >= 0) & (batch.initial_codes < 16)).all())
        and bool(((batch.before_codes >= 0) & (batch.before_codes < 16)).all())
        and bool(((batch.after_codes >= 0) & (batch.after_codes < 16)).all())
    )
    if not expected:
        raise ValueError("recurrent batch tensors have incompatible shapes or dtypes")
    floating = (batch.histories, batch.events, batch.initial_targets, batch.before_targets, batch.after_targets)
    if not all(torch.isfinite(tensor).all() for tensor in floating):
        raise ValueError("recurrent batch contains nonfinite values")


def _writer_parameters(writer: QueryPoolSlotWriter, optimizer: torch.optim.Optimizer) -> tuple[torch.Tensor, ...]:
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


def _check_state(state: LatentSlotState, count: int, writer: QueryPoolSlotWriter) -> None:
    if (
        state.values.shape != (count, writer.slots, writer.memory_width)
        or state.values.dtype != torch.float32
        or state.valid.shape != (count, writer.slots)
        or not state.valid.all()
        or not torch.isfinite(state.values).all()
        or bool((state.values.abs() > 1).any())
        or state.nbytes != count * 66
    ):
        raise ValueError("writer output must retain finite two-slot 66-byte states")


def train_recurrent_batch(
    writer: QueryPoolSlotWriter,
    batch: RecurrentBatch,
    optimizer: torch.optim.Optimizer,
    *,
    arm: str,
) -> dict[str, float | int | bool]:
    """Train one matched stream batch using the reset or recurrent state flow."""
    if arm not in ("reset", "recurrent"):
        raise ValueError("arm must be 'reset' or 'recurrent'")
    if writer.queries.device.type != "cpu":
        raise ValueError("writer must be on the CPU")
    _validate_batch(batch, writer)
    parameters = _writer_parameters(writer, optimizer)
    snapshots = tuple(
        tensor.clone() for tensor in batch.__dict__.values() if isinstance(tensor, torch.Tensor)
    )
    histories = batch.histories
    history_valid = batch.history_valid
    events = batch.events
    event_valid = batch.event_valid
    initial_targets = batch.initial_targets
    after_targets = batch.after_targets
    initial_codes = batch.initial_codes
    before_codes = batch.before_codes

    optimizer.zero_grad(set_to_none=True)
    initial = writer(writer.empty(16), histories, history_valid)
    _check_state(initial, 16, writer)
    initial_loss = F.mse_loss(initial.values, initial_targets)

    if arm == "reset":
        outputs = []
        for step in range(4):
            state = LatentSlotState(
                initial.values.index_select(0, before_codes[:, step]),
                initial.valid.index_select(0, before_codes[:, step]),
            )
            updated = writer(state, events[:, step], event_valid[:, step])
            _check_state(updated, events.shape[0], writer)
            outputs.append(updated.values)
        update_values = torch.stack(outputs, dim=1)
    else:
        state = LatentSlotState(
            initial.values.index_select(0, initial_codes),
            initial.valid.index_select(0, initial_codes),
        )
        outputs = []
        for step in range(4):
            state = writer(state, events[:, step], event_valid[:, step])
            _check_state(state, events.shape[0], writer)
            outputs.append(state.values)
        update_values = torch.stack(outputs, dim=1)

    update_loss = F.mse_loss(update_values, after_targets)
    loss = 0.5 * initial_loss + 0.5 * update_loss
    if not all(torch.isfinite(value) for value in (initial_loss, update_loss, loss)):
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
        (tensor for tensor in batch.__dict__.values() if isinstance(tensor, torch.Tensor)),
        snapshots,
        strict=True,
    )):
        raise ValueError("recurrent batch was mutated")
    return {
        "initial_state_mse": float(initial_loss.detach()),
        "updated_state_mse": float(update_loss.detach()),
        "state_mse": float(loss.detach()),
        "gradient_norm": float(norm),
        "gradient_clipped": bool(float(norm) > 1.0),
        "history_examples": 16,
        "event_examples": events.shape[0] * 4,
        "writer_examples": 16 + events.shape[0] * 4,
    }
