"""Readout training over the privileged known-correct oracle state."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.studies.delta.data import Episode, replay
from tinymem.studies.delta.encoding import _queries
from tinymem.studies.delta.readout import SlotReadout
from tinymem.studies.delta.evaluation import StateRecord
from tinymem.studies.oracle.state import (
    ORACLE_MEMORY_WIDTH,
    ORACLE_STATE_BYTES,
    oracle_bits,
    oracle_records,
)
from tinymem.reader.prefix import prefix_answer_losses
from tinymem.reader.pretrained import PretrainedReader
from tinymem.reader.adapter import ReadoutQuery


@dataclass(frozen=True)
class OracleEndpoint:
    after_write: int
    state: LatentSlotState
    queries: tuple[ReadoutQuery, ...]


@dataclass(frozen=True)
class OracleExample:
    episode_id: str
    split: str
    before_ids: tuple[int, ...]
    endpoints: tuple[OracleEndpoint, ...]


def _validate_episode(episode: Episode) -> None:
    if not isinstance(episode, Episode):
        raise TypeError("episode must be an Episode")
    if len(episode.prefix) != 8 or len(episode.tail) not in (0, 8):
        raise ValueError("episodes must contain an eight-statement prefix and optional tail")


def _owned_state(record: StateRecord) -> LatentSlotState:
    if not isinstance(record, StateRecord):
        raise TypeError("records must contain StateRecord values")
    state = LatentSlotState(record.values.detach().clone().contiguous(),
                            record.valid.detach().clone().contiguous())
    if oracle_bits(state) != tuple(record.truth):
        raise ValueError("oracle record does not decode to its independent truth")
    return state


def _record_map(records: Sequence[StateRecord], episode: Episode) -> dict[int, LatentSlotState]:
    result: dict[int, LatentSlotState] = {}
    for record in records:
        if record.episode_id != episode.id:
            continue
        if record.after_write in result:
            raise ValueError("duplicate oracle endpoint record")
        result[record.after_write] = _owned_state(record)
    expected = {8, 9, 16} if episode.tail else {8}
    if not expected.issubset(result):
        raise ValueError("oracle records do not cover the episode endpoints")
    return result


def encode_oracle_episode(
    reader: PretrainedReader,
    episode: Episode,
    records: Sequence[StateRecord] | None = None,
) -> OracleExample:
    """Create native read queries paired with a query-blind oracle state."""

    _validate_episode(episode)
    endpoint_states = _record_map(records, episode) if records is not None else _record_map(
        oracle_records((episode,)), episode
    )
    statements = (*episode.prefix, *episode.tail)
    endpoints: list[OracleEndpoint] = []
    before_ids: tuple[int, ...] | None = None
    for after_write in (8, 16) if episode.tail else (8,):
        if oracle_bits(endpoint_states[after_write]) != replay(statements[:after_write]):
            raise ValueError("oracle endpoint state disagrees with episode truth")
        native_before, queries = _queries(reader, episode, statements[:after_write], after_write)
        if before_ids is None:
            before_ids = native_before
        elif native_before != before_ids:
            raise ValueError("native endpoints do not share a stable before prompt")
        if endpoints and tuple(q.after_ids for q in queries) != tuple(q.after_ids for q in endpoints[0].queries):
            raise ValueError("native question tokens changed with the history endpoint")
        endpoints.append(OracleEndpoint(after_write, endpoint_states[after_write], queries))
    if before_ids is None:
        raise ValueError("episode must produce an endpoint")
    return OracleExample(episode.id, episode.split, before_ids, tuple(endpoints))


def _validate_example(example: OracleExample, reader_width: int) -> None:
    if not isinstance(example, OracleExample):
        raise TypeError("examples must contain OracleExample values")
    if example.split != "train":
        raise ValueError("optimization accepts training examples only")
    if not example.episode_id or not example.before_ids or not example.endpoints:
        raise ValueError("oracle examples require an identity, prompt, and endpoints")
    positions = [endpoint.after_write for endpoint in example.endpoints]
    if positions not in ([8], [8, 16]):
        raise ValueError("oracle endpoints must be [8] or [8, 16]")
    for endpoint in example.endpoints:
        if not endpoint.queries:
            raise ValueError("oracle endpoints require queries")
        oracle_bits(endpoint.state)
        if endpoint.state.nbytes != ORACLE_STATE_BYTES:
            raise ValueError("oracle state bytes differ from the declared control")
        for query in endpoint.queries:
            if (not query.after_ids or not query.answer_ids
                    or any(type(token) is not int for token in (*query.after_ids, *query.answer_ids))):
                raise ValueError("oracle query token IDs must be nonempty integer tuples")
    if type(reader_width) is not int or reader_width <= 0:
        raise ValueError("reader width must be positive")


def _validate_modules(
    reader: PretrainedReader, bridge: SlotReadout, adapter_parameters: tuple[nn.Parameter, ...],
) -> tuple[nn.Parameter, ...]:
    if not isinstance(bridge, SlotReadout):
        raise TypeError("bridge must be a SlotReadout")
    if bridge.memory_width != ORACLE_MEMORY_WIDTH or bridge.state_bytes != ORACLE_STATE_BYTES:
        raise ValueError("oracle bridge must use the declared 258-byte state")
    if bridge.reader_width != reader.model.config.hidden_size:
        raise ValueError("bridge reader width differs from the reader")
    if any(module.training for module in reader.model.modules()):
        raise ValueError("reader must remain in evaluation mode during oracle training")
    if not isinstance(adapter_parameters, tuple):
        raise TypeError("adapter_parameters must be a tuple")
    reader_owned = {id(parameter) for parameter in adapter_parameters}
    if (len(reader_owned) != len(adapter_parameters)
            or reader_owned != {id(p) for p in reader.model.parameters() if p.requires_grad}):
        raise ValueError("reader adapter ownership differs from the declared parameters")
    parameters = (*tuple(bridge.parameters()), *adapter_parameters)
    if len({id(parameter) for parameter in parameters}) != len(parameters) or any(
        not parameter.requires_grad for parameter in parameters
    ):
        raise ValueError("bridge and adapter parameters must be distinct and trainable")
    if any(parameter.device != reader.model.device for parameter in parameters):
        raise ValueError("all oracle training parameters must share the reader device")
    if any(parameter.grad is not None for parameter in reader.model.parameters()
           if id(parameter) not in reader_owned):
        raise ValueError("frozen reader has unexpected gradients")
    return parameters


def train_oracle_batch(
    reader: PretrainedReader,
    bridge: SlotReadout,
    examples: Sequence[OracleExample],
    optimizer: torch.optim.Optimizer,
    *,
    adapter_parameters: tuple[nn.Parameter, ...] = (),
) -> dict[str, float | int]:
    """Optimize only the bridge and declared Q/V LoRA parameters."""

    if isinstance(examples, (str, bytes)) or not isinstance(examples, Sequence) or not examples:
        raise ValueError("a training batch must contain examples")
    parameters = _validate_modules(reader, bridge, adapter_parameters)
    optimized = [p for group in optimizer.param_groups for p in group["params"]]
    if (len(optimized) != len(parameters)
            or {id(p) for p in optimized} != {id(p) for p in parameters}):
        raise ValueError("optimizer must own exactly the bridge and declared adapter")
    for example in examples:
        _validate_example(example, bridge.reader_width)

    optimizer.zero_grad(set_to_none=True)
    device = reader.model.device
    reads: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    weights: list[float] = []
    for example in examples:
        count = sum(len(endpoint.queries) for endpoint in example.endpoints)
        before = torch.tensor(example.before_ids, device=device)
        for endpoint in example.endpoints:
            oracle_bits(endpoint.state)
            values = endpoint.state.values.to(device=device)
            valid = endpoint.state.valid.to(device=device)
            state = LatentSlotState(values, valid)
            memory = bridge(state)[0]
            for query in endpoint.queries:
                reads.append((before, memory, torch.tensor(query.after_ids, device=device),
                              torch.tensor(query.answer_ids, device=device)))
                weights.append(1.0 / (len(examples) * count))
    losses = prefix_answer_losses(reader, reads)
    loss = (losses * losses.new_tensor(weights)).sum()
    if not torch.isfinite(loss):
        raise ValueError("nonfinite answer loss")
    loss.backward()
    frozen = [p for p in reader.model.parameters() if id(p) not in {id(a) for a in adapter_parameters}]
    if any(p.grad is None for p in parameters):
        raise ValueError("a declared training parameter did not receive a gradient")
    if any(p.grad is not None for p in frozen):
        raise ValueError("frozen reader gradient ownership violation")
    for parameter in parameters:
        if not bool(torch.isfinite(parameter.grad).all()):
            raise ValueError("nonfinite training gradient")
    bridge_parameters = tuple(bridge.parameters())
    bridge_norm = torch.stack([p.grad.float().square().sum() for p in bridge_parameters]).sum().sqrt()
    adapter_norm = (torch.stack([p.grad.float().square().sum() for p in adapter_parameters]).sum().sqrt()
                    if adapter_parameters else loss.new_zeros(()))
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer.step()
    if any(not bool(torch.isfinite(parameter).all()) for parameter in parameters):
        raise ValueError("optimizer produced nonfinite parameters")
    return {
        "answer_ce": float(loss.detach()),
        "gradient_norm": float(norm),
        "bridge_gradient_norm": float(bridge_norm),
        "adapter_gradient_norm": float(adapter_norm),
        "episodes": len(examples),
        "answer_sequences": len(reads),
        "write_calls": 0,
        "persistent_bytes": bridge.state_bytes,
    }


@torch.no_grad()
def oracle_validation_losses(
    reader: PretrainedReader, bridge: SlotReadout, episodes: Sequence[Episode],
) -> dict[str, dict[str, float | int]]:
    """Return validation CE grouped by condition with equal episode weight."""

    if isinstance(episodes, (str, bytes)) or not isinstance(episodes, Sequence) or not episodes:
        raise ValueError("episodes must be a nonempty sequence")
    if any(not isinstance(episode, Episode) or episode.split != "validation" for episode in episodes):
        raise ValueError("validation losses accept validation episodes only")
    if any(module.training for module in reader.model.modules()) or bridge.training:
        raise ValueError("validation requires evaluation-mode reader and bridge")
    if any(parameter.requires_grad or parameter.grad is not None for parameter in bridge.parameters()):
        raise ValueError("validation bridge must be frozen without gradients")
    if any(parameter.requires_grad or parameter.grad is not None for parameter in reader.model.parameters()):
        raise ValueError("validation reader must be frozen without gradients")
    records = oracle_records(episodes)
    by_episode: dict[str, list[StateRecord]] = {}
    for record in records:
        by_episode.setdefault(record.episode_id, []).append(record)
    totals: dict[str, dict[str, float | int]] = {}
    device = reader.model.device
    for episode in episodes:
        example = encode_oracle_episode(reader, episode, by_episode[episode.id])
        reads = []
        before = torch.tensor(example.before_ids, device=device)
        for endpoint in example.endpoints:
            state = LatentSlotState(endpoint.state.values.to(device=device),
                                    endpoint.state.valid.to(device=device))
            memory = bridge(state)[0]
            for query in endpoint.queries:
                reads.append((before, memory, torch.tensor(query.after_ids, device=device),
                              torch.tensor(query.answer_ids, device=device)))
        value = float(prefix_answer_losses(reader, reads).mean())
        row = totals.setdefault(episode.condition, {"sum_episode_ce": 0.0, "episodes": 0})
        row["sum_episode_ce"] = float(row["sum_episode_ce"]) + value
        row["episodes"] = int(row["episodes"]) + 1
    return {
        condition: {"episode_mean_ce": float(row["sum_episode_ce"]) / int(row["episodes"]),
                    "episodes": int(row["episodes"])}
        for condition, row in totals.items()
    }
