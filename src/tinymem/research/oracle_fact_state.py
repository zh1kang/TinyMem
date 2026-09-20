"""Privileged, query-blind known-state control for the delta fact study.

This module is deliberately independent of the learned writer.  It parses the
rendered statement text, writes a fixed binary code with the repository's
delta-rule update, and returns owned CPU states at the same endpoint positions
as the learned study.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.memory.delta_slots import delta_update
from tinymem.research.delta_fact_data import Episode, parse_statement, replay
from tinymem.research.delta_fact_evaluation import StateRecord


ORACLE_SLOTS = 2
ORACLE_MEMORY_WIDTH = 32
ORACLE_KEY_WIDTH = 8
ORACLE_VALUE_WIDTH = 8
ORACLE_BETA = 0.75
ORACLE_VALUE_MAGNITUDE = 0.5
ORACLE_STATE_BYTES = 2 * ORACLE_MEMORY_WIDTH * 4 + ORACLE_SLOTS
_FACT_ROWS = tuple(range(4))


def _validate_episode(episode: Episode) -> None:
    if not isinstance(episode, Episode):
        raise TypeError("episodes must contain Episode values")
    if len(episode.prefix) != 8 or len(episode.tail) not in (0, 8):
        raise ValueError("episodes must contain an eight-statement prefix and optional tail")


def _texts(statements: Sequence[object]) -> tuple[str, ...]:
    result: list[str] = []
    for statement in statements:
        text = getattr(statement, "text", None)
        if not isinstance(text, str):
            raise ValueError("oracle statements must provide rendered text")
        # Parse the text itself.  Metadata carried by Statement is not used to
        # construct the code or the labels.
        parse_statement(text)
        result.append(text)
    return tuple(result)


def _matrix_for_texts(texts: Sequence[str]) -> torch.Tensor:
    matrix = torch.zeros(1, ORACLE_KEY_WIDTH, ORACLE_VALUE_WIDTH, dtype=torch.float32)
    known: dict[int, int] = {}
    for text in texts:
        statement = parse_statement(text)
        key = torch.zeros(1, ORACLE_KEY_WIDTH, dtype=torch.float32)
        key[0, statement.entity] = 1.0
        value = torch.zeros(1, ORACLE_VALUE_WIDTH, dtype=torch.float32)
        value[0, 0] = ORACLE_VALUE_MAGNITUDE if statement.value else -ORACLE_VALUE_MAGNITUDE
        beta = torch.full((1,), ORACLE_BETA, dtype=torch.float32)
        previous = matrix
        matrix = delta_update(previous, key, value, beta)
        untouched = [row for row in range(ORACLE_KEY_WIDTH) if row != statement.entity]
        if not torch.equal(previous[:, untouched], matrix[:, untouched]):
            raise ValueError("oracle write changed a nonaddressed row")
        known[statement.entity] = statement.value
        for entity, bit in known.items():
            stored = float(matrix[0, entity, 0])
            if stored == 0 or int(stored > 0) != bit:
                raise ValueError("oracle write lost or failed to correct a known fact")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("oracle state became nonfinite")
    return matrix


def _state_for_texts(texts: Sequence[str]) -> LatentSlotState:
    if not texts:
        raise ValueError("oracle state requires at least one statement")
    matrix = _matrix_for_texts(texts)
    values = matrix.reshape(1, ORACLE_SLOTS, ORACLE_MEMORY_WIDTH).contiguous().clone()
    valid = torch.ones(1, ORACLE_SLOTS, dtype=torch.bool)
    state = LatentSlotState(values, valid)
    # Keep the public constructor honest if the layout constants are changed.
    if state.nbytes != ORACLE_STATE_BYTES:
        raise ValueError("oracle state does not match the declared persistent bytes")
    return state


def _truth(texts: Sequence[str]) -> tuple[int, ...]:
    labels = replay(tuple(texts))
    if len(labels) != 4 or any(value is None for value in labels):
        raise ValueError("oracle endpoint replay must define all four facts")
    return tuple(int(value) for value in labels)


def oracle_bits(state: LatentSlotState) -> tuple[int, ...]:
    """Decode the four sign bits from one owned oracle state.

    The check is intentionally strict about ownership and storage shape.  It
    does not infer facts from query IDs or from any state metadata.
    """

    if not isinstance(state, LatentSlotState):
        raise TypeError("state must be a LatentSlotState")
    values, valid = state.values, state.valid
    if (values.shape != (1, ORACLE_SLOTS, ORACLE_MEMORY_WIDTH)
            or valid.shape != (1, ORACLE_SLOTS)
            or values.dtype != torch.float32 or valid.dtype != torch.bool):
        raise ValueError("oracle state must have FP32 [1, 2, 32] values and boolean validity")
    if values.device.type != "cpu" or valid.device.type != "cpu":
        raise ValueError("oracle state must be stored on CPU")
    if values.device != valid.device or state.nbytes != ORACLE_STATE_BYTES:
        raise ValueError("oracle state storage does not match the declared bytes")
    if values.requires_grad or values.grad_fn is not None or valid.requires_grad or valid.grad_fn is not None:
        raise ValueError("oracle state must be detached")
    if values._base is not None or valid._base is not None or not values.is_contiguous() or not valid.is_contiguous():
        raise ValueError("oracle state tensors must be owned contiguous tensors")
    if not bool(torch.equal(valid, torch.ones_like(valid))):
        raise ValueError("both oracle slots must be valid")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("oracle state must be finite")
    occupied = values[0, 0, [row * ORACLE_VALUE_WIDTH for row in _FACT_ROWS]]
    if bool((occupied == 0).any()):
        raise ValueError("oracle fact cells must have nonzero signs")
    return tuple(int(value) for value in (occupied > 0).tolist())


def _record(episode: Episode, after_write: int, texts: Sequence[str]) -> StateRecord:
    state = _state_for_texts(texts)
    truth = _truth(texts)
    if oracle_bits(state) != truth:
        raise ValueError("oracle state signs disagree with independent replay truth")
    values = state.values.detach().clone().contiguous()
    valid = state.valid.detach().clone().contiguous()
    return StateRecord(
        episode.id, episode.prefix_id, episode.split, episode.wording,
        episode.condition, episode.target, after_write, values, valid, truth,
        float(torch.linalg.vector_norm(values)), None, None,
    )


def oracle_records(episodes: Sequence[Episode]) -> tuple[StateRecord, ...]:
    """Return owned correct states at the phase-two read endpoints."""

    if isinstance(episodes, (str, bytes)) or not isinstance(episodes, Sequence) or not episodes:
        raise ValueError("episodes must be a nonempty sequence")
    records: list[StateRecord] = []
    for episode in episodes:
        _validate_episode(episode)
        texts = _texts((*episode.prefix, *episode.tail))
        endpoints = (8, 9, 16) if episode.tail else (8,)
        for after_write in endpoints:
            records.append(_record(episode, after_write, texts[:after_write]))
    return tuple(records)
