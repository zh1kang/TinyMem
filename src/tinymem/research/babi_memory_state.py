"""Question-independent QA1 oracle state and packed explicit storage."""

from collections.abc import Sequence
import math

import torch

from tinymem.memory.delta_slots import delta_update
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.babi_memory_data import Vocabulary, parse_movement, query_entity


def _event(text: str, vocabulary: Vocabulary) -> tuple[int, int]:
    entity, room = parse_movement(text)
    if entity not in vocabulary.entities or room not in vocabulary.rooms:
        raise ValueError('movement contains an undeclared entity or room')
    return vocabulary.entities.index(entity), vocabulary.rooms.index(room)


def oracle_state(history: Sequence[str], vocabulary: Vocabulary) -> LatentSlotState:
    if isinstance(history, (str, bytes)):
        raise TypeError('history must be a sequence of fact strings')
    matrix = torch.zeros(1, 8, 8, dtype=torch.float32)
    for text in history:
        entity, room = _event(text, vocabulary)
        key = torch.zeros(1, 8)
        value = torch.zeros(1, 8)
        key[0, entity] = 1
        value[0, room] = 0.5
        matrix = delta_update(matrix, key, value, torch.tensor([0.75]))
    return LatentSlotState(matrix.reshape(1, 2, 32).clone().contiguous(),
                           torch.ones(1, 2, dtype=torch.bool))


def oracle_locations(state: LatentSlotState, vocabulary: Vocabulary) -> tuple[int, ...]:
    if (state.values.shape != (1, 2, 32) or state.valid.shape != (1, 2)
            or state.values.dtype != torch.float32 or state.valid.dtype != torch.bool
            or state.values.device.type != 'cpu' or state.valid.device.type != 'cpu'
            or any(t._base is not None or not t.is_contiguous() or t.requires_grad
                   or t.grad_fn is not None for t in (state.values, state.valid))
            or state.nbytes != 258 or not bool(state.valid.all())
            or not bool(torch.isfinite(state.values).all()) or bool((state.values < 0).any())):
        raise ValueError('oracle state must be finite nonnegative CPU FP32 with 258 owned bytes')
    matrix = state.values.reshape(8, 8)
    if bool(matrix[len(vocabulary.entities):].any()) or bool(matrix[:, len(vocabulary.rooms):].any()):
        raise ValueError('oracle state uses undeclared entity or room coordinates')
    result = []
    for row in matrix[:len(vocabulary.entities), :len(vocabulary.rooms)]:
        maximum = row.max()
        if maximum == 0:
            result.append(0)
        else:
            if int((row == maximum).sum()) != 1:
                raise ValueError('oracle state has an ambiguous room maximum')
            result.append(int(row.argmax()) + 1)
    return tuple(result)


def packed_size(vocabulary: Vocabulary) -> int:
    bits = math.ceil(math.log2(len(vocabulary.rooms) + 1))
    return (bits * len(vocabulary.entities) + 7) // 8


def unpack_locations(payload: bytes, vocabulary: Vocabulary) -> tuple[int, ...]:
    if type(payload) is not bytes or len(payload) != packed_size(vocabulary):
        raise ValueError('packed state has the wrong type or byte count')
    bits = math.ceil(math.log2(len(vocabulary.rooms) + 1))
    value = int.from_bytes(payload, 'little')
    if value >> (bits * len(vocabulary.entities)):
        raise ValueError('packed state has nonzero padding bits')
    codes = tuple((value >> (index * bits)) & ((1 << bits) - 1)
                  for index in range(len(vocabulary.entities)))
    if any(code > len(vocabulary.rooms) for code in codes):
        raise ValueError('packed state contains an undeclared room code')
    return codes


def packed_update(payload: bytes, text: str, vocabulary: Vocabulary) -> bytes:
    """Only the packed bytes persist between updates; no hidden world dictionary."""
    unpack_locations(payload, vocabulary)
    entity, room = _event(text, vocabulary)
    bits = math.ceil(math.log2(len(vocabulary.rooms) + 1))
    offset = bits * entity
    value = int.from_bytes(payload, 'little')
    value = (value & ~(((1 << bits) - 1) << offset)) | ((room + 1) << offset)
    return value.to_bytes(packed_size(vocabulary), 'little')


def packed_answer(payload: bytes, question: str, vocabulary: Vocabulary) -> str:
    code = unpack_locations(payload, vocabulary)[query_entity(question, vocabulary)]
    return 'unknown' if code == 0 else vocabulary.rooms[code - 1]
