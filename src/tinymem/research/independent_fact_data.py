"""Four independent oracle facts with complementary parity splits."""
from dataclasses import dataclass
from collections import Counter
from itertools import combinations
import hashlib

import torch

from tinymem.data.reader_gate import ReaderCase
from tinymem.data.memory_updates import replay_update_chunks, text_sha256
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.memory_prompt import encode_memory_example
from tinymem.research.readout_runner import ReadoutQuery


NAMESPACE = 'independent-fact-v1'
ENTITIES = tuple('person' + hashlib.sha256(f'{NAMESPACE}:entity:{i}'.encode()).hexdigest()[:20] for i in range(6))
ROOM_PAIRS = (('bathroom', 'hallway'), ('bedroom', 'kitchen'), ('garden', 'office'), ('bathroom', 'hallway'))


def history_id(code: int) -> str:
    if type(code) is not int or not 0 <= code < 16:
        raise ValueError('code must be an integer from zero through fifteen')
    return f'{NAMESPACE}:code-{code:02d}'


@dataclass(frozen=True)
class FactWorld:
    code: int
    cases: tuple[ReaderCase, ...]


@dataclass(frozen=True)
class FactPrompt:
    code: int
    history_id: str
    before_ids: tuple[int, ...]
    history_ids: tuple[int, ...]
    queries: tuple[ReadoutQuery, ...]


def build_worlds() -> tuple[FactWorld, ...]:
    result = []
    for code in range(16):
        context = '\n'.join(f'{entity} moved to the {ROOM_PAIRS[i][(code >> i) & 1]}.' for i, entity in enumerate(ENTITIES[:4]))
        truth = replay_update_chunks((context,))
        cases = tuple(ReaderCase(f'{history_id(code)}:query-{i}', 'update_known' if i < 4 else 'update_missing',
                                text_sha256(context), context, f'Where is {entity}?', truth.get(entity, 'unknown'))
                      for i, entity in enumerate(ENTITIES))
        result.append(FactWorld(code, cases))
    return tuple(result)


def oracle_state(code: int, device: torch.device) -> LatentSlotState:
    history_id(code)
    values = torch.zeros(1, 2, 8, dtype=torch.float32, device=device)
    values[0, 0, :4] = torch.tensor([1 if code & (1 << i) else -1 for i in range(4)], device=device)
    return LatentSlotState(values, torch.ones(1, 2, dtype=torch.bool, device=device))


def nearest_training_code(code: int, training_parity: int) -> int:
    history_id(code)
    if type(training_parity) is not int or training_parity not in (0, 1):
        raise ValueError('training parity must be zero or one')
    return min((other for other in range(16) if other.bit_count() % 2 == training_parity),
               key=lambda other: ((other ^ code).bit_count(), other))


def encode_worlds(reader, worlds) -> tuple[FactPrompt, ...]:
    if tuple(worlds) != build_worlds():
        raise ValueError('world labels, history text, or fixed assignment coverage differ')
    rows = []
    for world in worlds:
        native = tuple(encode_memory_example(reader, case) for case in world.cases)
        first = native[0]
        if any(q.before_ids != first.before_ids or q.history_ids != first.history_ids for q in native):
            raise ValueError('questions must share fixed prefix and history tokens')
        if any(len(q.before_ids) + len(q.history_ids) + len(q.after_ids) + max(8, len(q.answer_ids) - 1)
               > reader.model.config.max_position_embeddings for q in native):
            raise ValueError('full-text reference exceeds model context')
        rows.append(FactPrompt(world.code, history_id(world.code), first.before_ids, first.history_ids,
                               tuple(ReadoutQuery(q.case_id, case.category, case.answer, q.after_ids, q.answer_ids)
                                     for case, q in zip(world.cases, native, strict=True))))
    if len({row.before_ids for row in rows}) != 1 or len({row.history_ids for row in rows}) != 16:
        raise ValueError('fixed prefixes or distinct history token coverage differ')
    if any(len({row.queries[i].after_ids for row in rows}) != 1 for i in range(6)):
        raise ValueError('a non-memory question changes with the world code')
    return tuple(rows)


def audit_design() -> dict:
    worlds = build_worlds()
    splits = {str(parity): [code for code in range(16) if code.bit_count() % 2 == parity] for parity in (0, 1)}
    for codes in splits.values():
        for size in (1, 2, 3):
            for axes in combinations(range(4), size):
                counts = Counter(tuple((code >> i) & 1 for i in axes) for code in codes)
                if len(counts) != 2 ** size or set(counts.values()) != {8 // 2 ** size}:
                    raise ValueError('lower-order bit marginals are not balanced')
    return {'histories': len(worlds), 'questions': 96, 'known': 64, 'absent': 32,
            'codes_by_parity': splits, 'known_question_only_ceiling_per_split': .5,
            'independent_fact_bits': 4, 'unique_undirected_coordinate_edges': 32,
            'whole_world_nearest_neighbor_heldout_accuracy': .75,
            'entity_names': ENTITIES, 'room_pairs': ROOM_PAIRS,
            'generalization_scope': 'unseen_joint_codes_for_fixed_names_templates_and_room_pairs'}
