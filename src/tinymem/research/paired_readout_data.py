"""Matched training histories with conflicting answers to identical questions."""
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
import hashlib
import re

from tinymem.data.memory_updates import UpdateEpisode, replay_update_chunks, text_sha256, validate_update_episode
from tinymem.data.reader_gate import ReaderCase
from tinymem.research.memory_prompt import encode_memory_example
from tinymem.research.pretrained import PretrainedReader
from tinymem.research.readout_runner import EncodedBefore, ReadoutQuery


SELECTION_SALT = 'tinymem-adapted-readout-readiness-v1:'
ROOM_SWAP = {'bathroom': 'hallway', 'hallway': 'bathroom', 'bedroom': 'kitchen',
             'kitchen': 'bedroom', 'garden': 'office', 'office': 'garden'}


@dataclass(frozen=True)
class PairedHistory:
    history_id: str
    pair_id: str
    source: UpdateEpisode
    cases: tuple[ReaderCase, ...]


def build_pairs(episodes: Sequence[UpdateEpisode]) -> tuple[PairedHistory, ...]:
    if len(episodes) < 4 or len({row.episode_id for row in episodes}) != len(episodes):
        raise ValueError('at least four distinct source episodes are required')
    ordered = sorted(episodes, key=lambda row: (hashlib.sha256((SELECTION_SALT + row.episode_id).encode()).hexdigest(), row.episode_id))[:4]
    result = []
    for index, source in enumerate(ordered):
        validate_update_episode(source)
        pair = f'paired-readout-v1:pair-{index}'
        for variant in ('a', 'b'):
            context = source.before[0].context
            if variant == 'b':
                context = re.sub(r'\b(' + '|'.join(ROOM_SWAP) + r')(?=\.)', lambda match: ROOM_SWAP[match[0]], context)
            state = replay_update_chunks(tuple(context.split('\n\n')))
            history = f'{pair}:{variant}'
            cases = tuple(replace(case, case_id=f'{history}:query-{i}', history_id=text_sha256(context),
                                  context=context, answer=state.get(source.entities[i], 'unknown'))
                          for i, case in enumerate(source.before))
            result.append(PairedHistory(history, pair, source, cases))
    return tuple(result)


def encode_pairs(reader: PretrainedReader, pairs: Sequence[PairedHistory]) -> tuple[EncodedBefore, ...]:
    rows = []
    for pair in pairs:
        native = tuple(encode_memory_example(reader, case) for case in pair.cases)
        first = native[0]
        if any(q.before_ids != first.before_ids or q.history_ids != first.history_ids for q in native):
            raise ValueError('queries do not share the same history and fixed prefix')
        if any(len(q.before_ids) + len(q.history_ids) + len(q.after_ids) + max(8, len(q.answer_ids) - 1)
               > reader.model.config.max_position_embeddings for q in native):
            raise ValueError('paired full-text history exceeds context')
        source = pair.source
        rows.append(EncodedBefore(pair.history_id, source.source_group_ids, source.source_case_ids,
                                  source.source_context_sha256, first.before_ids, first.history_ids,
                                  tuple(ReadoutQuery(q.case_id, case.category, case.answer, q.after_ids, q.answer_ids)
                                        for case, q in zip(pair.cases, native, strict=True))))
    audit_pairs(rows)
    return tuple(rows)


def audit_pairs(rows: Sequence[EncodedBefore]) -> dict:
    """Prove conflicting labels under exactly equal non-memory read inputs."""
    expected = [f'paired-readout-v1:pair-{i}:{v}' for i in range(4) for v in ('a', 'b')]
    if [row.history_id for row in rows] != expected or len({row.before_ids for row in rows}) != 1:
        raise ValueError('paired history coverage or fixed prefix differs')
    groups = defaultdict(Counter)
    cases = set()
    donors = {}
    source_seen = {key: set() for key in ('source_group_ids', 'source_case_ids', 'source_context_sha256')}
    for left, right in zip(rows[::2], rows[1::2], strict=True):
        for key, seen in source_seen.items():
            values = getattr(left, key)
            if values != getattr(right, key):
                raise ValueError('variants must share exactly the same original source ownership')
            owned = tuple(value.rsplit(':question-', 1)[0] for value in values) if key == 'source_case_ids' else values
            if len(owned) != 2 or len(set(owned)) != 2 or seen.intersection(owned):
                raise ValueError('source ownership overlaps between different pairs')
            seen.update(owned)
        if left.history_ids == right.history_ids or len(left.queries) != 10 or len(right.queries) != 10:
            raise ValueError('paired histories must differ and each serve ten questions')
        donors[left.history_id], donors[right.history_id] = right.history_id, left.history_id
        for index, (a, b) in enumerate(zip(left.queries, right.queries, strict=True)):
            category = 'update_known' if index < 8 else 'update_missing'
            if a.after_ids != b.after_ids or a.category != category or b.category != category:
                raise ValueError('paired question tokens or categories differ')
            if (index < 8 and (a.answer not in ROOM_SWAP or b.answer != ROOM_SWAP[a.answer])) or (index >= 8 and (a.answer != 'unknown' or b.answer != 'unknown')):
                raise ValueError('paired answers do not implement the fixed conflicting room swap')
            for query in (a, b):
                if query.case_id in cases:
                    raise ValueError('duplicate query identity')
                cases.add(query.case_id)
                groups[query.category, left.before_ids, query.after_ids][query.answer] += 1
    known = [count for key, count in groups.items() if key[0] == 'update_known']
    missing = [count for key, count in groups.items() if key[0] == 'update_missing']
    if len(known) != 32 or len(missing) != 8 or any(sorted(c.values()) != [1, 1] for c in known) or any(c != {'unknown': 2} for c in missing):
        raise ValueError('expected thirty-two conflicting known and eight absent question groups')
    return {'histories': 8, 'questions': 80, 'unique_read_input_groups': 40,
            'known_question_only_ceiling': sum(max(c.values()) for c in known) / 64,
            'missing_question_only_ceiling': 1.0, 'donors': donors}
