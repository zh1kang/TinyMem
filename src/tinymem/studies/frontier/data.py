"""Question-blind records and whole-story splits for the storage comparison."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from pathlib import Path

from tinymem.data.babi import load_babi_file


@dataclass(frozen=True)
class StorageQuestion:
    id: str
    story: str
    task: str
    split: str
    records: tuple[str, ...]
    question: str
    answer: str


def load_questions(directory: Path, split: str) -> tuple[StorageQuestion, ...]:
    if split not in ('train', 'test'):
        raise ValueError('load the official train or test source')
    result = []
    for task in range(1, 6):
        paths = sorted(directory.glob(f'qa{task}_*_{split}.txt'))
        if len(paths) != 1:
            raise ValueError(f'expected one official qa{task} {split} file')
        for example in load_babi_file(paths[0], task_id=f'qa{task}', split=split):
            story, separator, _ = example.source_example_id.rpartition(':question-')
            if not separator:
                raise ValueError('missing official story identity')
            result.append(StorageQuestion(
                example.source_example_id, story, example.task_id, split,
                tuple(example.context.splitlines()), example.question, example.answer,
            ))
    return tuple(result)


def split_questions(questions: tuple[StorageQuestion, ...], *, seed: int
                    ) -> tuple[tuple[StorageQuestion, ...], tuple[StorageQuestion, ...]]:
    if not questions or any(q.split != 'train' for q in questions):
        raise ValueError('split only official training questions')
    if len({q.id for q in questions}) != len(questions):
        raise ValueError('duplicate question identity')
    held_out = set()
    for task in sorted({q.task for q in questions}):
        stories = sorted({q.story for q in questions if q.task == task}, key=lambda s: (
            hashlib.sha256(f'{seed}:{s}'.encode()).hexdigest(), s,
        ))
        if len(stories) < 10:
            raise ValueError('each task requires at least ten complete stories')
        held_out.update(stories[:len(stories) // 10])
    train = tuple(q for q in questions if q.story not in held_out)
    validation = tuple(replace(q, split='validation') for q in questions if q.story in held_out)
    return train, validation


def select_noise(rows: tuple[str, ...], tokenizer, *, count: int, seed: int,
                 excluded: frozenset[str] = frozenset(), max_tokens: int = 64) -> tuple[str, ...]:
    """Select text only, with no fact parser or supporting-fact annotations."""
    candidates = set()
    for row in rows:
        for sentence in re.split(r'(?<=[.!?])\s+', ' '.join(row.split())):
            if not sentence or sentence in excluded:
                continue
            candidates.add(sentence)
    ordered = sorted(candidates, key=lambda s: (hashlib.sha256(f'{seed}:{s}'.encode()).hexdigest(), s))
    result = []
    for text in ordered:
        if 8 <= len(tokenizer.encode(text + '\n', add_special_tokens=False)) <= max_tokens:
            result.append(text)
            if len(result) == count:
                return tuple(result)
    raise ValueError('not enough distinct bounded noise records')


def write_records(question: StorageQuestion, noise: tuple[str, ...], *, level: int,
                  seed: int) -> tuple[str, ...]:
    """Build a text stream without using the question text, answer, or supports.

    The interleaved portion is prefix-consistent within each story.
    A declared delay follows the final source record before the read.
    """
    if level not in (0, 1, 2) or (level and not noise):
        raise ValueError('noise level must be 0, 1, or 2 with a nonempty pool')
    if not level:
        return question.records
    per_block, tail = (1, 4) if level == 1 else (4, 16)

    def selected(tag: str) -> str:
        digest = hashlib.sha256(f'{seed}:{question.story}:{tag}'.encode()).digest()
        return noise[int.from_bytes(digest[:8], 'little') % len(noise)]

    records = []
    for position, record in enumerate(question.records, 1):
        records.append(record)
        if position % 4 == 0:
            records.extend(selected(f'block:{position}:{i}') for i in range(per_block))
    records.extend(selected(f'tail:{len(question.records)}:{i}') for i in range(tail))
    return tuple(records)


def dictionary_from_training(questions: tuple[StorageQuestion, ...]) -> tuple[str, ...]:
    if not questions or any(q.split != 'train' for q in questions):
        raise ValueError('the shared dictionary uses training text only')
    return tuple(sorted({record for q in questions for record in q.records}))
