"""Group-disjoint visible-evidence data for adapting the shared reader."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tinymem.data.babi import load_babi_file
from tinymem.data.reader_gate import BABI_GATE_FILES, ReaderCase
from tinymem.data.replacement_qa import generate_replacement_qa_examples, replacement_history_id, validate_replacement_example
from tinymem.data.symbolic_world import interpret
from tinymem.tokenization.byte_tokenizer import ByteTokenizer


@dataclass(frozen=True)
class ReaderAdaptationData:
    train: tuple[ReaderCase, ...]
    development: tuple[ReaderCase, ...]
    excluded_episodes: tuple[str, ...]


def _internal_split(history_id: str) -> str:
    digest = hashlib.sha256(f"reader-adaptation-v1:{history_id}".encode()).hexdigest()
    return "development" if int(digest, 16) % 5 == 0 else "train"


def make_reader_adaptation_data(
    babi_root: Path, gate_cases: Sequence[ReaderCase], *, train_per_task: int = 400,
    development_per_task: int = 50, replacement_examples: int = 1000, seed: int = 1337,
) -> ReaderAdaptationData:
    """Split connected history groups before selecting any questions."""
    if not gate_cases or min(train_per_task, development_per_task, replacement_examples) <= 0 or replacement_examples % 2 or seed < 0:
        raise ValueError("gate cases and positive counts are required; replacement count must be even")
    episodes = defaultdict(list)
    for task, filename in BABI_GATE_FILES.items():
        for row in load_babi_file(babi_root / filename, task_id=task, split="train"):
            episodes[row.source_example_id.rsplit(":question-", 1)[0]].append(row)
    parent = {episode: episode for episode in episodes}

    def root(episode):
        while parent[episode] != episode:
            parent[episode] = parent[parent[episode]]
            episode = parent[episode]
        return episode

    context_owner = {}
    for episode, rows in episodes.items():
        for row in rows:
            if row.context in context_owner:
                left, right = root(episode), root(context_owner[row.context])
                parent[max(left, right)] = min(left, right)
            else:
                context_owner[row.context] = episode
    components = defaultdict(list)
    for episode in episodes:
        components[root(episode)].append(episode)
    forbidden_episodes = {case.history_id for case in gate_cases}
    forbidden_contexts = {case.context for case in gate_cases}
    pools = {"train": defaultdict(list), "development": defaultdict(list)}
    excluded = []
    for component in components.values():
        if any(episode in forbidden_episodes or any(row.context in forbidden_contexts for row in episodes[episode]) for episode in component):
            excluded.extend(component)
            continue
        history_id = hashlib.sha256("\n".join(sorted(component)).encode()).hexdigest()
        split = _internal_split(history_id)
        for episode in component:
            row = random.Random(f"reader-adaptation-v1:{seed}:{episode}").choice(episodes[episode])
            pools[split][row.task_id].append(ReaderCase(
                row.source_example_id, f"babi_{row.task_id}", history_id,
                row.context, row.question, row.answer,
            ))
    selected = {"train": [], "development": []}
    for split, count in (("train", train_per_task), ("development", development_per_task)):
        for task in BABI_GATE_FILES:
            rows = pools[split][task]
            random.Random(f"reader-adaptation-v1:{seed}:{split}:{task}").shuffle(rows)
            if len(rows) < count:
                raise ValueError(f"not enough disjoint {split} {task} episodes")
            selected[split].extend(rows[:count])
    source_by_id = {row.source_example_id: row for rows in episodes.values() for row in rows}
    for rows in selected.values():
        for case in rows:
            interpret(source_by_id[case.case_id])

    tokenizer = ByteTokenizer()
    replacements = generate_replacement_qa_examples(
        tokenizer, split="train", count=replacement_examples, memory_capacity=2,
        segment_length=64, base_seed=seed, protocol="history_disjoint_v2",
    )
    gate_histories = {case.history_id for case in gate_cases}
    for row in replacements:
        validate_replacement_example(row)
        history_id = replacement_history_id(row)
        if history_id in gate_histories:
            raise ValueError("replacement training overlaps the reader gate")
        split = _internal_split(history_id)
        context = "".join(tokenizer.decode(ids) for ids in (*row.initial_fact_ids, row.correction_ids))
        question = tokenizer.decode(row.query_ids).removeprefix("Question: ").removesuffix("\nAnswer:")
        selected[split].append(ReaderCase(
            row.source_example_id, "correction_changed" if row.query_requires_correction else "correction_unchanged",
            history_id, context, question, tokenizer.decode(row.answer_ids),
        ))
    train_histories = {row.history_id for row in selected["train"]}
    development_histories = {row.history_id for row in selected["development"]}
    if train_histories & development_histories:
        raise ValueError("internal adaptation splits share histories")
    return ReaderAdaptationData(tuple(selected["train"]), tuple(selected["development"]), tuple(sorted(excluded)))
