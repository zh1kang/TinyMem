"""Visible-evidence development controls for selecting a capable reader."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

from tinymem.data.babi import load_babi_file
from tinymem.data.replacement_qa import (
    generate_replacement_qa_examples,
    replacement_history_id,
    validate_replacement_example,
)
from tinymem.data.symbolic_world import interpret
from tinymem.tokenization.byte_tokenizer import ByteTokenizer


BABI_GATE_FILES = {
    "qa1": "qa1_single-supporting-fact_train.txt",
    "qa2": "qa2_two-supporting-facts_train.txt",
    "qa3": "qa3_three-supporting-facts_train.txt",
}


@dataclass(frozen=True)
class ReaderCase:
    case_id: str
    category: str
    history_id: str
    context: str
    question: str
    answer: str


def make_reader_gate_cases(
    babi_root: Path, *, count: int = 100, replacement_count: int = 200,
    copy_count: int = 64, seed: int = 10000,
) -> list[ReaderCase]:
    """Use development inputs only; never load official test or external data."""
    if count <= 0 or replacement_count <= 0 or replacement_count % 2 or copy_count <= 0 or seed < 0:
        raise ValueError("counts must be positive, replacement_count even, seed nonnegative")
    cases = []
    for task, filename in BABI_GATE_FILES.items():
        rows = load_babi_file(babi_root / filename, task_id=task, split="train")
        random.Random(f"reader-gate-v1:{seed}:{task}").shuffle(rows)
        episodes = set()
        selected = 0
        for row in rows:
            episode = row.source_example_id.rsplit(":question-", 1)[0]
            if episode in episodes:
                continue
            episodes.add(episode)
            interpret(row)
            cases.append(ReaderCase(
                row.source_example_id, f"babi_{task}", episode,
                row.context, row.question, row.answer,
            ))
            selected += 1
            if selected == count:
                break
        if selected != count:
            raise ValueError(f"not enough distinct {task} training episodes")

    tokenizer = ByteTokenizer()
    replacements = generate_replacement_qa_examples(
        tokenizer, split="validation", count=replacement_count, memory_capacity=2,
        segment_length=64, base_seed=seed, protocol="history_disjoint_v2",
    )
    for row in replacements:
        validate_replacement_example(row)
        context = "".join(tokenizer.decode(ids) for ids in (*row.initial_fact_ids, row.correction_ids))
        question = tokenizer.decode(row.query_ids).removeprefix("Question: ").removesuffix("\nAnswer:")
        cases.append(ReaderCase(
            row.source_example_id,
            "correction_changed" if row.query_requires_correction else "correction_unchanged",
            replacement_history_id(row), context, question, tokenizer.decode(row.answer_ids),
        ))

    for category in ("exact_copy", "randomized_bindings"):
        rng = random.Random(f"reader-gate-v1:{seed}:{category}")
        for index in range(copy_count):
            values = [f"{value:08x}" for value in rng.sample(range(2**32), 4)]
            if category == "exact_copy":
                context = f"The access code is {values[0]}."
                question = "What is the access code?"
                answer = values[0]
            else:
                names = [f"person{value:06x}" for value in rng.sample(range(2**24), 4)]
                context = "\n".join(f"{name} is at {value}." for name, value in zip(names, values, strict=True))
                slot = index % 4
                question = f"Where is {names[slot]}?"
                answer = values[slot]
            history = hashlib.sha256(context.encode()).hexdigest()
            cases.append(ReaderCase(f"reader-gate-v1:{category}:{history}", category, history, context, question, answer))
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("reader gate contains duplicate case IDs")
    return cases
