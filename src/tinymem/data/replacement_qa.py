"""Controlled correction episodes for content-aware memory replacement."""

from __future__ import annotations

import random
from dataclasses import dataclass
from numbers import Integral

from tinymem.data.correction_deletion import ENTITIES, VALUES
from tinymem.data.schema import SUPPORTED_SPLITS
from tinymem.tokenization.byte_tokenizer import ByteTokenizer


_SPLIT_SEED_OFFSET = {
    "train": 0,
    "validation": 1_000_000,
    "test": 2_000_000,
}


@dataclass(frozen=True)
class ReplacementQAExample:
    """Hold a full bank, one correction, and one answer-supervised query."""

    initial_fact_ids: tuple[tuple[int, ...], ...]
    correction_ids: tuple[int, ...]
    query_ids: tuple[int, ...]
    answer_ids: tuple[int, ...]
    correction_slot: int
    query_slot: int
    segment_length: int
    source_example_id: str
    split: str

    def __post_init__(self) -> None:
        if not isinstance(self.initial_fact_ids, tuple) or len(
            self.initial_fact_ids
        ) < 2:
            raise ValueError("initial_fact_ids must contain at least two slots")
        for name, sequences in (
            ("initial_fact_ids", self.initial_fact_ids),
            ("correction_ids", (self.correction_ids,)),
            ("query_ids", (self.query_ids,)),
            ("answer_ids", (self.answer_ids,)),
        ):
            for sequence in sequences:
                if not isinstance(sequence, tuple) or not sequence:
                    raise ValueError(f"{name} must contain nonempty tuples")
                if any(
                    isinstance(token_id, bool)
                    or not isinstance(token_id, int)
                    or not 0 <= token_id < ByteTokenizer.vocab_size
                    for token_id in sequence
                ):
                    raise ValueError(f"{name} must contain valid byte-token IDs")
        if isinstance(self.segment_length, bool) or not isinstance(
            self.segment_length,
            Integral,
        ):
            raise TypeError("segment_length must be an integer")
        if self.segment_length <= 0:
            raise ValueError("segment_length must be positive")
        if any(
            len(sequence) > self.segment_length
            for sequence in (*self.initial_fact_ids, self.correction_ids)
        ):
            raise ValueError("each fact and correction must fit inside one segment")
        if len(self.query_ids) + len(self.answer_ids) + 1 > self.segment_length:
            raise ValueError("query and answer must fit inside one segment")
        for name, slot in (
            ("correction_slot", self.correction_slot),
            ("query_slot", self.query_slot),
        ):
            if isinstance(slot, bool) or not isinstance(slot, Integral):
                raise TypeError(f"{name} must be an integer")
            if not 0 <= slot < self.memory_capacity:
                raise ValueError(f"{name} is outside the memory bank")
        for name in ("source_example_id", "split"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")
        if self.split not in SUPPORTED_SPLITS:
            raise ValueError(f"unsupported split {self.split!r}")

    @property
    def memory_capacity(self) -> int:
        return len(self.initial_fact_ids)

    @property
    def query_requires_correction(self) -> bool:
        return self.query_slot == self.correction_slot

    @property
    def query_position_offset(self) -> int:
        return (self.memory_capacity + 1) * self.segment_length


def generate_replacement_qa_examples(
    tokenizer: ByteTokenizer,
    *,
    split: str,
    count: int,
    memory_capacity: int = 3,
    segment_length: int = 64,
    base_seed: int = 0,
) -> list[ReplacementQAExample]:
    """Generate deterministic episodes where one stale entity fact is replaced."""
    if not isinstance(tokenizer, ByteTokenizer):
        raise TypeError("tokenizer must be a ByteTokenizer")
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"unsupported split {split!r}")
    for name, value in (
        ("count", count),
        ("memory_capacity", memory_capacity),
        ("segment_length", segment_length),
        ("base_seed", base_seed),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    if count <= 0 or segment_length <= 0 or base_seed < 0:
        raise ValueError("count and segment_length must be positive and seed nonnegative")
    if not 2 <= memory_capacity <= min(len(ENTITIES), len(VALUES["location"]) - 1):
        raise ValueError("memory_capacity exceeds the controlled entity/value bank")

    generated = []
    for index in range(int(count)):
        episode_seed = int(base_seed) + _SPLIT_SEED_OFFSET[split] + index
        rng = random.Random(episode_seed)
        entities = rng.sample(ENTITIES, k=int(memory_capacity))
        values = rng.sample(VALUES["location"], k=int(memory_capacity) + 1)
        correction_slot = index % int(memory_capacity)
        query_slot = (index // int(memory_capacity)) % int(memory_capacity)
        initial_fact_ids = tuple(
            tuple(tokenizer.encode(f"Fact: {entity} moved to the {value}.\n"))
            for entity, value in zip(
                entities,
                values[: int(memory_capacity)],
                strict=True,
            )
        )
        corrected_value = values[-1]
        correction_ids = tuple(
            tokenizer.encode(
                "Correction: "
                f"{entities[correction_slot]} moved to the {corrected_value}.\n"
            )
        )
        answer = (
            corrected_value
            if query_slot == correction_slot
            else values[query_slot]
        )
        query_ids = tuple(
            tokenizer.encode(
                f"Question: Where is {entities[query_slot]}?\nAnswer:"
            )
        )
        answer_ids = tuple(tokenizer.encode(answer))
        generated.append(
            ReplacementQAExample(
                initial_fact_ids=initial_fact_ids,
                correction_ids=correction_ids,
                query_ids=query_ids,
                answer_ids=answer_ids,
                correction_slot=correction_slot,
                query_slot=query_slot,
                segment_length=int(segment_length),
                source_example_id=(
                    f"replacement-qa:{split}:seed-{episode_seed}"
                ),
                split=split,
            )
        )
    return generated
