"""Controlled correction episodes for content-aware memory replacement."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, permutations
from numbers import Integral

from tinymem.data.correction_deletion import ENTITIES, VALUES
from tinymem.data.schema import SUPPORTED_SPLITS
from tinymem.data.symbolic_world import parse_qa1_movement, parse_qa1_question
from tinymem.tokenization.byte_tokenizer import ByteTokenizer


_SPLIT_SEED_OFFSET = {
    "train": 0,
    "validation": 1_000_000,
    "test": 2_000_000,
}
REPLACEMENT_PROTOCOLS = ("history_disjoint_v2", "legacy_seed_v1")


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
    protocol: str = "history_disjoint_v2",
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
    if protocol not in REPLACEMENT_PROTOCOLS:
        raise ValueError(f"unsupported replacement protocol {protocol!r}")
    if protocol == "history_disjoint_v2":
        return _generate_disjoint(
            tokenizer, split, int(count), int(memory_capacity),
            int(segment_length), int(base_seed),
        )

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


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _history_digest(
    facts: tuple[tuple[str, str], ...], correction: tuple[str, str],
) -> str:
    return _digest({"facts": sorted(facts), "correction": correction})


def _parse_history(
    example: ReplacementQAExample,
) -> tuple[tuple[tuple[str, str], ...], tuple[str, str]]:
    tokenizer = ByteTokenizer()

    def movement(ids: tuple[int, ...], prefix: str, fact_id: int):
        text = tokenizer.decode(ids, skip_special_tokens=False)
        if not text.startswith(prefix) or not text.endswith("\n"):
            raise ValueError(f"expected {prefix!r} movement line")
        person, value = parse_qa1_movement(text[len(prefix):-1], fact_id)
        return person, value.value

    facts = tuple(
        movement(ids, "Fact: ", index + 1)
        for index, ids in enumerate(example.initial_fact_ids)
    )
    if len({person for person, _ in facts}) != len(facts):
        raise ValueError("initial facts must have distinct entities")
    correction = movement(
        example.correction_ids, "Correction: ", len(facts) + 1,
    )
    return facts, correction


def replacement_history_id(example: ReplacementQAExample) -> str:
    """Identify the semantic history, without question or initial fact order."""
    facts, correction = _parse_history(example)
    return _history_digest(facts, correction)


def validate_replacement_example(example: ReplacementQAExample) -> None:
    """Replay decoded movements independently of generator labels and slots."""
    facts, correction = _parse_history(example)
    if correction[0] != facts[example.correction_slot][0]:
        raise ValueError("correction_slot does not identify the corrected entity")
    tokenizer = ByteTokenizer()
    query = tokenizer.decode(example.query_ids, skip_special_tokens=False)
    prefix, suffix = "Question: ", "\nAnswer:"
    if not query.startswith(prefix) or not query.endswith(suffix):
        raise ValueError("invalid replacement query format")
    person = parse_qa1_question(query[len(prefix):-len(suffix)])
    if person != facts[example.query_slot][0]:
        raise ValueError("query_slot does not identify the queried entity")
    state = dict(facts)
    state[correction[0]] = correction[1]
    answer = tokenizer.decode(example.answer_ids, skip_special_tokens=False)
    if answer != state[person]:
        raise ValueError("answer disagrees with symbolic replay")


@lru_cache(maxsize=9)
def _partition_histories(
    capacity: int, split: str,
) -> tuple[tuple[tuple[tuple[str, str], ...], tuple[str, str]], ...]:
    histories = []
    for entities in combinations(ENTITIES, capacity):
        for values in permutations(VALUES["location"], capacity):
            facts = tuple(zip(entities, values, strict=True))
            for entity in entities:
                for new_value in VALUES["location"]:
                    if new_value in values:
                        continue
                    correction = (entity, new_value)
                    bucket = int(_history_digest(facts, correction), 16) % 10
                    assigned = "train" if bucket < 8 else (
                        "validation" if bucket == 8 else "test"
                    )
                    if assigned == split:
                        histories.append((facts, correction))
    return tuple(histories)


def _generate_disjoint(
    tokenizer: ByteTokenizer, split: str, count: int, capacity: int,
    segment_length: int, seed: int,
) -> list[ReplacementQAExample]:
    histories = list(_partition_histories(capacity, split))
    needed = (count + capacity - 1) // capacity
    if needed > len(histories):
        raise ValueError(
            f"{split} has only {len(histories)} distinct histories at capacity "
            f"{capacity}; requested {needed}"
        )
    rng = random.Random(seed)
    rng.shuffle(histories)
    generated = []
    for index, (facts, correction) in enumerate(histories[:needed]):
        correction_slot = index % capacity
        ordered = [fact for fact in facts if fact[0] != correction[0]]
        rng.shuffle(ordered)
        ordered.insert(correction_slot, next(f for f in facts if f[0] == correction[0]))
        initial_ids = tuple(
            tuple(tokenizer.encode(f"Fact: {entity} moved to the {value}.\n"))
            for entity, value in ordered
        )
        correction_ids = tuple(tokenizer.encode(
            f"Correction: {correction[0]} moved to the {correction[1]}.\n"
        ))
        for query_slot, (entity, old_value) in enumerate(ordered):
            if len(generated) == count:
                break
            query_ids = tuple(tokenizer.encode(f"Question: Where is {entity}?\nAnswer:"))
            answer = correction[1] if entity == correction[0] else old_value
            content_id = _digest((initial_ids, correction_ids, query_ids))
            example = ReplacementQAExample(
                initial_fact_ids=initial_ids,
                correction_ids=correction_ids,
                query_ids=query_ids,
                answer_ids=tuple(tokenizer.encode(answer)),
                correction_slot=correction_slot,
                query_slot=query_slot,
                segment_length=segment_length,
                source_example_id=f"replacement-qa:history-disjoint-v2:{content_id}",
                split=split,
            )
            validate_replacement_example(example)
            generated.append(example)
    return generated
