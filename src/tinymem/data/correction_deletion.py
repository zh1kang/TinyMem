"""Small deterministic correction-and-deletion benchmark extension."""

from __future__ import annotations

import random
from dataclasses import dataclass

from tinymem.data.schema import ReasoningExample, SUPPORTED_SPLITS
from tinymem.data.symbolic_world import OracleResult


ENTITIES = ("Mary", "Sandra", "John", "Daniel")
ATTRIBUTES = ("location", "color")
VALUES = {
    "location": ("bathroom", "bedroom", "garden", "hallway", "kitchen", "office"),
    "color": ("blue", "green", "purple", "red", "yellow"),
}
UNKNOWN = "UNKNOWN"
_SPLIT_SEED_OFFSET = {"train": 0, "validation": 1_000_000, "test": 2_000_000}
_DEFAULT_CORRECTIONS = {"train": (0, 1), "validation": (2,), "test": (3, 4)}


@dataclass(frozen=True)
class UpdateExample:
    """One generated example and its controlled generation metadata."""

    example: ReasoningExample
    episode_seed: int
    correction_count: int
    deleted: bool
    query_delay: int
    distractor_count: int


def interpret_update_example(example: ReasoningExample) -> OracleResult:
    """Replay SET, CORRECT, DELETE, and DISTRACTOR operations in order."""
    if example.task_id != "correction_deletion":
        raise ValueError("update interpreter requires task_id 'correction_deletion'")
    if example.context_fact_ids is None:
        raise ValueError("update interpretation requires context_fact_ids")
    if not example.question.startswith("QUERY ") or not example.question.endswith("?"):
        raise ValueError("invalid update query")
    query_parts = example.question[6:-1].split()
    if len(query_parts) != 2:
        raise ValueError("update query must contain entity and attribute")
    query_key = (query_parts[0], query_parts[1])

    state: dict[tuple[str, str], tuple[str, int]] = {}
    for line, fact_id in zip(
        example.context.splitlines(), example.context_fact_ids, strict=True
    ):
        parts = line[:-1].split() if line.endswith(".") else []
        if not parts or parts[0] not in {"SET", "CORRECT", "DELETE", "DISTRACTOR", "WAIT"}:
            raise ValueError(f"invalid update operation: {line!r}")
        operation = parts[0]
        if operation == "WAIT":
            if len(parts) != 2:
                raise ValueError("WAIT must contain one step identifier")
            continue
        if operation == "DELETE":
            if len(parts) != 3:
                raise ValueError("DELETE must contain entity and attribute")
            state[(parts[1], parts[2])] = (UNKNOWN, fact_id)
        else:
            if len(parts) != 4:
                raise ValueError(f"{operation} must contain entity, attribute, and value")
            if operation != "DISTRACTOR":
                state[(parts[1], parts[2])] = (parts[3], fact_id)

    if query_key not in state:
        raise ValueError("queried state was never established")
    answer, fact_id = state[query_key]
    return OracleResult(answer, (fact_id,))


def generate_update_examples(
    *,
    split: str,
    count: int,
    base_seed: int = 0,
    deletion_rate: float = 0.25,
    query_delay: int = 4,
    distractor_count: int = 4,
    correction_counts: tuple[int, ...] | None = None,
) -> list[UpdateExample]:
    """Generate a reproducible split with disjoint episode seeds."""
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"unsupported split {split!r}")
    for name, value in (("count", count), ("base_seed", base_seed), ("query_delay", query_delay), ("distractor_count", distractor_count)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if count <= 0 or query_delay < 0 or distractor_count < 0:
        raise ValueError("count must be positive and delay counts must be nonnegative")
    if not isinstance(deletion_rate, (int, float)) or isinstance(deletion_rate, bool):
        raise TypeError("deletion_rate must be numeric")
    if not 0.0 <= deletion_rate <= 1.0:
        raise ValueError("deletion_rate must be between 0 and 1")
    corrections = (
        _DEFAULT_CORRECTIONS[split]
        if correction_counts is None
        else correction_counts
    )
    if not isinstance(corrections, tuple) or not corrections or any(
        isinstance(n, bool) or not isinstance(n, int) or n < 0
        for n in corrections
    ):
        raise ValueError("correction_counts must contain nonnegative integers")

    generated: list[UpdateExample] = []
    for index in range(count):
        episode_seed = base_seed + _SPLIT_SEED_OFFSET[split] + index
        rng = random.Random(episode_seed)
        entity = rng.choice(ENTITIES)
        attribute = rng.choice(ATTRIBUTES)
        available_values = list(VALUES[attribute])
        rng.shuffle(available_values)
        correction_count = corrections[index % len(corrections)]
        needed_values = correction_count + 1
        if needed_values > len(available_values):
            raise ValueError("correction_count exceeds distinct available values")

        lines: list[str] = []
        trailing_distractors = min(query_delay, distractor_count)
        leading_distractors = distractor_count - trailing_distractors
        for distractor_index in range(leading_distractors):
            other_entity = ENTITIES[(ENTITIES.index(entity) + distractor_index + 1) % len(ENTITIES)]
            other_attribute = ATTRIBUTES[(ATTRIBUTES.index(attribute) + 1) % len(ATTRIBUTES)]
            other_value = VALUES[other_attribute][distractor_index % len(VALUES[other_attribute])]
            lines.append(f"DISTRACTOR {other_entity} {other_attribute} {other_value}.")

        lines.append(f"SET {entity} {attribute} {available_values[0]}.")
        relevant_ids = [1]
        relevant_ids[0] = len(lines)
        for value in available_values[1:needed_values]:
            lines.append(f"CORRECT {entity} {attribute} {value}.")
            relevant_ids.append(len(lines))

        deleted = rng.random() < deletion_rate
        if deleted:
            lines.append(f"DELETE {entity} {attribute}.")
            relevant_ids.append(len(lines))

        for distractor_index in range(trailing_distractors):
            other_entity = ENTITIES[(ENTITIES.index(entity) + distractor_index + 1) % len(ENTITIES)]
            other_attribute = ATTRIBUTES[(ATTRIBUTES.index(attribute) + 1) % len(ATTRIBUTES)]
            other_value = VALUES[other_attribute][distractor_index % len(VALUES[other_attribute])]
            lines.append(f"DISTRACTOR {other_entity} {other_attribute} {other_value}.")
        for wait_index in range(query_delay - trailing_distractors):
            lines.append(f"WAIT {wait_index + 1}.")

        context = "\n".join(lines)
        answer = UNKNOWN if deleted else available_values[correction_count]
        source_id = f"correction-deletion:{split}:seed-{episode_seed}"
        example = ReasoningExample(
            dataset="tinymem_updates",
            task_id="correction_deletion",
            split=split,
            context=context,
            question=f"QUERY {entity} {attribute}?",
            answer=answer,
            supporting_fact_ids=(relevant_ids[-1],),
            source_length=len(context),
            source_example_id=source_id,
            context_fact_ids=tuple(range(1, len(lines) + 1)),
        )
        result = interpret_update_example(example)
        if result.answer != example.answer or result.supporting_fact_ids != example.supporting_fact_ids:
            raise RuntimeError("generated update example failed symbolic validation")
        generated.append(UpdateExample(example, episode_seed, correction_count, deleted, query_delay, distractor_count))
    return generated
