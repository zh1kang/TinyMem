"""Deterministic sampling helpers for controlled reasoning examples."""

from __future__ import annotations

import random
from collections.abc import Sequence

from tinymem.data.schema import ReasoningExample


def select_reasoning_examples(
    examples: Sequence[ReasoningExample],
    *,
    count: int | None,
    seed: int,
) -> list[ReasoningExample]:
    """Select examples without replacement, or preserve all source examples."""
    if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)):
        raise TypeError("examples must be a sequence")
    if not examples or not all(
        isinstance(example, ReasoningExample) for example in examples
    ):
        raise ValueError("examples must contain ReasoningExample values")
    if count is not None and (isinstance(count, bool) or not isinstance(count, int)):
        raise TypeError("count must be an integer or None")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if count is None:
        return list(examples)
    if count <= 0 or count > len(examples):
        raise ValueError("count must select available examples")
    return random.Random(seed).sample(list(examples), count)
