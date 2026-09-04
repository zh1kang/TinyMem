"""Delayed-recall construction for byte-level conversational QA.

A filler session of ordinary WikiText text is inserted between the controlled
facts and the question so that the evidence sits a known number of bytes
before the answer position.  This is the byte-level analogue of the
distractor-token delays used by the controlled-vocabulary path.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from numbers import Integral

from tinymem.data.schema import ReasoningExample
from tinymem.tokenization.byte_tokenizer import ByteTokenizer
from tinymem.training.conversational_qa import (
    ByteQAExample,
    format_conversational_qa_prompt,
)


QUESTION_MARKER = "[question | undated]\n"
FILLER_SESSION_HEADER = "[session wikitext-2 | undated]\n"


def sample_filler(
    text: str,
    *,
    byte_length: int,
    rng: random.Random,
) -> str:
    """Return a single-line slice of ``text`` whose UTF-8 size is ``byte_length``.

    Multi-byte characters are trimmed at the end, so the encoded size may fall
    short by at most three bytes.
    """
    if not isinstance(text, str) or not text:
        raise ValueError("text must be a nonempty string")
    if isinstance(byte_length, bool) or not isinstance(byte_length, Integral):
        raise TypeError("byte_length must be an integer")
    if byte_length <= 0:
        raise ValueError("byte_length must be positive")
    if not isinstance(rng, random.Random):
        raise TypeError("rng must be a random.Random")
    encoded_text_length = len(text.encode("utf-8"))
    if byte_length > encoded_text_length:
        raise ValueError("byte_length exceeds the filler text")

    maximum_start = max(0, len(text) - int(byte_length))
    start = rng.randrange(0, maximum_start + 1)
    window = text[start : start + int(byte_length)].replace("\n", " ")
    encoded = window.encode("utf-8")[: int(byte_length)]
    return encoded.decode("utf-8", errors="ignore")


def format_delayed_prompt(example: ReasoningExample, filler: str) -> str:
    """Insert a filler session between the controlled facts and the question."""
    if not isinstance(filler, str) or not filler:
        raise ValueError("filler must be a nonempty string")
    if "\n" in filler:
        raise ValueError("filler must be a single line")
    prompt = format_conversational_qa_prompt(example)
    head, marker, tail = prompt.rpartition(QUESTION_MARKER)
    if not marker:
        raise ValueError("prompt does not contain the question marker")
    return f"{head}{FILLER_SESSION_HEADER}User: {filler}\n{marker}{tail}"


def build_delayed_examples(
    examples: Sequence[ReasoningExample],
    tokenizer: ByteTokenizer,
    *,
    filler_text: str,
    filler_bytes: int,
    seed: int,
) -> list[ByteQAExample]:
    """Encode examples with ``filler_bytes`` of deterministic filler each.

    ``filler_bytes == 0`` reproduces the undelayed conversational prompt.
    """
    if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)):
        raise TypeError("examples must be a sequence")
    if not examples or not all(
        isinstance(example, ReasoningExample) for example in examples
    ):
        raise ValueError("examples must contain ReasoningExample values")
    if not isinstance(tokenizer, ByteTokenizer):
        raise TypeError("tokenizer must be a ByteTokenizer")
    if isinstance(filler_bytes, bool) or not isinstance(filler_bytes, Integral):
        raise TypeError("filler_bytes must be an integer")
    if filler_bytes < 0:
        raise ValueError("filler_bytes must be nonnegative")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")

    rng = random.Random(int(seed))
    encoded = []
    for example in examples:
        if filler_bytes == 0:
            prompt = format_conversational_qa_prompt(example)
        else:
            prompt = format_delayed_prompt(
                example,
                sample_filler(filler_text, byte_length=int(filler_bytes), rng=rng),
            )
        encoded.append(
            ByteQAExample(
                prompt_ids=tuple(tokenizer.encode(prompt)),
                answer_ids=tuple(tokenizer.encode(example.answer)),
                source_example_id=example.source_example_id,
                dataset=example.dataset,
                task_id=example.task_id,
                split=example.split,
            )
        )
    return encoded


def build_mixed_delay_examples(
    examples: Sequence[ReasoningExample],
    tokenizer: ByteTokenizer,
    *,
    filler_text: str,
    max_filler_bytes: int,
    delayed_fraction: float,
    seed: int,
) -> list[ByteQAExample]:
    """Delay a random fraction of examples by a uniform random filler length.

    This is the training-time curriculum: each selected example receives
    between one and ``max_filler_bytes`` bytes of filler so that its evidence
    can fall beyond the local segment while the remaining examples stay short.
    """
    if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)):
        raise TypeError("examples must be a sequence")
    if not examples or not all(
        isinstance(example, ReasoningExample) for example in examples
    ):
        raise ValueError("examples must contain ReasoningExample values")
    if isinstance(max_filler_bytes, bool) or not isinstance(
        max_filler_bytes,
        Integral,
    ):
        raise TypeError("max_filler_bytes must be an integer")
    if max_filler_bytes <= 0:
        raise ValueError("max_filler_bytes must be positive")
    if isinstance(delayed_fraction, bool) or not isinstance(
        delayed_fraction,
        (int, float),
    ):
        raise TypeError("delayed_fraction must be a real number")
    if not 0 <= delayed_fraction <= 1:
        raise ValueError("delayed_fraction must be in [0, 1]")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")

    rng = random.Random(int(seed))
    encoded = []
    for example in examples:
        delay = 0
        if rng.random() < delayed_fraction:
            delay = rng.randint(1, int(max_filler_bytes))
        encoded.extend(
            build_delayed_examples(
                [example],
                tokenizer,
                filler_text=filler_text,
                filler_bytes=delay,
                seed=rng.randrange(0, 2**31),
            )
        )
    return encoded
