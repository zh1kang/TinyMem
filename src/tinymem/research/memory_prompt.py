"""Native prompt fragments with all history-bearing tokens inside memory."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from tinymem.data.reader_gate import ReaderCase
from tinymem.evaluation.reader_gate import reader_messages
from tinymem.research.pretrained import PretrainedReader


_MEMORY_MARKER = "__TINYMEM_NATIVE_HISTORY_BOUNDARY__"


@dataclass(frozen=True)
class NativeMemoryExample:
    case_id: str
    before_ids: tuple[int, ...]
    history_ids: tuple[int, ...]
    after_ids: tuple[int, ...]
    answer_ids: tuple[int, ...]


def encode_memory_example(reader: PretrainedReader, case: ReaderCase) -> NativeMemoryExample:
    """Verify native token equivalence; the writer receives history_ids only."""
    if not case.context or not case.question.strip() or not case.answer.strip():
        raise ValueError("history, question, and answer must be nonempty")
    if _MEMORY_MARKER in case.context or _MEMORY_MARKER in case.question:
        raise ValueError("input contains the reserved memory boundary marker")
    tokenizer = reader.tokenizer
    envelope = tokenizer.apply_chat_template(
        reader_messages(replace(case, context=_MEMORY_MARKER), condition="full_context"),
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    boundary = _MEMORY_MARKER + "\n\n"
    if envelope.count(boundary) != 1:
        raise ValueError("chat template does not expose one supported memory boundary")
    before, after = envelope.split(boundary)
    full_prompt = tokenizer.apply_chat_template(
        reader_messages(case, condition="full_context"), tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    history = case.context + "\n\n"
    if before + history + after != full_prompt:
        raise ValueError("chat template changes with history content")
    before_ids = tuple(tokenizer.encode(before, add_special_tokens=False))
    history_ids = tuple(tokenizer.encode(history, add_special_tokens=False))
    after_ids = tuple(tokenizer.encode(after, add_special_tokens=False))
    native_ids = tuple(tokenizer.encode(full_prompt, add_special_tokens=False))
    if before_ids + history_ids + after_ids != native_ids:
        raise ValueError("tokenizer merges across the memory boundary; this input is unsupported")
    answer_ids = (*tokenizer.encode(case.answer, add_special_tokens=False), tokenizer.eos_token_id)
    if not before_ids or not history_ids or not after_ids or len(answer_ids) < 2 or answer_ids[-1] is None:
        raise ValueError("native prompt fragments, answer, and end-of-turn token are required")
    if len(before_ids) + len(after_ids) + len(answer_ids) > reader.model.config.max_position_embeddings:
        raise ValueError("query envelope and answer exceed reader context before memory is added")
    return NativeMemoryExample(case.case_id, before_ids, history_ids, after_ids, answer_ids)


def encode_history_chunks(
    reader: PretrainedReader, case: ReaderCase, chunks: Sequence[str],
) -> tuple[tuple[int, ...], ...]:
    """Preserve declared line boundaries without changing native history tokens."""
    if isinstance(chunks, str) or not chunks or any(not isinstance(chunk, str) or not chunk for chunk in chunks):
        raise ValueError("chunks must be a nonempty sequence of nonempty strings")
    if "\n".join(chunks) != case.context:
        raise ValueError("chunks must reconstruct the exact source context")
    encoded = tuple(tuple(reader.tokenizer.encode(
        chunk + ("\n\n" if index == len(chunks) - 1 else "\n"), add_special_tokens=False,
    )) for index, chunk in enumerate(chunks))
    expected = tuple(reader.tokenizer.encode(case.context + "\n\n", add_special_tokens=False))
    if any(not chunk for chunk in encoded) or tuple(token for chunk in encoded for token in chunk) != expected:
        raise ValueError("tokenizer merges across a declared history chunk boundary")
    return encoded
