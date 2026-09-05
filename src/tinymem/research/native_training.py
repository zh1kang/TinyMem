"""Answer supervision for several queries sharing one query-blind history state."""

from collections.abc import Sequence

import torch

from tinymem.research.memory_prompt import NativeMemoryExample
from tinymem.research.prefix_reader import prefix_answer_loss
from tinymem.research.pretrained import PretrainedReader
from tinymem.research.recurrent_memory import NativeRecurrentMemory


def native_history_answer_loss(
    reader: PretrainedReader, writer: NativeRecurrentMemory,
    examples: Sequence[NativeMemoryExample],
    *, history_chunks: Sequence[tuple[int, ...]] | None = None,
) -> torch.Tensor:
    """Write history once, then average its query losses without detaching state."""
    if not examples or not examples[0].history_ids:
        raise ValueError("a nonempty history and at least one query are required")
    first = examples[0]
    if any(example.history_ids != first.history_ids or example.before_ids != first.before_ids for example in examples):
        raise ValueError("queries must share one history and native prompt opening")
    if len({example.after_ids for example in examples}) != len(examples):
        raise ValueError("queries within a history must be distinct")
    if history_chunks is None:
        history_chunks = tuple(first.history_ids[start:start + writer.segment_length]
                               for start in range(0, len(first.history_ids), writer.segment_length))
    if not history_chunks or any(not chunk or len(chunk) > writer.segment_length for chunk in history_chunks):
        raise ValueError("history chunks must be nonempty and within segment_length")
    if any(isinstance(token, bool) or not isinstance(token, int) for chunk in history_chunks for token in chunk):
        raise TypeError("history chunks must contain integer token IDs")
    if tuple(token for chunk in history_chunks for token in chunk) != first.history_ids:
        raise ValueError("history chunks must reconstruct the exact shared history")
    if any(not 0 <= token < reader.model.config.vocab_size for chunk in history_chunks for token in chunk):
        raise ValueError("history chunks contain an out-of-vocabulary token")
    state = writer.writer.empty(1)
    for chunk in history_chunks:
        ids = torch.tensor(chunk, device=reader.model.device)
        state = writer.write(reader, state, ids)
    memory = writer.memory_vectors(state)
    losses = []
    for example in examples:
        before, after, answer = (torch.tensor(ids, device=reader.model.device) for ids in (example.before_ids, example.after_ids, example.answer_ids))
        losses.append(prefix_answer_loss(reader, before, memory, after, answer))
    return torch.stack(losses).mean()
