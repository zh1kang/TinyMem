"""Joint answer training with only serialized memory surviving a text stream."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from tinymem.data.reader_gate import ReaderCase
from tinymem.memory.quantized_slots import QuantizedSlotMemory
from tinymem.research.memory_prompt import encode_memory_example
from tinymem.research.prefix_reader import prefix_answer_losses
from tinymem.research.storage_frontier_data import StorageQuestion


@dataclass(frozen=True)
class AnswerTokens:
    before: tuple[int, ...]
    after: tuple[int, ...]
    answer: tuple[int, ...]


def encode_answer(reader, question: StorageQuestion) -> AnswerTokens:
    # The placeholder establishes the native envelope; it is never read or written.
    case = ReaderCase(question.id, question.task, question.story,
                      'Memory.', question.question, question.answer)
    native = encode_memory_example(reader, case)
    return AnswerTokens(native.before_ids, native.after_ids, native.answer_ids)


def rollout(memory: QuantizedSlotMemory, histories: Sequence[Sequence[str]],
            features: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Each feature matrix is from this record alone, with reset positions/cache."""
    if not histories or any(not history for history in histories):
        raise ValueError('each history must contain at least one text record')
    state = memory.empty(len(histories))
    for step in range(max(map(len, histories))):
        current = [features[history[step]] if step < len(history) else None for history in histories]
        length = max(feature.shape[0] for feature in current if feature is not None)
        hidden = state.new_zeros(len(histories), length, memory.reader_width)
        valid = torch.zeros(len(histories), length, dtype=torch.bool, device=state.device)
        for row, feature in enumerate(current):
            if feature is None:
                continue
            if (feature.requires_grad or feature.ndim != 2 or feature.shape[1] != memory.reader_width
                    or feature.dtype != torch.float32 or feature.shape[0] == 0):
                raise ValueError('features must be detached FP32 current-record token matrices')
            hidden[row, :len(feature)] = feature.to(state.device)
            valid[row, :len(feature)] = True
        state = memory(state, hidden, valid)
    return state


def answer_loss(reader, tokens: Sequence[AnswerTokens], memories: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(tokens) != len(memories) or not tokens:
        raise ValueError('one memory is required per answer')
    device = reader.model.device
    reads = [(torch.tensor(t.before, device=device), m,
              torch.tensor(t.after, device=device), torch.tensor(t.answer, device=device))
             for t, m in zip(tokens, memories, strict=True)]
    return prefix_answer_losses(reader, reads).mean()


def text_vectors(reader, texts: Sequence[str]) -> list[torch.Tensor]:
    embedding = reader.model.get_input_embeddings()
    return [embedding(torch.tensor(reader.tokenizer.encode(text + '\n\n', add_special_tokens=False),
                                   dtype=torch.long, device=reader.model.device))
            if text else embedding.weight.new_empty((0, embedding.embedding_dim)) for text in texts]


def optimizer_step(reader, memory: QuantizedSlotMemory | None, adapters: tuple,
                   optimizer: torch.optim.Optimizer, loss: torch.Tensor) -> dict[str, float]:
    parameters = (*tuple(memory.parameters()), *adapters) if memory is not None else adapters
    owned = {id(p) for p in parameters}
    optimized = [p for group in optimizer.param_groups for p in group['params']]
    if (len(owned) != len(parameters) or len(optimized) != len(parameters)
            or {id(p) for p in optimized} != owned
            or {id(p) for p in reader.model.parameters() if p.requires_grad} != {id(p) for p in adapters}):
        raise ValueError('optimizer must own exactly the writer, bridge, and declared adapters')
    if reader.model.training or not bool(torch.isfinite(loss)):
        raise ValueError('reader must be in evaluation mode and loss must be finite')
    loss.backward()
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
        raise ValueError('missing or nonfinite gradient')
    if any(p.grad is not None for p in reader.model.parameters() if id(p) not in owned):
        raise ValueError('frozen base received a gradient')
    groups = {'adapter': adapters}
    if memory is not None:
        groups['writer'] = tuple(p for n, p in memory.named_parameters() if not n.startswith('memory_projection.'))
        groups['bridge'] = tuple(memory.memory_projection.parameters())
    norms = {name + '_gradient_norm': float(torch.stack([p.grad.float().square().sum() for p in group]).sum().sqrt())
             for name, group in groups.items()}
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in parameters):
        raise ValueError('optimizer produced nonfinite parameters')
    return {'answer_ce': float(loss.detach()), 'gradient_norm': float(norm), **norms}
