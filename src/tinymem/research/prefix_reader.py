"""Cache-free single-query reads from raw-token or continuous memory inputs."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from tinymem.research.pretrained import PretrainedReader


def _check_ids(reader: PretrainedReader, ids: torch.Tensor, name: str) -> None:
    if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be a one-dimensional integer tensor")
    if ids.device != reader.model.device:
        raise ValueError(f"{name} and reader must share a device")
    if ((ids < 0) | (ids >= reader.model.config.vocab_size)).any():
        raise ValueError(f"{name} contains an out-of-vocabulary token")


def _prompt_embeddings(
    reader: PretrainedReader, before_ids: torch.Tensor, memory: torch.Tensor,
    after_ids: torch.Tensor, *, continuation_positions: int = 0,
) -> torch.Tensor:
    _check_ids(reader, before_ids, "before_ids")
    _check_ids(reader, after_ids, "after_ids")
    if after_ids.numel() == 0:
        raise ValueError("after_ids must contain the question and assistant suffix")
    embedding = reader.model.get_input_embeddings()
    if memory.ndim != 2 or memory.shape[1] != embedding.embedding_dim or not memory.is_floating_point():
        raise ValueError("memory must have floating [positions, reader_width] shape")
    if memory.device != reader.model.device:
        raise ValueError("memory and reader must share a device")
    total_positions = before_ids.numel() + memory.shape[0] + after_ids.numel() + continuation_positions
    if total_positions > reader.model.config.max_position_embeddings:
        raise ValueError("input and continuation exceed reader context; truncation is forbidden")
    reader_memory = memory.to(dtype=embedding.weight.dtype)
    if not torch.isfinite(reader_memory).all():
        raise ValueError("memory must be finite in the reader dtype")
    return torch.cat((embedding(before_ids), reader_memory, embedding(after_ids))).unsqueeze(0)


def _forward(reader: PretrainedReader, embeddings: torch.Tensor, keep: int) -> torch.Tensor:
    length = embeddings.shape[1]
    if length > reader.model.config.max_position_embeddings:
        raise ValueError("input exceeds reader context; truncation is forbidden")
    positions = torch.arange(length, device=embeddings.device).unsqueeze(0)
    output = reader.model(
        inputs_embeds=embeddings, attention_mask=torch.ones_like(positions),
        position_ids=positions, use_cache=False, logits_to_keep=keep,
    )
    return output.logits


def prefix_answer_loss(
    reader: PretrainedReader, before_ids: torch.Tensor, memory: torch.Tensor,
    after_ids: torch.Tensor, answer_ids: torch.Tensor,
) -> torch.Tensor:
    """Preserve gradients into memory; answer_ids includes the end-of-turn token."""
    _check_ids(reader, answer_ids, "answer_ids")
    if answer_ids.numel() == 0:
        raise ValueError("answer_ids must be nonempty")
    prompt = _prompt_embeddings(reader, before_ids, memory, after_ids, continuation_positions=answer_ids.numel() - 1)
    answer_prefix = reader.model.get_input_embeddings()(answer_ids[:-1]).unsqueeze(0)
    logits = _forward(reader, torch.cat((prompt, answer_prefix), dim=1), answer_ids.numel())
    return F.cross_entropy(logits[0].float(), answer_ids.long())


@torch.inference_mode()
def generate_prefix_answer(
    reader: PretrainedReader, before_ids: torch.Tensor, memory: torch.Tensor,
    after_ids: torch.Tensor, *, max_new_tokens: int = 16,
) -> dict[str, object]:
    """Greedily recompute the supplied prompt and suffix; retain no hidden cache."""
    if reader.model.training:
        raise ValueError("generation requires the reader in evaluation mode")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    prompt = _prompt_embeddings(reader, before_ids, memory, after_ids, continuation_positions=max_new_tokens)
    eos = reader.model.generation_config.eos_token_id
    stop_ids = {eos} if isinstance(eos, int) else set(eos or ())
    generated = []
    for _ in range(max_new_tokens):
        ids = torch.tensor(generated, dtype=torch.long, device=prompt.device)
        suffix = reader.model.get_input_embeddings()(ids).unsqueeze(0)
        logits = _forward(reader, torch.cat((prompt, suffix), dim=1), 1)
        token = int(logits[0, -1].argmax())
        generated.append(token)
        if token in stop_ids:
            break
    return {
        "prediction": reader.tokenizer.decode(generated, skip_special_tokens=True),
        "generated_ids": generated, "input_positions": prompt.shape[1],
        "memory_positions": memory.shape[0], "native_envelope_tokens": before_ids.numel() + after_ids.numel(),
    }
