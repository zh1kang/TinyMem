"""Inference-only token scores for complete prefix-reader answers."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from tinymem.research.prefix_reader import _check_ids, _forward, _prompt_embeddings
from tinymem.research.pretrained import PretrainedReader


@torch.inference_mode()
def prefix_answer_log_probs(
    reader: PretrainedReader,
    before_ids: torch.Tensor,
    memory: torch.Tensor,
    after_ids: torch.Tensor,
    answer_ids: torch.Tensor,
) -> dict[str, object]:
    """Return per-token log probabilities for a complete native answer.

    ``answer_ids`` must contain at least one answer token followed by the
    tokenizer's native end-of-sequence token.
    """
    if any(module.training for module in reader.model.modules()):
        raise ValueError("log-probability scoring requires the reader in evaluation mode")
    if any(parameter.requires_grad or parameter.grad is not None for parameter in reader.model.parameters()):
        raise ValueError("log-probability scoring requires a frozen reader")
    _check_ids(reader, answer_ids, "answer_ids")
    if answer_ids.numel() < 2:
        raise ValueError("answer_ids must contain at least two tokens")
    eos_token_id = reader.tokenizer.eos_token_id
    if not isinstance(eos_token_id, int) or int(answer_ids[-1]) != eos_token_id:
        raise ValueError("answer_ids must end with the native EOS token")

    prompt = _prompt_embeddings(
        reader, before_ids, memory, after_ids, continuation_positions=answer_ids.numel() - 1,
    )
    answer_prefix = reader.model.get_input_embeddings()(answer_ids[:-1]).unsqueeze(0)
    logits = _forward(reader, torch.cat((prompt, answer_prefix), dim=1), answer_ids.numel())
    log_probs = F.log_softmax(logits[0].float(), dim=-1)
    token_log_probs = log_probs.gather(1, answer_ids.long().unsqueeze(1)).squeeze(1).tolist()
    sequence_log_probability = sum(token_log_probs)
    return {
        "answer_ids": answer_ids.tolist(),
        "token_log_probs": token_log_probs,
        "sequence_log_probability": sequence_log_probability,
        "mean_answer_ce": -sequence_log_probability / len(token_log_probs),
    }
