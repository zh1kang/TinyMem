"""Build scored persistent-memory candidates from expired raw tokens."""

import torch
from torch import nn

from tinymem.memory.attention_tracker import AttentionScoreState
from tinymem.memory.state import MemoryState
from tinymem.memory.token_window import RawTokenBatch


def build_scored_token_candidates(
    expired_tokens: RawTokenBatch,
    expired_scores: AttentionScoreState,
    token_embedding: nn.Embedding,
) -> MemoryState:
    """Convert aligned expired tokens and scores into memory candidates.

    Input shapes:
        expired_tokens.positions: [tokens]
        expired_tokens.token_ids: [batch, tokens]
        expired_scores.positions: [tokens]
        expired_scores.scores: [batch, tokens]

    Output shapes:
        values: [batch, tokens, model_width]
        valid, positions, token_ids, scores: [batch, tokens]
    """
    if not isinstance(expired_tokens, RawTokenBatch):
        raise TypeError("expired_tokens must be a RawTokenBatch")
    if not isinstance(expired_scores, AttentionScoreState):
        raise TypeError("expired_scores must be an AttentionScoreState")
    if not isinstance(token_embedding, nn.Embedding):
        raise TypeError("token_embedding must be an nn.Embedding")
    if expired_tokens.positions.device != expired_scores.positions.device:
        raise ValueError("expired tokens and scores must be on the same device")
    if not torch.equal(expired_tokens.positions, expired_scores.positions):
        raise ValueError("expired tokens and scores must have identical positions")
    if expired_tokens.token_ids.shape[0] != expired_scores.scores.shape[0]:
        raise ValueError("expired tokens and scores must have the same batch size")
    if expired_tokens.token_ids.device != token_embedding.weight.device:
        raise ValueError("expired tokens and token_embedding must be on the same device")

    token_ids = expired_tokens.token_ids
    if token_ids.numel() > 0 and (
        (token_ids < 0).any() or (token_ids >= token_embedding.num_embeddings).any()
    ):
        raise ValueError(
            "expired token IDs must be in "
            f"[0, {token_embedding.num_embeddings})"
        )

    token_values = token_embedding(token_ids)
    positions = (
        expired_tokens.positions.unsqueeze(0)
        .expand(token_ids.shape[0], -1)
        .clone()
    )
    valid = torch.ones_like(token_ids, dtype=torch.bool)
    return MemoryState(
        values=token_values.detach().clone(),
        valid=valid,
        positions=positions,
        token_ids=token_ids.detach().clone(),
        scores=expired_scores.scores.detach().clone(),
    )
