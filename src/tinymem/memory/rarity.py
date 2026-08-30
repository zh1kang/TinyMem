"""Query-independent token-rarity scoring for extractive memory."""

import math

import torch

from tinymem.memory.state import INTEGER_DTYPES, MemoryState


def count_token_frequencies(
    token_ids: torch.Tensor,
    valid: torch.Tensor,
    *,
    vocab_size: int,
) -> torch.Tensor:
    """Count valid training-token occurrences for each vocabulary entry."""
    if not isinstance(token_ids, torch.Tensor):
        raise TypeError("token_ids must be a torch.Tensor")
    if not isinstance(valid, torch.Tensor):
        raise TypeError("valid must be a torch.Tensor")
    if token_ids.dtype not in INTEGER_DTYPES:
        raise TypeError("token_ids must be an integer tensor")
    if valid.dtype != torch.bool:
        raise TypeError("valid must be a boolean tensor")
    if token_ids.shape != valid.shape:
        raise ValueError("token_ids and valid must have the same shape")
    if token_ids.device != valid.device:
        raise ValueError("token_ids and valid must be on the same device")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
        raise TypeError("vocab_size must be an integer")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")

    valid_token_ids = token_ids[valid]
    if valid_token_ids.numel() > 0:
        if (valid_token_ids < 0).any() or (valid_token_ids >= vocab_size).any():
            raise ValueError(
                f"valid token IDs must be in [0, {vocab_size})"
            )
    return torch.bincount(valid_token_ids, minlength=vocab_size)


class TokenRarityScorer:
    """Score tokens using negative log smoothed training frequency."""

    def __init__(
        self,
        token_counts: torch.Tensor,
        *,
        smoothing: float = 1.0,
    ) -> None:
        if not isinstance(token_counts, torch.Tensor):
            raise TypeError("token_counts must be a torch.Tensor")
        if token_counts.ndim != 1 or token_counts.numel() == 0:
            raise ValueError("token_counts must be a nonempty one-dimensional tensor")
        if token_counts.dtype not in INTEGER_DTYPES:
            raise TypeError("token_counts must be an integer tensor")
        if (token_counts < 0).any():
            raise ValueError("token_counts must be nonnegative")
        if token_counts.sum().item() == 0:
            raise ValueError("token_counts must contain at least one observation")
        if isinstance(smoothing, bool) or not isinstance(smoothing, (int, float)):
            raise TypeError("smoothing must be a real number")
        if not math.isfinite(smoothing) or smoothing <= 0:
            raise ValueError("smoothing must be finite and positive")

        self.token_counts = token_counts.detach().clone().to(torch.float32)
        self.smoothing = float(smoothing)
        self.vocab_size = self.token_counts.numel()
        self.total_count = self.token_counts.sum()

    def score(
        self,
        token_ids: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Return one rarity score per slot and negative infinity when invalid."""
        if not isinstance(token_ids, torch.Tensor):
            raise TypeError("token_ids must be a torch.Tensor")
        if not isinstance(valid, torch.Tensor):
            raise TypeError("valid must be a torch.Tensor")
        if token_ids.dtype not in INTEGER_DTYPES:
            raise TypeError("token_ids must be an integer tensor")
        if valid.dtype != torch.bool:
            raise TypeError("valid must be a boolean tensor")
        if token_ids.shape != valid.shape:
            raise ValueError("token_ids and valid must have the same shape")
        if token_ids.device != valid.device:
            raise ValueError("token_ids and valid must be on the same device")

        scores = torch.full(
            token_ids.shape,
            float("-inf"),
            device=token_ids.device,
            dtype=torch.float32,
        )
        valid_token_ids = token_ids[valid]
        if valid_token_ids.numel() == 0:
            return scores
        if (valid_token_ids < 0).any() or (
            valid_token_ids >= self.vocab_size
        ).any():
            raise ValueError(
                f"valid token IDs must be in [0, {self.vocab_size})"
            )

        counts = self.token_counts.to(token_ids.device)[valid_token_ids]
        counts = counts + self.smoothing
        total = self.total_count + self.smoothing * self.vocab_size
        probabilities = counts / total
        scores[valid] = -torch.log(probabilities)
        return scores

    def score_state(self, state: MemoryState) -> MemoryState:
        """Return a fresh state with rarity scores and unchanged provenance."""
        if not isinstance(state, MemoryState):
            raise TypeError("state must be a MemoryState")
        if state.token_ids is None:
            raise ValueError("token-rarity scoring requires token_ids")

        scores = self.score(state.token_ids, state.valid)
        return MemoryState(
            values=state.values.clone(),
            valid=state.valid.clone(),
            positions=state.positions.clone(),
            token_ids=state.token_ids.clone(),
            scores=scores,
        )
