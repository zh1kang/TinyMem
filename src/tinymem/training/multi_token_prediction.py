"""Losses and diagnostics for independent future-token heads."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class MultiTokenLoss:
    """Hold the mean auxiliary loss and each horizon loss."""

    total: torch.Tensor
    by_horizon: dict[int, torch.Tensor]


def multi_token_cross_entropy(
    logits_by_horizon: dict[int, torch.Tensor],
    target_ids: torch.Tensor,
    token_valid: torch.Tensor,
) -> MultiTokenLoss:
    """Align each position with the token at its requested future offset."""
    if not isinstance(logits_by_horizon, dict):
        raise TypeError("logits_by_horizon must be a dictionary")
    if not logits_by_horizon:
        raise ValueError("logits_by_horizon must be nonempty")
    if not isinstance(target_ids, torch.Tensor):
        raise TypeError("target_ids must be a torch.Tensor")
    if target_ids.ndim != 2:
        raise ValueError("target_ids must have shape [batch, tokens]")
    if target_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("target_ids must be an integer tensor")
    if not isinstance(token_valid, torch.Tensor):
        raise TypeError("token_valid must be a torch.Tensor")
    if token_valid.shape != target_ids.shape:
        raise ValueError(f"token_valid must have shape {target_ids.shape}")
    if token_valid.dtype != torch.bool:
        raise TypeError("token_valid must be a boolean tensor")
    if token_valid.device != target_ids.device:
        raise ValueError("token_valid and target_ids must share a device")

    horizons = tuple(logits_by_horizon)
    if any(isinstance(horizon, bool) or not isinstance(horizon, int) for horizon in horizons):
        raise TypeError("horizons must be integers")
    if any(horizon < 2 for horizon in horizons):
        raise ValueError("auxiliary horizons must be at least two")
    if tuple(sorted(set(horizons))) != horizons:
        raise ValueError("horizons must be unique and strictly increasing")

    losses: dict[int, torch.Tensor] = {}
    vocabulary_size: int | None = None
    for horizon, logits in logits_by_horizon.items():
        if not isinstance(logits, torch.Tensor):
            raise TypeError("logits must be torch.Tensor values")
        if logits.ndim != 3:
            raise ValueError("logits must have shape [batch, tokens, vocabulary]")
        if logits.shape[:2] != target_ids.shape:
            raise ValueError("logits and target_ids must share batch and token dimensions")
        if not logits.is_floating_point():
            raise TypeError("logits must be floating point")
        if logits.device != target_ids.device:
            raise ValueError("logits and target_ids must share a device")
        if vocabulary_size is None:
            vocabulary_size = logits.shape[-1]
        elif logits.shape[-1] != vocabulary_size:
            raise ValueError("all horizons must use the same vocabulary size")
        if target_ids.shape[1] <= horizon:
            raise ValueError(f"sequence must be longer than horizon {horizon}")

        aligned_valid = (
            token_valid[:, :-horizon] & token_valid[:, horizon:]
        )
        if not aligned_valid.any():
            raise ValueError(f"horizon {horizon} has no valid targets")
        aligned_targets = target_ids[:, horizon:]
        assert vocabulary_size is not None
        invalid = aligned_valid & (
            (aligned_targets < 0) | (aligned_targets >= vocabulary_size)
        )
        if invalid.any():
            raise ValueError("valid target IDs must be in the vocabulary")
        masked_targets = aligned_targets.masked_fill(~aligned_valid, -100)
        losses[horizon] = F.cross_entropy(
            logits[:, :-horizon, :].reshape(-1, vocabulary_size),
            masked_targets.reshape(-1),
            ignore_index=-100,
        )

    return MultiTokenLoss(
        total=torch.stack(tuple(losses.values())).mean(),
        by_horizon=losses,
    )
