"""Training losses for TinyMem language modeling."""

import torch
from torch.nn import functional as F


def next_token_cross_entropy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute causal next-token cross-entropy from unshifted sequences."""

    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"logits must be a torch.Tensor, got {type(logits)}")
    if logits.ndim != 3:
        raise ValueError(
            f"logits must have shape [batch, sequence, vocab], got {logits.shape}"
        )
    if not logits.is_floating_point():
        raise TypeError(f"logits must be a floating-point tensor, got {logits.dtype}")
    if logits.shape[1] < 2:
        raise ValueError("logits must contain at least two sequence positions")

    if not isinstance(target_ids, torch.Tensor):
        raise TypeError(
            f"target_ids must be a torch.Tensor, got {type(target_ids)}"
        )
    if target_ids.ndim != 2:
        raise ValueError(
            f"target_ids must have shape [batch, sequence], got {target_ids.shape}"
        )
    if target_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"target_ids must be an integer tensor, got dtype {target_ids.dtype}"
        )
    if target_ids.shape != logits.shape[:2]:
        raise ValueError(
            "target_ids batch and sequence dimensions must match logits, got "
            f"{target_ids.shape} and {logits.shape[:2]}"
        )

    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
        raise TypeError(f"ignore_index must be an int, got {type(ignore_index)}")

    shifted_targets = target_ids[:, 1:]
    invalid = (shifted_targets != ignore_index) & (
        (shifted_targets < 0) | (shifted_targets >= logits.shape[-1])
    )
    if invalid.any():
        raise ValueError("target IDs must be in the vocabulary or equal ignore_index")
    if not (shifted_targets != ignore_index).any():
        raise ValueError("next-token targets must contain at least one non-ignored token")

    shifted_logits = logits[:, :-1, :]
    return F.cross_entropy(
        shifted_logits.reshape(-1, logits.shape[-1]),
        shifted_targets.reshape(-1),
        ignore_index=ignore_index,
    )
