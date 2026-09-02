"""Training objectives for adaptive memory-write decisions."""

import torch

from tinymem.memory.controller import WRITE_ACTION


def controller_write_cost(
    probabilities: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Return the mean write probability over valid segments."""
    if not isinstance(probabilities, torch.Tensor):
        raise TypeError("probabilities must be a torch.Tensor")
    if not isinstance(valid, torch.Tensor):
        raise TypeError("valid must be a torch.Tensor")
    if probabilities.ndim != 3 or probabilities.shape[2] != 2:
        raise ValueError("probabilities must have shape [batch, segments, 2]")
    if not probabilities.is_floating_point():
        raise TypeError("probabilities must be floating point")
    if valid.shape != probabilities.shape[:2]:
        raise ValueError(f"valid must have shape {probabilities.shape[:2]}")
    if valid.dtype != torch.bool:
        raise TypeError("valid must be a boolean tensor")
    if valid.device != probabilities.device:
        raise ValueError("valid and probabilities must share a device")
    if not valid.any():
        raise ValueError("write cost requires a valid segment")

    weights = valid.to(dtype=probabilities.dtype)
    writes = probabilities[:, :, WRITE_ACTION]
    return (writes * weights).sum() / weights.sum()
