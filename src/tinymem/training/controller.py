"""Training objectives for adaptive memory-write decisions."""

import torch

from tinymem.memory.controller import WRITE_ACTION


def controller_write_cost(
    assignments: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Return the mean straight-through write rate over valid segments."""
    if not isinstance(assignments, torch.Tensor):
        raise TypeError("assignments must be a torch.Tensor")
    if not isinstance(valid, torch.Tensor):
        raise TypeError("valid must be a torch.Tensor")
    if assignments.ndim != 3 or assignments.shape[2] != 2:
        raise ValueError("assignments must have shape [batch, segments, 2]")
    if not assignments.is_floating_point():
        raise TypeError("assignments must be floating point")
    if valid.shape != assignments.shape[:2]:
        raise ValueError(f"valid must have shape {assignments.shape[:2]}")
    if valid.dtype != torch.bool:
        raise TypeError("valid must be a boolean tensor")
    if valid.device != assignments.device:
        raise ValueError("valid and assignments must share a device")
    if not valid.any():
        raise ValueError("write cost requires a valid segment")

    weights = valid.to(dtype=assignments.dtype)
    writes = assignments[:, :, WRITE_ACTION]
    return (writes * weights).sum() / weights.sum()
