"""Validated persistent-memory input for Transformer attention."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AttentionMemory:
    """Provide read-only memory values and provenance to one attention layer.

    values:    [batch, slots, model_width]
    valid:     [batch, slots]
    positions: [batch, slots]
    """

    values: torch.Tensor
    valid: torch.Tensor
    positions: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.values, torch.Tensor):
            raise TypeError("values must be a torch.Tensor")
        if not isinstance(self.valid, torch.Tensor):
            raise TypeError("valid must be a torch.Tensor")
        if not isinstance(self.positions, torch.Tensor):
            raise TypeError("positions must be a torch.Tensor")
        if self.values.ndim != 3:
            raise ValueError("values must have shape [batch, slots, model_width]")
        if not self.values.is_floating_point():
            raise TypeError("values must be a floating-point tensor")
        if self.values.shape[0] == 0:
            raise ValueError("values must contain at least one batch row")

        expected_shape = self.values.shape[:2]
        if self.valid.ndim != 2 or self.valid.shape != expected_shape:
            raise ValueError(f"valid must have shape {expected_shape}")
        if self.valid.dtype != torch.bool:
            raise TypeError("valid must be a boolean tensor")
        if self.positions.ndim != 2 or self.positions.shape != expected_shape:
            raise ValueError(f"positions must have shape {expected_shape}")
        if self.positions.dtype not in (torch.int32, torch.int64):
            raise TypeError("positions must be an integer tensor")
        if self.valid.device != self.values.device:
            raise ValueError("valid and values must be on the same device")
        if self.positions.device != self.values.device:
            raise ValueError("positions and values must be on the same device")
        if (self.positions[self.valid] < 0).any():
            raise ValueError("valid memory positions must be nonnegative")

    @property
    def slot_count(self) -> int:
        """Return the fixed number of memory slots."""
        return self.values.shape[1]
