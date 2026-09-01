"""Differentiable fixed-capacity updates for continuous memory."""

from numbers import Integral

import torch
from torch import nn


class RecurrentMemoryBank(nn.Module):
    """Append valid summaries to a fixed-capacity continuous memory bank.

    Inputs:
        memory:        [batch, capacity, model_width]
        memory_valid:  [batch, capacity]
        summary:       [batch, 1, model_width]
        summary_valid: [batch, 1]

    Outputs:
        next_memory:       [batch, capacity, model_width]
        next_memory_valid: [batch, capacity]
    """

    def __init__(self, *, capacity: int, model_width: int) -> None:
        super().__init__()
        dimensions = {
            "capacity": capacity,
            "model_width": model_width,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        self.capacity = int(capacity)
        self.model_width = int(model_width)

    def forward(
        self,
        memory: torch.Tensor,
        memory_valid: torch.Tensor,
        summary: torch.Tensor,
        summary_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the next bank without modifying any input tensor in place."""
        tensors = {
            "memory": memory,
            "memory_valid": memory_valid,
            "summary": summary,
            "summary_valid": summary_valid,
        }
        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")

        if memory.ndim != 3:
            raise ValueError(
                "memory must have shape [batch, capacity, model_width]"
            )
        if not memory.is_floating_point():
            raise TypeError("memory must be a floating-point tensor")
        if memory.shape[0] == 0:
            raise ValueError("memory must contain at least one batch row")
        expected_memory_shape = (
            memory.shape[0],
            self.capacity,
            self.model_width,
        )
        if memory.shape != expected_memory_shape:
            raise ValueError(f"memory must have shape {expected_memory_shape}")

        expected_valid_shape = expected_memory_shape[:2]
        if memory_valid.shape != expected_valid_shape:
            raise ValueError(
                f"memory_valid must have shape {expected_valid_shape}"
            )
        if memory_valid.dtype != torch.bool:
            raise TypeError("memory_valid must be a boolean tensor")

        expected_summary_shape = (memory.shape[0], 1, self.model_width)
        if summary.shape != expected_summary_shape:
            raise ValueError(f"summary must have shape {expected_summary_shape}")
        if not summary.is_floating_point():
            raise TypeError("summary must be a floating-point tensor")

        expected_summary_valid_shape = expected_summary_shape[:2]
        if summary_valid.shape != expected_summary_valid_shape:
            raise ValueError(
                "summary_valid must have shape "
                f"{expected_summary_valid_shape}"
            )
        if summary_valid.dtype != torch.bool:
            raise TypeError("summary_valid must be a boolean tensor")

        if any(tensor.device != memory.device for tensor in tensors.values()):
            raise ValueError("all recurrent memory tensors must share a device")
        if summary.dtype != memory.dtype:
            raise TypeError("summary and memory must share a dtype")

        shifted_memory = torch.cat((memory[:, 1:, :], summary), dim=1)
        shifted_valid = torch.cat(
            (memory_valid[:, 1:], summary_valid),
            dim=1,
        )
        next_memory = torch.where(
            summary_valid.unsqueeze(-1),
            shifted_memory,
            memory,
        )
        next_memory_valid = torch.where(
            summary_valid,
            shifted_valid,
            memory_valid,
        )
        return next_memory, next_memory_valid
