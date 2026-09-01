"""Differentiable fixed-capacity updates for continuous memory."""

from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F


class RecurrentMemoryBank(nn.Module):
    """Append valid summaries to a fixed-capacity continuous memory bank.

    Inputs:
        memory:        [batch, capacity, model_width]
        memory_valid:  [batch, capacity]
        summary:       [batch, writes, model_width]
        summary_valid: [batch, writes]

    Outputs:
        next_memory:       [batch, capacity, model_width]
        next_memory_valid: [batch, capacity]
        write_applied:     [batch, 1]
        write_logits:      [batch, 1] for learned gates, otherwise None
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
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
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

        if summary.ndim != 3:
            raise ValueError("summary must have shape [batch, writes, model_width]")
        if summary.shape[0] != memory.shape[0]:
            raise ValueError("summary and memory must have the same batch size")
        if summary.shape[2] != self.model_width:
            raise ValueError(f"summary model width must be {self.model_width}")
        write_count = summary.shape[1]
        if write_count == 0 or write_count > self.capacity:
            raise ValueError("summary writes must be between one and capacity")
        if not summary.is_floating_point():
            raise TypeError("summary must be a floating-point tensor")

        expected_summary_valid_shape = (memory.shape[0], write_count)
        if summary_valid.shape != expected_summary_valid_shape:
            raise ValueError(
                "summary_valid must have shape "
                f"{expected_summary_valid_shape}"
            )
        if summary_valid.dtype != torch.bool:
            raise TypeError("summary_valid must be a boolean tensor")
        if (summary_valid != summary_valid[:, :1]).any():
            raise ValueError("all summary slots in a row must share validity")

        if any(tensor.device != memory.device for tensor in tensors.values()):
            raise ValueError("all recurrent memory tensors must share a device")
        if summary.dtype != memory.dtype:
            raise TypeError("summary and memory must share a dtype")

        shifted_memory = torch.cat(
            (memory[:, write_count:, :], summary),
            dim=1,
        )
        shifted_valid = torch.cat(
            (memory_valid[:, write_count:], summary_valid),
            dim=1,
        )
        row_valid = summary_valid[:, :1]
        next_memory = torch.where(
            row_valid.unsqueeze(-1),
            shifted_memory,
            memory,
        )
        next_memory_valid = torch.where(
            row_valid,
            shifted_valid,
            memory_valid,
        )
        return next_memory, next_memory_valid, row_valid, None


class GatedRecurrentMemoryBank(RecurrentMemoryBank):
    """Learn when a candidate group should replace the oldest memory slots."""

    def __init__(self, *, capacity: int, model_width: int) -> None:
        super().__init__(capacity=capacity, model_width=model_width)
        self.write_score = nn.Linear(self.model_width, 1)
        nn.init.zeros_(self.write_score.weight)
        nn.init.ones_(self.write_score.bias)

    def forward(
        self,
        memory: torch.Tensor,
        memory_valid: torch.Tensor,
        summary: torch.Tensor,
        summary_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply a straight-through learned write decision per batch row."""
        shifted_memory, shifted_valid, row_valid, _ = super().forward(
            memory,
            memory_valid,
            summary,
            summary_valid,
        )
        candidate = F.normalize(summary.mean(dim=1), dim=-1)
        write_logits = self.write_score(candidate)
        write_probability = torch.sigmoid(write_logits)
        hard_write = (write_probability >= 0.5) & row_valid
        straight_through_write = (
            hard_write.to(dtype=write_probability.dtype)
            + write_probability
            - write_probability.detach()
        )
        next_memory = memory + straight_through_write.unsqueeze(-1) * (
            shifted_memory - memory
        )
        next_valid = torch.where(
            hard_write,
            shifted_valid,
            memory_valid,
        )
        return next_memory, next_valid, hard_write, write_logits
