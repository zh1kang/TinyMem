"""Learned continuous-memory compressors."""

from numbers import Integral

import torch
from torch import nn


class MeanPoolMemoryCompressor(nn.Module):
    """Compress a masked group of expired hidden states into one memory slot.

    Inputs:
        expired_hidden: [batch, expired_tokens, model_width]
        expired_valid:  [batch, expired_tokens]

    Outputs:
        summary:       [batch, 1, model_width]
        summary_valid: [batch, 1]
    """

    def __init__(self, model_width: int) -> None:
        super().__init__()
        if isinstance(model_width, bool) or not isinstance(model_width, Integral):
            raise TypeError("model_width must be an integer")
        if model_width <= 0:
            raise ValueError("model_width must be positive")

        self.model_width = int(model_width)
        self.projection = nn.Linear(self.model_width, self.model_width)

    def forward(
        self,
        expired_hidden: torch.Tensor,
        expired_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one learned masked-mean summary for each batch row."""
        if not isinstance(expired_hidden, torch.Tensor):
            raise TypeError("expired_hidden must be a torch.Tensor")
        if not isinstance(expired_valid, torch.Tensor):
            raise TypeError("expired_valid must be a torch.Tensor")
        if expired_hidden.ndim != 3:
            raise ValueError(
                "expired_hidden must have shape [batch, expired_tokens, model_width]"
            )
        if not expired_hidden.is_floating_point():
            raise TypeError("expired_hidden must be a floating-point tensor")
        if expired_hidden.shape[0] == 0:
            raise ValueError("expired_hidden must contain at least one batch row")
        if expired_hidden.shape[2] != self.model_width:
            raise ValueError(
                f"expired_hidden final dimension must be {self.model_width}"
            )
        expected_mask_shape = expired_hidden.shape[:2]
        if expired_valid.shape != expected_mask_shape:
            raise ValueError(
                f"expired_valid must have shape {expected_mask_shape}"
            )
        if expired_valid.dtype != torch.bool:
            raise TypeError("expired_valid must be a boolean tensor")
        if expired_valid.device != expired_hidden.device:
            raise ValueError("expired_valid and expired_hidden must share a device")
        if expired_hidden.device != self.projection.weight.device:
            raise ValueError("expired_hidden and compressor must share a device")
        if expired_hidden.dtype != self.projection.weight.dtype:
            raise TypeError("expired_hidden and compressor must share a dtype")

        expanded_valid = expired_valid.unsqueeze(-1).to(
            dtype=expired_hidden.dtype
        )
        masked_sum = (expired_hidden * expanded_valid).sum(
            dim=1,
            keepdim=True,
        )
        valid_count = expanded_valid.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = masked_sum / valid_count
        summary = self.projection(pooled)
        summary_valid = expired_valid.any(dim=1, keepdim=True)
        summary = summary * summary_valid.unsqueeze(-1).to(dtype=summary.dtype)
        return summary, summary_valid
