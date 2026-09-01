"""Learned continuous-memory compressors."""

from abc import ABC, abstractmethod
from numbers import Integral

import torch
from torch import nn


class ContinuousMemoryCompressor(nn.Module, ABC):
    """Define the shared one-slot continuous compression contract."""

    def __init__(self, model_width: int) -> None:
        super().__init__()
        if isinstance(model_width, bool) or not isinstance(model_width, Integral):
            raise TypeError("model_width must be an integer")
        if model_width <= 0:
            raise ValueError("model_width must be positive")
        self.model_width = int(model_width)

    def _validate_inputs(
        self,
        expired_hidden: torch.Tensor,
        expired_valid: torch.Tensor,
        *,
        parameter: torch.Tensor,
    ) -> torch.Tensor:
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
        if expired_hidden.shape[0] == 0 or expired_hidden.shape[1] == 0:
            raise ValueError("expired_hidden must contain batch rows and tokens")
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
        if expired_hidden.device != parameter.device:
            raise ValueError("expired_hidden and compressor must share a device")
        if expired_hidden.dtype != parameter.dtype:
            raise TypeError("expired_hidden and compressor must share a dtype")
        return expired_valid.any(dim=1, keepdim=True)

    @abstractmethod
    def forward(
        self,
        expired_hidden: torch.Tensor,
        expired_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one summary and one validity bit for each batch row."""


class MeanPoolMemoryCompressor(ContinuousMemoryCompressor):
    """Compress a masked group of expired hidden states into one memory slot.

    Inputs:
        expired_hidden: [batch, expired_tokens, model_width]
        expired_valid:  [batch, expired_tokens]

    Outputs:
        summary:       [batch, 1, model_width]
        summary_valid: [batch, 1]
    """

    def __init__(self, model_width: int) -> None:
        super().__init__(model_width)
        self.projection = nn.Linear(self.model_width, self.model_width)

    def forward(
        self,
        expired_hidden: torch.Tensor,
        expired_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one learned masked-mean summary for each batch row."""
        summary_valid = self._validate_inputs(
            expired_hidden,
            expired_valid,
            parameter=self.projection.weight,
        )

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
        summary = summary * summary_valid.unsqueeze(-1).to(dtype=summary.dtype)
        return summary, summary_valid


class AttentionPoolMemoryCompressor(ContinuousMemoryCompressor):
    """Use one learned query to pool valid hidden states into one slot."""

    def __init__(self, model_width: int) -> None:
        super().__init__(model_width)
        self.query = nn.Parameter(torch.zeros(self.model_width))
        self.projection = nn.Linear(self.model_width, self.model_width)
        self.scale = self.model_width**-0.5

    def forward(
        self,
        expired_hidden: torch.Tensor,
        expired_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a masked learned-attention summary for each batch row."""
        summary_valid = self._validate_inputs(
            expired_hidden,
            expired_valid,
            parameter=self.query,
        )
        scores = torch.einsum(
            "btd,d->bt",
            expired_hidden,
            self.query,
        ) * self.scale
        scores = scores.masked_fill(
            ~expired_valid,
            torch.finfo(scores.dtype).min,
        )
        weights = torch.softmax(scores, dim=1)
        weights = weights * expired_valid.to(dtype=weights.dtype)
        normalizer = weights.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).tiny
        )
        weights = weights / normalizer
        pooled = torch.einsum("bt,btd->bd", weights, expired_hidden).unsqueeze(1)
        summary = self.projection(pooled)
        summary = summary * summary_valid.unsqueeze(-1).to(dtype=summary.dtype)
        return summary, summary_valid
