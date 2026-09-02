"""Learned discrete memory codebooks."""

from dataclasses import dataclass
import math
from numbers import Integral, Real

import torch
from torch import nn
from torch.nn import functional as F


CODEBOOK_EVALUATION_MODES = frozenset(("hard", "soft"))


@dataclass(frozen=True)
class CodebookOutput:
    """Return decoded memory vectors and their discrete assignment trace."""

    values: torch.Tensor
    indices: torch.Tensor
    assignments: torch.Tensor
    probabilities: torch.Tensor


class GumbelSoftmaxCodebook(nn.Module):
    """Select learned memory vectors with straight-through Gumbel-Softmax."""

    def __init__(
        self,
        model_width: int,
        codebook_size: int,
        *,
        evaluation_mode: str = "hard",
    ) -> None:
        super().__init__()
        for name, value in (
            ("model_width", model_width),
            ("codebook_size", codebook_size),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        self.model_width = int(model_width)
        self.codebook_size = int(codebook_size)
        self.embedding = nn.Embedding(self.codebook_size, self.model_width)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.set_evaluation_mode(evaluation_mode)

    def set_evaluation_mode(self, mode: str) -> None:
        """Select hard discrete codes or soft mixtures during evaluation."""
        if not isinstance(mode, str):
            raise TypeError("evaluation mode must be a string")
        if mode not in CODEBOOK_EVALUATION_MODES:
            raise ValueError("evaluation mode must be 'hard' or 'soft'")
        self.evaluation_mode = mode

    def _validate_inputs(
        self,
        logits: torch.Tensor,
        valid: torch.Tensor,
        temperature: float,
    ) -> float:
        if not isinstance(logits, torch.Tensor):
            raise TypeError("logits must be a torch.Tensor")
        if not isinstance(valid, torch.Tensor):
            raise TypeError("valid must be a torch.Tensor")
        if logits.ndim != 3:
            raise ValueError(
                "logits must have shape [batch, slots, codebook_size]"
            )
        if not logits.is_floating_point():
            raise TypeError("logits must be a floating-point tensor")
        if logits.shape[0] == 0 or logits.shape[1] == 0:
            raise ValueError("logits must contain batch rows and slots")
        if logits.shape[2] != self.codebook_size:
            raise ValueError(
                f"logits final dimension must be {self.codebook_size}"
            )
        if valid.shape != logits.shape[:2]:
            raise ValueError(f"valid must have shape {logits.shape[:2]}")
        if valid.dtype != torch.bool:
            raise TypeError("valid must be a boolean tensor")
        if valid.device != logits.device:
            raise ValueError("valid and logits must share a device")
        if logits.device != self.embedding.weight.device:
            raise ValueError("logits and codebook must share a device")
        if logits.dtype != self.embedding.weight.dtype:
            raise TypeError("logits and codebook must share a dtype")
        if isinstance(temperature, bool) or not isinstance(temperature, Real):
            raise TypeError("temperature must be a real number")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        return float(temperature)

    def forward(
        self,
        logits: torch.Tensor,
        valid: torch.Tensor,
        *,
        temperature: float,
    ) -> CodebookOutput:
        """Select one code per valid slot and decode it to model width."""
        temperature = self._validate_inputs(logits, valid, temperature)

        probabilities = torch.softmax(logits / temperature, dim=-1)
        if self.training:
            dtype_info = torch.finfo(logits.dtype)
            uniform = torch.rand_like(logits).clamp(
                min=dtype_info.tiny,
                max=1.0 - dtype_info.eps,
            )
            gumbel_noise = -torch.log(-torch.log(uniform))
            relaxed_assignments = torch.softmax(
                (logits + gumbel_noise) / temperature,
                dim=-1,
            )
        else:
            relaxed_assignments = probabilities

        indices = (
            relaxed_assignments.argmax(dim=-1)
            if self.training
            else logits.argmax(dim=-1)
        )
        hard_assignments = F.one_hot(
            indices,
            num_classes=self.codebook_size,
        ).to(dtype=logits.dtype)
        if self.training:
            assignments = (
                hard_assignments
                - relaxed_assignments.detach()
                + relaxed_assignments
            )
        elif self.evaluation_mode == "hard":
            assignments = hard_assignments
        else:
            assignments = probabilities

        expanded_valid = valid.unsqueeze(-1)
        probabilities = probabilities * expanded_valid
        assignments = assignments * expanded_valid
        values = assignments @ self.embedding.weight
        indices = indices.masked_fill(~valid, -1)
        return CodebookOutput(
            values=values,
            indices=indices,
            assignments=assignments,
            probabilities=probabilities,
        )
