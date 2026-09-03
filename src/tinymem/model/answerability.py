"""Predict whether the model has enough evidence to answer."""

from numbers import Integral, Real

import torch
from torch import nn

from tinymem.model.memory_input import AttentionMemory


class AnswerabilityHead(nn.Module):
    """Score local hidden states against the currently readable memory."""

    def __init__(
        self,
        model_width: int,
        *,
        hidden_width: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if isinstance(model_width, bool) or not isinstance(model_width, Integral):
            raise TypeError("model_width must be an integer")
        if model_width <= 0:
            raise ValueError("model_width must be positive")
        if hidden_width is None:
            hidden_width = model_width
        if isinstance(hidden_width, bool) or not isinstance(hidden_width, Integral):
            raise TypeError("hidden_width must be an integer or None")
        if hidden_width <= 0:
            raise ValueError("hidden_width must be positive")
        if isinstance(dropout, bool) or not isinstance(dropout, Real):
            raise TypeError("dropout must be a real number")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.model_width = int(model_width)
        self.hidden_width = int(hidden_width)
        self.projection = nn.Sequential(
            nn.Linear(2 * self.model_width, self.hidden_width),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_width, 1),
        )

    def forward(
        self,
        local_hidden: torch.Tensor,
        memory: AttentionMemory,
    ) -> torch.Tensor:
        """Return answerability logits with shape ``[batch, tokens]``."""
        if not isinstance(local_hidden, torch.Tensor):
            raise TypeError("local_hidden must be a torch.Tensor")
        if local_hidden.ndim != 3:
            raise ValueError(
                "local_hidden must have shape [batch, tokens, model_width]"
            )
        if not local_hidden.is_floating_point():
            raise TypeError("local_hidden must be a floating-point tensor")
        if local_hidden.shape[0] == 0 or local_hidden.shape[1] == 0:
            raise ValueError("local_hidden must contain a batch and tokens")
        if local_hidden.shape[2] != self.model_width:
            raise ValueError("local_hidden width must match model_width")
        if not isinstance(memory, AttentionMemory):
            raise TypeError("memory must be an AttentionMemory")
        if memory.values.shape[0] != local_hidden.shape[0]:
            raise ValueError("memory and local_hidden batch sizes must match")
        if memory.values.shape[2] != self.model_width:
            raise ValueError("memory width must match model_width")
        if memory.values.device != local_hidden.device:
            raise ValueError("memory and local_hidden must share a device")
        if memory.values.dtype != local_hidden.dtype:
            raise ValueError("memory and local_hidden must share a dtype")

        weights = memory.valid.unsqueeze(-1).to(dtype=memory.values.dtype)
        memory_summary = (memory.values * weights).sum(dim=1)
        memory_summary = memory_summary / weights.sum(dim=1).clamp_min(1)
        expanded_memory = memory_summary.unsqueeze(1).expand(
            -1,
            local_hidden.shape[1],
            -1,
        )
        features = torch.cat((local_hidden, expanded_memory), dim=-1)
        return self.projection(features).squeeze(-1)
