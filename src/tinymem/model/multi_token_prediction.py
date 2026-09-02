"""Independent vocabulary heads for future-token prediction."""

from numbers import Integral

import torch
from torch import nn


class MultiTokenPredictionHeads(nn.Module):
    """Project one hidden state toward several future token horizons."""

    def __init__(
        self,
        model_width: int,
        vocab_size: int,
        horizons: tuple[int, ...],
    ) -> None:
        super().__init__()
        for name, value in (
            ("model_width", model_width),
            ("vocab_size", vocab_size),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not isinstance(horizons, tuple):
            raise TypeError("horizons must be a tuple")
        if not horizons:
            raise ValueError("horizons must be nonempty")
        if any(
            isinstance(horizon, bool) or not isinstance(horizon, Integral)
            for horizon in horizons
        ):
            raise TypeError("horizons must contain integers")
        if any(horizon < 2 for horizon in horizons):
            raise ValueError("auxiliary horizons must be at least two")
        if tuple(sorted(set(horizons))) != horizons:
            raise ValueError("horizons must be unique and strictly increasing")

        self.model_width = int(model_width)
        self.vocab_size = int(vocab_size)
        self.horizons = tuple(int(horizon) for horizon in horizons)

        self.heads = nn.ModuleList(
            nn.Linear(self.model_width, self.vocab_size, bias=False)
            for _ in self.horizons
        )

    def forward(self, hidden_states: torch.Tensor) -> dict[int, torch.Tensor]:
        """Return one [B, T, V] logit tensor for each future horizon."""
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("hidden_states must be a torch.Tensor")
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, tokens, width]")
        if hidden_states.shape[0] == 0 or hidden_states.shape[1] == 0:
            raise ValueError("hidden_states must contain a batch and token")
        if hidden_states.shape[2] != self.model_width:
            raise ValueError(
                f"hidden_states must have width {self.model_width}"
            )
        if not hidden_states.is_floating_point():
            raise TypeError("hidden_states must be floating point")

        return {
            horizon: head(hidden_states)
            for horizon, head in zip(self.horizons, self.heads, strict=True)
        }
