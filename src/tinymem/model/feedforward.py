"""Position-wise feed-forward network for the TinyMem decoder."""

from numbers import Real

import torch
from torch import nn


class FeedForward(nn.Module):
    """Apply the same nonlinear feature transformation to every token.

    Input shape:  [B, T, D]
    Output shape: [B, T, D]
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()

        if isinstance(d_model, bool) or not isinstance(d_model, int):
            raise TypeError(f"d_model must be an int, got {type(d_model)}")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")

        if isinstance(d_ff, bool) or not isinstance(d_ff, int):
            raise TypeError(f"d_ff must be an int, got {type(d_ff)}")
        if d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {d_ff}")

        if isinstance(dropout, bool) or not isinstance(dropout, Real):
            raise TypeError(f"dropout must be a real number, got {type(dropout)}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}")

        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_probability = dropout

        self.input_proj = nn.Linear(d_model, d_ff)
        self.output_proj = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a nonlinear transformation with the final size unchanged."""

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x)}")
        if not x.is_floating_point():
            raise TypeError(f"x must be a floating-point tensor, got {x.dtype}")
        if x.ndim != 3:
            raise ValueError(f"x must be a 3D tensor, got {x.ndim}D")

        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"x last dimension must be {self.d_model}, got {x.shape[-1]}"
            )

        hidden = self.input_proj(x)
        hidden = torch.nn.functional.gelu(hidden)
        hidden = self.dropout(hidden)
        output = self.output_proj(hidden)
        return output
