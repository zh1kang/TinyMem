"""Normalization layers for the TinyMem decoder-only Transformer."""

from numbers import Real

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Normalize each token vector by its root mean square.

    Input shape:  [..., d_model]
    Output shape: [..., d_model]

    Implement this module from first principles. Do not call nn.RMSNorm.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()

        if isinstance(d_model, bool) or not isinstance(d_model, int):
            raise TypeError(f"d_model must be an int, got {type(d_model)}")
        if d_model <= 0:
            raise ValueError(f"d_model must be a positive integer, got {d_model}")

        if isinstance(eps, bool) or not isinstance(eps, Real):
            raise TypeError(f"eps must be a float, got {type(eps)}")
        if eps <= 0:
            raise ValueError(f"eps must be a positive real number, got {eps}")

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return an RMS-normalized tensor with the same shape as x."""

        if not x.is_floating_point():
            raise TypeError(f"Input tensor must be floating point, got {x.dtype}")

        if x.shape[-1] != self.weight.shape[0]:
            raise ValueError(
                f"Input tensor's final dimension must be {self.weight.shape[0]}, "
                f"got {x.shape[-1]}"
            )

        mean_square = torch.mean(x ** 2, dim=-1, keepdim=True)
        rms = torch.sqrt(mean_square + self.eps)
        normalized_x = x / rms
        output = normalized_x * self.weight
        return output.to(x.dtype)
