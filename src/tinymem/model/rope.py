"""Rotary positional embeddings for the TinyMem decoder-only Transformer."""

from numbers import Real

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Rotate paired query and key features according to token position.

    Input and output shape: [batch, heads, sequence, head_dim]
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        base: float = 10_000.0,
    ) -> None:
        super().__init__()

        if isinstance(head_dim, bool) or not isinstance(head_dim, int):
            raise TypeError(f"head_dim must be an int, got {type(head_dim)}")
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError(f"head_dim must be a positive even integer, got {head_dim}")

        if (
            isinstance(max_position_embeddings, bool)
            or not isinstance(max_position_embeddings, int)
        ):
            raise TypeError(
                "max_position_embeddings must be an int, "
                f"got {type(max_position_embeddings)}"
            )
        if max_position_embeddings <= 0:
            raise ValueError(
                "max_position_embeddings must be positive, "
                f"got {max_position_embeddings}"
            )

        if isinstance(base, bool) or not isinstance(base, Real):
            raise TypeError(f"base must be a real number, got {type(base)}")
        if base <= 0:
            raise ValueError(f"base must be positive, got {base}")

        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = base ** (
            -torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(
        self,
        x: torch.Tensor,
        *,
        position_offset: int = 0,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Rotate features using contiguous or explicitly supplied positions."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x)}")
        if not x.is_floating_point():
            raise TypeError(f"x must be a floating-point tensor, got {x.dtype}")
        if x.ndim != 4:
            raise ValueError("x must have shape [batch, heads, sequence, head_dim]")
        if x.shape[-1] != self.head_dim:
            raise ValueError(
                f"x.shape[-1] must equal head_dim ({self.head_dim}), "
                f"got {x.shape[-1]}"
            )
        if isinstance(position_offset, bool) or not isinstance(position_offset, int):
            raise TypeError(
                f"position_offset must be an int, got {type(position_offset)}"
            )
        if position_offset < 0:
            raise ValueError("position_offset must be nonnegative")

        sequence_length = x.shape[-2]
        if sequence_length > self.max_position_embeddings:
            raise ValueError("sequence exceeds the maximum position supported")
        if position_ids is not None:
            if not isinstance(position_ids, torch.Tensor):
                raise TypeError("position_ids must be a torch.Tensor or None")
            if position_offset != 0:
                raise ValueError(
                    "position_offset must be zero when position_ids are supplied"
                )
            if position_ids.dtype not in (torch.int32, torch.int64):
                raise TypeError("position_ids must be an integer tensor")
            if position_ids.device != x.device:
                raise ValueError("position_ids and x must be on the same device")
            if position_ids.ndim not in (1, 2):
                raise ValueError("position_ids must have shape [T] or [B, T]")
            expected_shape = (sequence_length,)
            if position_ids.ndim == 1 and position_ids.shape != expected_shape:
                raise ValueError(f"position_ids must have shape {expected_shape}")
            expected_batch_shape = (x.shape[0], sequence_length)
            if position_ids.ndim == 2 and position_ids.shape != expected_batch_shape:
                raise ValueError(
                    f"position_ids must have shape {expected_batch_shape}"
                )
            if (position_ids < 0).any():
                raise ValueError("position_ids must be nonnegative")
        else:
            position_ids = torch.arange(
                position_offset,
                position_offset + sequence_length,
                device=x.device,
            )

        if position_ids.ndim == 1:
            position_batches = position_ids.unsqueeze(0)
        else:
            position_batches = position_ids

        angles = position_batches[..., None] * self.inv_freq[None, None, :]
        sin = angles.sin().unsqueeze(1)
        cos = angles.cos().unsqueeze(1)

        even = x[..., 0::2]
        odd = x[..., 1::2]
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos

        rotated = torch.stack(
            (rotated_even, rotated_odd),
            dim=-1,
        ).flatten(start_dim=-2)
        return rotated.to(dtype=x.dtype)
