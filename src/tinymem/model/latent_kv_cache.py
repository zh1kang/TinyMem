"""Compressed latent cache for MLA-lite autoregressive decoding."""

from dataclasses import dataclass

import torch


@dataclass
class LatentKVCache:
    """Store one shared key-value latent per token.

    Latent shape: [batch, sequence, latent_dim]
    """

    max_length: int
    latents: torch.Tensor | None = None
    start_position: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.max_length, bool) or not isinstance(self.max_length, int):
            raise TypeError("max_length must be an integer")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if isinstance(self.start_position, bool) or not isinstance(
            self.start_position, int
        ):
            raise TypeError("start_position must be an integer")
        if self.start_position < 0:
            raise ValueError("start_position must be nonnegative")
        if self.latents is not None:
            self._validate_latents(self.latents)
            if self.latents.shape[1] == 0:
                raise ValueError("cached latents must contain at least one token")
            if self.latents.shape[1] > self.max_length:
                raise ValueError("cached latents exceed max_length")

    @staticmethod
    def _validate_latents(latents: torch.Tensor) -> None:
        if not isinstance(latents, torch.Tensor):
            raise TypeError("latents must be a torch.Tensor")
        if latents.ndim != 3:
            raise ValueError("latents must be a rank-three tensor")
        if not latents.is_floating_point():
            raise TypeError("latents must be a floating-point tensor")

    @property
    def sequence_length(self) -> int:
        """Return the number of cached tokens."""
        if self.latents is None:
            return 0
        return self.latents.shape[1]

    @property
    def latent_dim(self) -> int | None:
        """Return the compressed width, or None for an empty cache."""
        if self.latents is None:
            return None
        return self.latents.shape[2]

    @property
    def end_position(self) -> int:
        """Return the exclusive absolute position after the cache."""
        return self.start_position + self.sequence_length

    @property
    def nbytes(self) -> int:
        """Return the storage used by the cached latent tensor in bytes."""
        if self.latents is None:
            return 0
        return self.latents.numel() * self.latents.element_size()

    def append(self, latents: torch.Tensor) -> None:
        """Append latent states and retain the newest tokens."""
        self._validate_latents(latents)
        if latents.shape[1] == 0:
            raise ValueError("appended latents must contain at least one token")

        if self.latents is None:
            combined = latents
        else:
            if latents.shape[0] != self.latents.shape[0]:
                raise ValueError("new latents must match the cache batch size")
            if latents.shape[2] != self.latents.shape[2]:
                raise ValueError("new latents must match the cache latent dimension")
            if latents.dtype != self.latents.dtype:
                raise ValueError("new latents must match the cache dtype")
            if latents.device != self.latents.device:
                raise ValueError("new latents must match the cache device")
            combined = torch.cat((self.latents, latents), dim=1)

        excess_length = combined.shape[1] - self.max_length
        if excess_length > 0:
            combined = combined[:, excess_length:, :]
            self.start_position += excess_length

        self.latents = combined

    def get(self) -> torch.Tensor:
        """Return the cached latent states."""
        if self.latents is None:
            raise ValueError("cache is empty")
        return self.latents

    def clear(self) -> None:
        """Remove all cached states and reset the absolute position."""
        self.latents = None
        self.start_position = 0
