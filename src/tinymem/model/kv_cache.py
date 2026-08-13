"""Key-value cache for cached autoregressive decoding."""

from dataclasses import dataclass

import torch


@dataclass
class KVCache:
    """Store projected keys and values for one attention layer."""

    max_length: int
    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None
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

        if (self.keys is None) != (self.values is None):
            raise ValueError("keys and values must both be set or both be None")
        if self.keys is not None and self.values is not None:
            self._validate_tensor_pair(self.keys, self.values)
            if self.keys.shape[-2] == 0:
                raise ValueError("cached tensors must contain at least one token")
            if self.keys.shape[-2] > self.max_length:
                raise ValueError("cached tensors exceed max_length")

    @staticmethod
    def _validate_tensor_pair(keys: torch.Tensor, values: torch.Tensor) -> None:
        if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
            raise TypeError("keys and values must be torch.Tensor objects")
        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError("keys and values must be rank-four tensors")
        if not keys.is_floating_point() or not values.is_floating_point():
            raise TypeError("keys and values must be floating-point tensors")
        if keys.shape != values.shape:
            raise ValueError("keys and values must have the same shape")
        if keys.dtype != values.dtype:
            raise ValueError("keys and values must have the same dtype")
        if keys.device != values.device:
            raise ValueError("keys and values must be on the same device")

    @property
    def sequence_length(self) -> int:
        """Return the number of cached tokens."""
        if self.keys is None:
            return 0
        assert self.values is not None
        return self.keys.shape[-2]

    @property
    def end_position(self) -> int:
        """Return the exclusive absolute position after the cache."""
        return self.start_position + self.sequence_length

    @property
    def nbytes(self) -> int:
        """Return the storage used by cached keys and values in bytes."""
        if self.keys is None or self.values is None:
            return 0
        return (
            self.keys.numel() * self.keys.element_size()
            + self.values.numel() * self.values.element_size()
        )

    def append(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Append projected key/value states and retain the newest tokens."""
        self._validate_tensor_pair(keys, values)
        if keys.shape[-2] == 0:
            raise ValueError("appended tensors must contain at least one token")

        if (self.keys is None) != (self.values is None):
            raise RuntimeError("cache keys and values must be set together")
        if self.keys is None:
            combined_keys = keys
            combined_values = values
        else:
            assert self.values is not None
            if (
                keys.shape[:2] != self.keys.shape[:2]
                or keys.shape[-1] != self.keys.shape[-1]
            ):
                raise ValueError(
                    "new states must match the cache batch, heads, and head dimension"
                )
            if keys.dtype != self.keys.dtype or keys.device != self.keys.device:
                raise ValueError("new states must match the cache dtype and device")
            combined_keys = torch.cat((self.keys, keys), dim=-2)
            combined_values = torch.cat((self.values, values), dim=-2)

        excess_length = combined_keys.shape[-2] - self.max_length
        if excess_length > 0:
            combined_keys = combined_keys[..., excess_length:, :]
            combined_values = combined_values[..., excess_length:, :]
            self.start_position += excess_length

        self.keys = combined_keys
        self.values = combined_values

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached keys and values for attention."""
        if self.keys is None or self.values is None:
            raise ValueError("cache is empty")
        return self.keys, self.values

    def clear(self) -> None:
        """Remove all cached states and reset the absolute position."""
        self.keys = None
        self.values = None
        self.start_position = 0
