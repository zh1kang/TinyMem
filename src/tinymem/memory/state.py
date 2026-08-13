"""Shared bounded state representation for TinyMem memory methods."""

from dataclasses import dataclass

import torch


INTEGER_DTYPES = (torch.int32, torch.int64)


@dataclass
class MemoryState:
    """Store fixed-capacity memory slots and their provenance metadata.

    values:    [batch, slots, model_width]
    valid:     [batch, slots]
    positions: [batch, slots]
    token_ids: [batch, slots] or None for non-token memory
    scores:    [batch, slots] or None for unscored memory
    """

    values: torch.Tensor
    valid: torch.Tensor
    positions: torch.Tensor
    token_ids: torch.Tensor | None = None
    scores: torch.Tensor | None = None

    def __post_init__(self) -> None:
        tensors = {
            "values": self.values,
            "valid": self.valid,
            "positions": self.positions,
        }
        if self.token_ids is not None:
            tensors["token_ids"] = self.token_ids
        if self.scores is not None:
            tensors["scores"] = self.scores
        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")

        if self.values.ndim != 3:
            raise ValueError("values must have shape [batch, slots, model_width]")
        if not self.values.is_floating_point():
            raise TypeError("values must be a floating-point tensor")

        expected_shape = self.values.shape[:2]
        if self.valid.ndim != 2 or self.valid.shape != expected_shape:
            raise ValueError(f"valid must have shape {expected_shape}")
        if self.valid.dtype != torch.bool:
            raise TypeError("valid must be a boolean tensor")

        if self.positions.ndim != 2 or self.positions.shape != expected_shape:
            raise ValueError(f"positions must have shape {expected_shape}")
        if self.positions.dtype not in INTEGER_DTYPES:
            raise TypeError("positions must be an integer tensor")

        if self.token_ids is not None:
            if self.token_ids.ndim != 2 or self.token_ids.shape != expected_shape:
                raise ValueError(f"token_ids must have shape {expected_shape}")
            if self.token_ids.dtype not in INTEGER_DTYPES:
                raise TypeError("token_ids must be an integer tensor")

        if self.scores is not None:
            if self.scores.ndim != 2 or self.scores.shape != expected_shape:
                raise ValueError(f"scores must have shape {expected_shape}")
            if not self.scores.is_floating_point():
                raise TypeError("scores must be a floating-point tensor")

        if any(tensor.device != self.values.device for tensor in tensors.values()):
            raise ValueError("all state tensors must be on the same device")

    @classmethod
    def empty(
        cls,
        *,
        batch_size: int,
        capacity: int,
        model_width: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        with_scores: bool = False,
    ) -> "MemoryState":
        """Create an empty fixed-capacity state."""
        dimensions = {
            "batch_size": batch_size,
            "capacity": capacity,
            "model_width": model_width,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not isinstance(with_scores, bool):
            raise TypeError("with_scores must be a boolean")

        values = torch.zeros(
            (batch_size, capacity, model_width),
            device=device,
            dtype=dtype,
        )
        valid = torch.zeros(
            (batch_size, capacity),
            device=device,
            dtype=torch.bool,
        )
        positions = torch.full(
            (batch_size, capacity),
            -1,
            device=device,
            dtype=torch.int64,
        )
        token_ids = torch.full(
            (batch_size, capacity),
            -1,
            device=device,
            dtype=torch.int64,
        )
        scores = None
        if with_scores:
            scores = torch.full(
                (batch_size, capacity),
                float("-inf"),
                device=device,
                dtype=torch.float32,
            )
        return cls(
            values=values,
            valid=valid,
            positions=positions,
            token_ids=token_ids,
            scores=scores,
        )

    @property
    def batch_size(self) -> int:
        """Return the state batch size."""
        return self.values.shape[0]

    @property
    def capacity(self) -> int:
        """Return the number of memory slots."""
        return self.values.shape[1]

    @property
    def model_width(self) -> int:
        """Return the width of each memory value."""
        return self.values.shape[2]

    @property
    def occupied_count(self) -> torch.Tensor:
        """Return occupied-slot counts for each batch item."""
        return self.valid.sum(dim=1)

    @property
    def nbytes(self) -> int:
        """Return the storage used by all state tensors in bytes."""
        tensors = [self.values, self.valid, self.positions]
        if self.token_ids is not None:
            tensors.append(self.token_ids)
        if self.scores is not None:
            tensors.append(self.scores)
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def clear(self) -> None:
        """Mark all slots empty without changing the allocated shapes."""
        self.values.zero_()
        self.valid.fill_(False)
        self.positions.fill_(-1)
        if self.token_ids is not None:
            self.token_ids.fill_(-1)
        if self.scores is not None:
            self.scores.fill_(float("-inf"))
