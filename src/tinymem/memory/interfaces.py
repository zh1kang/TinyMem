"""Common interface for bounded TinyMem memory policies."""

from abc import ABC, abstractmethod

import torch

from tinymem.memory.state import MemoryState


class MemoryPolicy(ABC):
    """Define how expired candidates update a fixed-capacity memory state."""

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("capacity must be an integer")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity

    def initialize(
        self,
        *,
        batch_size: int,
        model_width: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> MemoryState:
        """Create an empty state with this policy's fixed capacity."""
        return MemoryState.empty(
            batch_size=batch_size,
            capacity=self.capacity,
            model_width=model_width,
            device=device,
            dtype=dtype,
        )

    def read(self, state: MemoryState) -> tuple[torch.Tensor, torch.Tensor]:
        """Return memory values and the mask identifying occupied slots."""
        self._validate_state(state)
        return state.values, state.valid

    def _validate_state(self, state: MemoryState) -> None:
        if not isinstance(state, MemoryState):
            raise TypeError("state must be a MemoryState")
        if state.capacity != self.capacity:
            raise ValueError(
                f"state capacity must be {self.capacity}, got {state.capacity}"
            )

    def _validate_update(
        self,
        state: MemoryState,
        candidates: MemoryState,
    ) -> None:
        self._validate_state(state)
        if not isinstance(candidates, MemoryState):
            raise TypeError("candidates must be a MemoryState")
        if candidates.batch_size != state.batch_size:
            raise ValueError("candidates and state must have the same batch size")
        if candidates.model_width != state.model_width:
            raise ValueError("candidates and state must have the same model width")
        if candidates.values.dtype != state.values.dtype:
            raise ValueError("candidates and state must have the same value dtype")
        if candidates.values.device != state.values.device:
            raise ValueError("candidates and state must be on the same device")

    @abstractmethod
    def update(
        self,
        state: MemoryState,
        candidates: MemoryState,
        *,
        generator: torch.Generator | None = None,
    ) -> MemoryState:
        """Select candidates and return the next fixed-capacity state."""
        raise NotImplementedError
