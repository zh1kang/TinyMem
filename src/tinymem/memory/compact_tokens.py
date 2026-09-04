"""Contiguous raw-ID retention without stored token embeddings or KV."""

from dataclasses import dataclass

import torch

from tinymem.memory.storage import tensor_storage_bytes


def token_storage_dtype(vocab_size: int) -> torch.dtype:
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
        raise TypeError("vocab_size must be an integer")
    if not 1 <= vocab_size <= 2**31:
        raise ValueError("vocab_size must fit positive int32 token IDs")
    if vocab_size <= 256:
        return torch.uint8
    return torch.int16 if vocab_size <= 2**15 else torch.int32


@dataclass(frozen=True)
class CompactTokenState:
    """Keep left-aligned retained IDs and per-row lengths in fixed allocations."""

    token_ids: torch.Tensor
    lengths: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.token_ids, torch.Tensor) or not isinstance(self.lengths, torch.Tensor):
            raise TypeError("token_ids and lengths must be tensors")
        if self.token_ids.ndim != 2 or min(self.token_ids.shape) <= 0:
            raise ValueError("token_ids must have nonempty [batch, capacity] shape")
        if self.token_ids.dtype not in (torch.uint8, torch.int16, torch.int32):
            raise TypeError("token_ids must use a compact integer dtype")
        if self.lengths.shape != self.token_ids.shape[:1] or self.lengths.dtype != torch.int64:
            raise ValueError("lengths must be int64 with shape [batch]")
        if self.lengths.device != self.token_ids.device:
            raise ValueError("state tensors must share a device")
        if ((self.lengths < 0) | (self.lengths > self.token_ids.shape[1])).any():
            raise ValueError("lengths must be within capacity")

    @property
    def nbytes(self) -> int:
        return tensor_storage_bytes((self.token_ids, self.lengths))


class CompactTokenRetention:
    """Retain only recent IDs; reconstruct reader input when a query arrives."""

    def __init__(self, capacity: int, vocab_size: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("capacity must be an integer")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.vocab_size = vocab_size
        self.dtype = token_storage_dtype(vocab_size)

    def empty(self, batch_size: int, *, device: torch.device | str) -> CompactTokenState:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        return CompactTokenState(
            torch.zeros(batch_size, self.capacity, dtype=self.dtype, device=device),
            torch.zeros(batch_size, dtype=torch.int64, device=device),
        )

    def append(
        self, state: CompactTokenState, token_ids: torch.Tensor, valid: torch.Tensor,
    ) -> CompactTokenState:
        self._validate_state(state)
        if token_ids.ndim != 2 or token_ids.shape[0] != state.token_ids.shape[0]:
            raise ValueError("token_ids must match the state batch")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input token_ids must be int32 or int64")
        if valid.shape != token_ids.shape or valid.dtype != torch.bool:
            raise ValueError("valid must be boolean and match token_ids")
        if token_ids.device != state.token_ids.device or valid.device != token_ids.device:
            raise ValueError("inputs and state must share a device")
        if ((token_ids < 0) | (token_ids >= self.vocab_size)).any():
            raise ValueError("token IDs must be in the vocabulary")
        result = self.empty(token_ids.shape[0], device=token_ids.device)
        for row in range(token_ids.shape[0]):
            previous = state.token_ids[row, :int(state.lengths[row])].to(torch.int64)
            current = token_ids[row, valid[row]].to(torch.int64)
            retained = torch.cat((previous, current))[-self.capacity:]
            result.token_ids[row, :len(retained)] = retained.to(self.dtype)
            result.lengths[row] = len(retained)
        return result

    def _validate_state(self, state: CompactTokenState) -> None:
        if not isinstance(state, CompactTokenState):
            raise TypeError("state must be a CompactTokenState")
        if state.token_ids.shape[1] != self.capacity or state.token_ids.dtype != self.dtype:
            raise ValueError("state must match retention capacity and dtype")
        indices = torch.arange(self.capacity, device=state.token_ids.device)
        retained = indices.unsqueeze(0) < state.lengths.unsqueeze(1)
        ids = state.token_ids.to(torch.int64)
        if (retained & ((ids < 0) | (ids >= self.vocab_size))).any():
            raise ValueError("retained token IDs must be in the vocabulary")

    def materialize(self, state: CompactTokenState, *, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return temporary reader IDs and mask, without adding retained state."""
        self._validate_state(state)
        if not 0 <= pad_id < self.vocab_size:
            raise ValueError("pad_id must be in the vocabulary")
        indices = torch.arange(self.capacity, device=state.token_ids.device)
        valid = indices.unsqueeze(0) < state.lengths.unsqueeze(1)
        return state.token_ids.to(torch.int64).masked_fill(~valid, pad_id), valid
