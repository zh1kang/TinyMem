"""Lossless fixed-width native-ID retention in a bounded byte tensor."""

from dataclasses import dataclass

import torch

from tinymem.memory.storage import tensor_storage_bytes


@dataclass(frozen=True)
class PackedTokenState:
    """The owning policy supplies the shared capacity and bit-width format."""

    payload: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.payload, torch.Tensor):
            raise TypeError("payload must be a tensor")
        if self.payload.ndim != 2 or min(self.payload.shape) <= 0:
            raise ValueError("payload must have nonempty [batch, bytes] shape")
        if self.payload.dtype != torch.uint8:
            raise TypeError("payload must use uint8 storage")

    @property
    def nbytes(self) -> int:
        return tensor_storage_bytes((self.payload,))


class PackedTokenRetention:
    """Pack a length header then recent IDs, least-significant bit first."""

    def __init__(self, capacity: int, vocab_size: int) -> None:
        for name, value in (("capacity", capacity), ("vocab_size", vocab_size)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if vocab_size > 2**31:
            raise ValueError("vocab_size must fit positive int32 token IDs")
        self.capacity = capacity
        self.vocab_size = vocab_size
        self.bits_per_id = max(1, (vocab_size - 1).bit_length())
        self.length_bits = capacity.bit_length()
        self.payload_bytes = (self.length_bits + capacity * self.bits_per_id + 7) // 8

    def empty(self, batch_size: int, *, device: torch.device | str) -> PackedTokenState:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        return PackedTokenState(
            torch.zeros(batch_size, self.payload_bytes, dtype=torch.uint8, device=device),
        )

    def append(
        self, state: PackedTokenState, token_ids: torch.Tensor, valid: torch.Tensor,
    ) -> PackedTokenState:
        previous, previous_lengths = self._unpack(state)
        if token_ids.ndim != 2 or token_ids.shape[0] != state.payload.shape[0]:
            raise ValueError("token_ids must match the state batch")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input token_ids must be int32 or int64")
        if valid.shape != token_ids.shape or valid.dtype != torch.bool:
            raise ValueError("valid must be boolean and match token_ids")
        if token_ids.device != state.payload.device or valid.device != token_ids.device:
            raise ValueError("inputs and state must share a device")
        if ((token_ids < 0) | (token_ids >= self.vocab_size)).any():
            raise ValueError("token IDs must be in the vocabulary")
        retained = torch.zeros_like(previous)
        lengths = torch.empty_like(previous_lengths)
        for row in range(token_ids.shape[0]):
            old = previous[row, :int(previous_lengths[row])]
            current = token_ids[row, valid[row]].to(torch.int64)
            recent = torch.cat((old, current))[-self.capacity:]
            retained[row, :len(recent)] = recent
            lengths[row] = len(recent)
        return PackedTokenState(self._pack(retained, lengths))

    def _pack(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        shifts = torch.arange(self.bits_per_id, device=token_ids.device)
        bits = (token_ids.unsqueeze(-1) >> shifts) & 1
        padded = token_ids.new_zeros(token_ids.shape[0], self.payload_bytes * 8)
        length_shifts = torch.arange(self.length_bits, device=token_ids.device)
        padded[:, :self.length_bits] = (lengths.unsqueeze(-1) >> length_shifts) & 1
        padded[:, self.length_bits:self.length_bits + self.capacity * self.bits_per_id] = bits.flatten(1)
        byte_shifts = torch.arange(8, device=token_ids.device)
        return (padded.view(token_ids.shape[0], -1, 8) << byte_shifts).sum(-1).to(torch.uint8)

    def _unpack(self, state: PackedTokenState) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(state, PackedTokenState):
            raise TypeError("state must be a PackedTokenState")
        if state.payload.shape[1] != self.payload_bytes:
            raise ValueError("state must match the shared packed format")
        byte_shifts = torch.arange(8, device=state.payload.device)
        bits = ((state.payload.to(torch.int64).unsqueeze(-1) >> byte_shifts) & 1).flatten(1)
        length_shifts = torch.arange(self.length_bits, device=state.payload.device)
        lengths = (bits[:, :self.length_bits] << length_shifts).sum(-1)
        if (lengths > self.capacity).any():
            raise ValueError("packed length exceeds capacity")
        used_bits = self.length_bits + self.capacity * self.bits_per_id
        if bits[:, used_bits:].any():
            raise ValueError("unused payload bits must be zero")
        shifts = torch.arange(self.bits_per_id, device=state.payload.device)
        ids = (bits[:, self.length_bits:used_bits].reshape(-1, self.capacity, self.bits_per_id) << shifts).sum(-1)
        indices = torch.arange(self.capacity, device=state.payload.device)
        valid = indices.unsqueeze(0) < lengths.unsqueeze(1)
        if (valid & (ids >= self.vocab_size)).any():
            raise ValueError("retained token IDs must be in the vocabulary")
        if ((~valid) & (ids != 0)).any():
            raise ValueError("unused token positions must be zero")
        return ids, lengths

    def materialize(self, state: PackedTokenState, *, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode temporary int64 reader IDs; retain neither IDs nor embeddings."""
        if isinstance(pad_id, bool) or not isinstance(pad_id, int) or not 0 <= pad_id < self.vocab_size:
            raise ValueError("pad_id must be an integer in the vocabulary")
        ids, lengths = self._unpack(state)
        indices = torch.arange(self.capacity, device=state.payload.device)
        valid = indices.unsqueeze(0) < lengths.unsqueeze(1)
        return ids.masked_fill(~valid, pad_id), valid
