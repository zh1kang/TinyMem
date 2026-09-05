"""Lossless token retention with a shared vocabulary and native-ID escapes."""

from collections.abc import Sequence

import torch

from tinymem.memory.packed_tokens import PackedTokenState


class VocabularyTokenRetention:
    """Store recent native IDs in a fixed byte budget, never splitting an escape.

    Vocabulary IDs are shared, frozen training artifacts, not per-stream state.
    Known IDs use short codes; unseen IDs use an escape followed by the native ID.
    """

    def __init__(self, byte_budget: int, vocab_size: int, vocabulary: Sequence[int]) -> None:
        for name, value in (("byte_budget", byte_budget), ("vocab_size", vocab_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if vocab_size > 2**31:
            raise ValueError("vocab_size must fit positive int32 token IDs")
        if any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < vocab_size for value in vocabulary):
            raise ValueError("vocabulary must contain valid native integer IDs")
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("vocabulary IDs must be unique")
        self.payload_bytes = byte_budget
        self.vocab_size = vocab_size
        self.vocabulary = tuple(sorted(vocabulary))
        self.code_by_id = {value: index for index, value in enumerate(self.vocabulary)}
        self.escape = len(self.vocabulary)
        self.code_bits = max(1, self.escape.bit_length())
        self.native_bits = max(1, (vocab_size - 1).bit_length())
        minimum_bits = self.code_bits if self.vocabulary else self.code_bits + self.native_bits
        self.capacity = byte_budget * 8 // minimum_bits
        while self.capacity > 0 and self.capacity * minimum_bits + self.capacity.bit_length() > byte_budget * 8:
            self.capacity -= 1
        if self.capacity == 0:
            raise ValueError("byte_budget cannot hold one encoded token and its length")
        self.length_bits = self.capacity.bit_length()

    @property
    def dictionary_serialized_bytes(self) -> int:
        """Shared uint32 ID table size, not Python lookup-table runtime memory."""
        return len(self.vocabulary) * 4

    def empty(self, batch_size: int, *, device: torch.device | str) -> PackedTokenState:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        return PackedTokenState(torch.zeros(batch_size, self.payload_bytes, dtype=torch.uint8, device=device))

    def _token_bits(self, token_id: int) -> int:
        return self.code_bits + (0 if token_id in self.code_by_id else self.native_bits)

    def fits(self, token_ids: Sequence[int]) -> bool:
        if any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < self.vocab_size for value in token_ids):
            raise ValueError("token IDs must be native integers in the vocabulary")
        return len(token_ids) <= self.capacity and self.length_bits + sum(self._token_bits(value) for value in token_ids) <= self.payload_bytes * 8

    def append(self, state: PackedTokenState, token_ids: torch.Tensor, valid: torch.Tensor) -> PackedTokenState:
        previous = self._unpack(state)
        if token_ids.ndim != 2 or token_ids.shape[0] != len(previous):
            raise ValueError("token_ids must match the state batch")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("token_ids must use int32 or int64")
        if valid.shape != token_ids.shape or valid.dtype != torch.bool:
            raise ValueError("valid must be boolean and match token_ids")
        if token_ids.device != state.payload.device or valid.device != token_ids.device:
            raise ValueError("state and inputs must share a device")
        if ((token_ids < 0) | (token_ids >= self.vocab_size)).any():
            raise ValueError("token IDs must be in the native vocabulary")
        payloads = []
        for row, old in enumerate(previous):
            values = old + token_ids[row, valid[row]].tolist()
            bits = self.length_bits + sum(self._token_bits(value) for value in values)
            start = 0
            while bits > self.payload_bytes * 8 or len(values) - start > self.capacity:
                bits -= self._token_bits(values[start])
                start += 1
            values = values[start:]
            word, offset = len(values), self.length_bits
            for value in values:
                code = self.code_by_id.get(value, self.escape)
                word |= code << offset
                offset += self.code_bits
                if code == self.escape:
                    word |= value << offset
                    offset += self.native_bits
            payloads.append(list(word.to_bytes(self.payload_bytes, "little")))
        return PackedTokenState(torch.tensor(payloads, dtype=torch.uint8, device=state.payload.device))

    def _unpack(self, state: PackedTokenState) -> list[list[int]]:
        if not isinstance(state, PackedTokenState):
            raise TypeError("state must be a PackedTokenState")
        if state.payload.shape[1] != self.payload_bytes:
            raise ValueError("state must match the shared byte format")
        result = []
        for payload in state.payload.tolist():
            word = int.from_bytes(bytes(payload), "little")
            count = word & ((1 << self.length_bits) - 1)
            if count > self.capacity:
                raise ValueError("packed length exceeds capacity")
            offset, values = self.length_bits, []
            for _ in range(count):
                if offset + self.code_bits > self.payload_bytes * 8:
                    raise ValueError("truncated token code")
                code = (word >> offset) & ((1 << self.code_bits) - 1)
                offset += self.code_bits
                if code < self.escape:
                    value = self.vocabulary[code]
                elif code == self.escape:
                    if offset + self.native_bits > self.payload_bytes * 8:
                        raise ValueError("truncated native-ID escape")
                    value = (word >> offset) & ((1 << self.native_bits) - 1)
                    offset += self.native_bits
                    if value >= self.vocab_size or value in self.code_by_id:
                        raise ValueError("escape must identify an unseen valid native ID")
                else:
                    raise ValueError("unknown vocabulary code")
                values.append(value)
            if word >> offset:
                raise ValueError("unused payload bits must be zero")
            result.append(values)
        return result

    def materialize(self, state: PackedTokenState, *, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return temporary native IDs; the stored codes never enter the reader."""
        if isinstance(pad_id, bool) or not isinstance(pad_id, int) or not 0 <= pad_id < self.vocab_size:
            raise ValueError("pad_id must be a valid native integer ID")
        rows = self._unpack(state)
        ids = torch.full((len(rows), self.capacity), pad_id, dtype=torch.long, device=state.payload.device)
        valid = torch.zeros_like(ids, dtype=torch.bool)
        for row, values in enumerate(rows):
            ids[row, :len(values)] = torch.tensor(values, dtype=torch.long, device=ids.device)
            valid[row, :len(values)] = True
        return ids, valid
