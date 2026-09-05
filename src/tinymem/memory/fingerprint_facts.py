"""A domain-specific approximate association reference, not raw retention."""

import hashlib

import torch

from tinymem.data.opaque_qa1 import ROOMS
from tinymem.data.symbolic_world import parse_qa1_movement
from tinymem.memory.packed_tokens import PackedTokenState


def entity_fingerprint(person: str) -> int:
    if not isinstance(person, str) or not person or person != person.strip() or len(person.splitlines()) != 1:
        raise ValueError("person must be a nonempty single-line name")
    return int.from_bytes(hashlib.sha256(person.encode()).digest()[:2], "little")


class FingerprintFactRetention:
    """Store 16-bit name fingerprints and 3-bit rooms, using latest-write wins.

    Distinct names can collide and overwrite one another. The stored state cannot
    detect that collision or distinguish it from an ordinary location update.
    Queries can also falsely match an absent name. Report these limits explicitly.
    """

    def __init__(self, byte_budget: int) -> None:
        if isinstance(byte_budget, bool) or not isinstance(byte_budget, int) or byte_budget <= 0:
            raise ValueError("byte_budget must be a positive integer")
        self.payload_bytes = byte_budget
        self.capacity = byte_budget * 8 // 19
        while self.capacity > 0 and self.capacity * 19 + self.capacity.bit_length() > byte_budget * 8:
            self.capacity -= 1
        if not self.capacity:
            raise ValueError("byte_budget cannot hold one association")
        self.length_bits = self.capacity.bit_length()

    def empty(self, *, device: torch.device | str = "cpu") -> PackedTokenState:
        return PackedTokenState(torch.zeros(1, self.payload_bytes, dtype=torch.uint8, device=device))

    def _unpack(self, state: PackedTokenState) -> list[tuple[int, int]]:
        if not isinstance(state, PackedTokenState):
            raise TypeError("state must be a PackedTokenState")
        if state.payload.shape != (1, self.payload_bytes):
            raise ValueError("state must match the single-stream byte format")
        word = int.from_bytes(bytes(state.payload[0].tolist()), "little")
        count = word & ((1 << self.length_bits) - 1)
        if count > self.capacity:
            raise ValueError("association count exceeds capacity")
        offset, entries = self.length_bits, []
        for _ in range(count):
            key, room = (word >> offset) & 65535, (word >> (offset + 16)) & 7
            if room >= len(ROOMS):
                raise ValueError("invalid room code")
            entries.append((key, room))
            offset += 19
        if word >> offset:
            raise ValueError("unused payload bits must be zero")
        if len({key for key, _ in entries}) != len(entries):
            raise ValueError("stored fingerprints must be distinct")
        return entries

    def append_sentence(self, state: PackedTokenState, sentence: str) -> PackedTokenState:
        person, location = parse_qa1_movement(sentence, 1)
        if location.value not in ROOMS:
            raise ValueError("location must be one of the six declared rooms")
        key = entity_fingerprint(person)
        entries = [(old_key, room) for old_key, room in self._unpack(state) if old_key != key]
        entries.append((key, ROOMS.index(location.value)))
        entries = entries[-self.capacity:]
        word, offset = len(entries), self.length_bits
        for key, room in entries:
            word |= (key | (room << 16)) << offset
            offset += 19
        payload = torch.tensor([list(word.to_bytes(self.payload_bytes, "little"))], dtype=torch.uint8, device=state.payload.device)
        return PackedTokenState(payload)

    def lookup(self, state: PackedTokenState, person: str) -> str:
        """Return a room directly; this handcrafted decoder bypasses Qwen."""
        key = entity_fingerprint(person)
        for stored, room in self._unpack(state):
            if stored == key:
                return ROOMS[room]
        return "unknown"
