"""Lossless latest-sentence storage for the declared qa1 movement grammar."""

import json
import re

import torch

from tinymem.data.opaque_qa1 import ROOMS, SHORT_NAMES
from tinymem.data.symbolic_world import MOVEMENT_SEPARATORS
from tinymem.memory.packed_tokens import PackedTokenState


class TemplateFactRetention:
    """Keep actual sentences using shared grammar codes and exact 80-bit names.

    This is a domain-specific lossless raw baseline, not semantic compression.
    The only per-stream object is a fixed byte payload; decoded text is temporary.
    """

    def __init__(self, byte_budget: int) -> None:
        if isinstance(byte_budget, bool) or not isinstance(byte_budget, int) or byte_budget <= 0:
            raise ValueError("byte_budget must be a positive integer")
        self.payload_bytes = byte_budget
        self.capacity = byte_budget * 8 // 11
        while self.capacity > 0 and 11 * self.capacity + self.capacity.bit_length() > byte_budget * 8:
            self.capacity -= 1
        if not self.capacity:
            raise ValueError("byte_budget cannot store one short-name sentence")
        self.length_bits = self.capacity.bit_length()

    @property
    def dictionary_serialized_bytes(self) -> int:
        grammar = {"names": SHORT_NAMES, "rooms": ROOMS, "movements": MOVEMENT_SEPARATORS,
                   "opaque_name": "person[0-9a-f]{20}", "terminal": "."}
        return len(json.dumps(grammar, sort_keys=True, separators=(",", ":")).encode())

    def empty(self, *, device: torch.device | str = "cpu") -> PackedTokenState:
        return PackedTokenState(torch.zeros(1, self.payload_bytes, dtype=torch.uint8, device=device))

    def _fields(self, sentence: str) -> tuple[str, int, int]:
        if not isinstance(sentence, str):
            raise TypeError("sentence must be a string")
        for verb, separator in enumerate(MOVEMENT_SEPARATORS):
            if sentence.count(separator) != 1:
                continue
            person, suffix = sentence.split(separator)
            if person not in SHORT_NAMES and re.fullmatch(r"person[0-9a-f]{20}", person) is None:
                continue
            for room, name in enumerate(ROOMS):
                if suffix == name + ".":
                    return person, verb, room
        raise ValueError("sentence is outside the declared exact movement grammar")

    def _bits(self, sentence: str) -> int:
        return 11 if self._fields(sentence)[0] in SHORT_NAMES else 87

    def _pack(self, sentences: list[str], device: torch.device) -> PackedTokenState:
        word, offset = len(sentences), self.length_bits
        for sentence in sentences:
            person, verb, room = self._fields(sentence)
            opaque = person not in SHORT_NAMES
            word |= int(opaque) << offset
            offset += 1
            word |= (int(person[6:], 16) if opaque else SHORT_NAMES.index(person)) << offset
            offset += 80 if opaque else 4
            word |= verb << offset
            word |= room << (offset + 3)
            offset += 6
        if len(sentences) > self.capacity or offset > self.payload_bytes * 8:
            raise ValueError("sentences exceed the byte budget")
        return PackedTokenState(torch.tensor([list(word.to_bytes(self.payload_bytes, "little"))], dtype=torch.uint8, device=device))

    def sentences(self, state: PackedTokenState) -> list[str]:
        if not isinstance(state, PackedTokenState):
            raise TypeError("state must be a PackedTokenState")
        if state.payload.shape != (1, self.payload_bytes):
            raise ValueError("state must match the single-stream byte format")
        word = int.from_bytes(bytes(state.payload[0].tolist()), "little")
        count = word & ((1 << self.length_bits) - 1)
        if count > self.capacity:
            raise ValueError("sentence count exceeds capacity")
        offset, result = self.length_bits, []

        def read(width: int) -> int:
            nonlocal offset
            if offset + width > self.payload_bytes * 8:
                raise ValueError("truncated movement sentence")
            value = (word >> offset) & ((1 << width) - 1)
            offset += width
            return value

        for _ in range(count):
            opaque = read(1)
            identity = read(80 if opaque else 4)
            if not opaque and identity >= len(SHORT_NAMES):
                raise ValueError("invalid short-name code")
            person = f"person{identity:020x}" if opaque else SHORT_NAMES[identity]
            verb, room = read(3), read(3)
            if verb >= len(MOVEMENT_SEPARATORS) or room >= len(ROOMS):
                raise ValueError("invalid movement or room code")
            result.append(person + MOVEMENT_SEPARATORS[verb] + ROOMS[room] + ".")
        if word >> offset:
            raise ValueError("unused payload bits must be zero")
        people = [self._fields(sentence)[0] for sentence in result]
        if len(set(people)) != len(people):
            raise ValueError("retained people must be distinct")
        return result

    def append_sentence(self, state: PackedTokenState, sentence: str) -> PackedTokenState:
        person = self._fields(sentence)[0]
        retained = [old for old in self.sentences(state) if self._fields(old)[0] != person]
        if self.length_bits + self._bits(sentence) <= self.payload_bytes * 8:
            retained.append(sentence)
        while len(retained) > self.capacity or self.length_bits + sum(self._bits(old) for old in retained) > self.payload_bytes * 8:
            retained.pop(0)
        return self._pack(retained, state.payload.device)

    def text(self, state: PackedTokenState) -> str:
        sentences = self.sentences(state)
        return "\n".join(sentences) + "\n\n" if sentences else ""
