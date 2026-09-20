"""Query-blind text stores with an explicit per-history byte budget."""

from __future__ import annotations

import re
import struct
import zlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

SUPPORTED_CODECS: Final = (
    "compressed_recent",
    "compressed_diverse",
    "dictionary_recent",
)
MAX_RECORD_TOKENS: Final = 128
MAX_RECORD_BYTES: Final = 16_384
MAX_DECODE_BYTES: Final = 1_048_576
MAX_DICTIONARY_ENTRIES: Final = 65_535
_LITERAL: Final = 65_535
_ZLIB: Final = 1
_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class StoredText:
    """Optional typed wrapper for a raw per-history payload."""

    payload: bytes
    codec: str

    def __post_init__(self) -> None:
        if not isinstance(self.payload, bytes):
            raise TypeError("payload must be bytes")
        if self.codec not in SUPPORTED_CODECS:
            raise ValueError(f"unsupported codec: {self.codec!r}")

    @property
    def nbytes(self) -> int:
        return len(self.payload)


def _validate_record(record: str) -> str:
    if not isinstance(record, str):
        raise TypeError("record must be str")
    if not record:
        raise ValueError("record must not be empty")
    encoded = record.encode("utf-8")
    if len(encoded) > MAX_RECORD_BYTES:
        raise ValueError("record exceeds the byte limit")
    if len(_WORD.findall(record)) > MAX_RECORD_TOKENS:
        raise ValueError("record exceeds the token limit")
    return record


def build_statement_dictionary(records: Iterable[str]) -> tuple[str, ...]:
    """Build a deterministic codebook from a caller-selected train split."""

    if isinstance(records, (str, bytes)):
        raise TypeError("records must be an iterable of text records")
    values = {_validate_record(record) for record in records}
    if len(values) > MAX_DICTIONARY_ENTRIES:
        raise ValueError("shared dictionary is too large")
    return tuple(sorted(values, key=lambda value: value.encode("utf-8")))


def _decode_utf8(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("payload contains invalid UTF-8") from exc


def _valid_utf8_suffix(data: bytes) -> bytes:
    """Return the longest decodable suffix of ``data``."""

    for offset in range(min(4, len(data)) + 1):
        candidate = data[offset:]
        try:
            candidate.decode("utf-8")
        except UnicodeDecodeError:
            continue
        return candidate
    return b""


def _decompress(data: bytes) -> bytes:
    """Decompress one complete zlib stream under a hard output bound."""

    stream = zlib.decompressobj()
    try:
        result = stream.decompress(data, MAX_DECODE_BYTES + 1)
    except zlib.error as exc:
        raise ValueError("payload contains invalid compressed data") from exc
    if len(result) > MAX_DECODE_BYTES or not stream.eof or stream.unused_data or stream.unconsumed_tail:
        raise ValueError("compressed payload is incomplete, has a trailer, or exceeds the limit")
    return result


def _fit_zlib_suffix(text: str, budget: int) -> bytes:
    raw = text.encode("utf-8")
    best = b""
    # Compression size is not monotonic.  Check every suffix and retain the
    # largest valid UTF-8 body that fits the actual payload budget.
    for length in range(len(raw), -1, -1):
        candidate = _valid_utf8_suffix(raw[-length:]) if length else b""
        if len(candidate) <= len(best):
            continue
        payload = zlib.compress(candidate, level=9)
        if len(payload) <= budget:
            best = candidate
            if len(best) == len(raw):
                break
    return zlib.compress(best, level=9) if best or budget >= 8 else b""


def _varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint value must be nonnegative")
    result = bytearray()
    while value >= 0x80:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value)
    return bytes(result)


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, 35, 7):
        if offset >= len(data):
            raise ValueError("truncated varint")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
    raise ValueError("varint is too long")


def _record_body(records: tuple[str, ...]) -> bytes:
    return b"".join(
        _varint(len(encoded)) + encoded
        for record in records
        for encoded in (record.encode("utf-8"),)
    )


def _parse_record_body(body: bytes) -> tuple[str, ...]:
    records: list[str] = []
    offset = 0
    while offset < len(body):
        length, offset = _read_varint(body, offset)
        if length == 0:
            raise ValueError("record length must be positive")
        end = offset + length
        if end > len(body):
            raise ValueError("record length exceeds payload")
        records.append(_decode_utf8(body[offset:end]))
        offset = end
    return tuple(records)


def _log_payload(records: tuple[str, ...], budget: int) -> bytes:
    payload = zlib.compress(_record_body(records), level=9)
    if len(payload) > budget:
        raise ValueError("record log does not fit the storage budget")
    return payload


def _unpack_log(payload: bytes, budget: int) -> tuple[str, ...]:
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if not payload:
        return ()
    if len(payload) > budget:
        raise ValueError("payload exceeds the storage budget")
    return _parse_record_body(_decompress(payload))


def _novelty(record: str, covered: set[str]) -> int:
    words = {match.group(0).casefold() for match in _WORD.finditer(record)}
    return len(words - covered)


def _diverse_records(records: tuple[str, ...], budget: int) -> tuple[str, ...]:
    if not records:
        return ()
    selected = [len(records) - 1]
    covered = {match.group(0).casefold() for match in _WORD.finditer(records[-1])}
    remaining = list(range(len(records) - 1))
    while remaining:
        index = max(remaining, key=lambda item: (_novelty(records[item], covered), item))
        candidate_indices = sorted((*selected, index))
        try:
            _log_payload(tuple(records[item] for item in candidate_indices), budget)
        except ValueError:
            remaining.remove(index)
            continue
        selected.append(index)
        covered.update(match.group(0).casefold() for match in _WORD.finditer(records[index]))
        remaining.remove(index)
    return tuple(records[item] for item in sorted(selected))


def _fit_diverse_single(record: str, budget: int) -> bytes:
    raw = record.encode("utf-8")
    best = b""
    best_length = -1
    for length in range(len(raw), -1, -1):
        candidate = _valid_utf8_suffix(raw[-length:]) if length else b""
        if not candidate or len(candidate) <= best_length:
            continue
        try:
            payload = _log_payload((_decode_utf8(candidate),), budget)
        except ValueError:
            continue
        best, best_length = payload, len(candidate)
    return best


def _dictionary_body(records: tuple[str, ...], dictionary: tuple[str, ...]) -> bytes:
    lookup = {record: index for index, record in enumerate(dictionary)}
    chunks: list[bytes] = []
    for record in records:
        index = lookup.get(record)
        if index is not None:
            chunks.append(struct.pack(">H", index))
        else:
            encoded = record.encode("utf-8")
            chunks.append(struct.pack(">H", _LITERAL) + _varint(len(encoded)) + encoded)
    return b"".join(chunks)


def _parse_dictionary_body(body: bytes, dictionary: tuple[str, ...]) -> tuple[str, ...]:
    records: list[str] = []
    offset = 0
    while offset < len(body):
        if offset + 2 > len(body):
            raise ValueError("truncated dictionary ID")
        (index,) = struct.unpack(">H", body[offset:offset + 2])
        offset += 2
        if index != _LITERAL:
            if index >= len(dictionary):
                raise ValueError("dictionary ID is outside the shared codebook")
            records.append(dictionary[index])
            continue
        length, offset = _read_varint(body, offset)
        if length == 0:
            raise ValueError("literal length must be positive")
        end = offset + length
        if end > len(body):
            raise ValueError("literal length exceeds payload")
        records.append(_decode_utf8(body[offset:end]))
        offset = end
    return tuple(records)


def _dictionary_payload(records: tuple[str, ...], dictionary: tuple[str, ...], budget: int) -> bytes:
    raw = _dictionary_body(records, dictionary)
    compressed = zlib.compress(raw, level=9)
    if len(compressed) < len(raw):
        payload = bytes((_ZLIB,)) + compressed
    else:
        payload = bytes((0,)) + raw
    if len(payload) > budget:
        raise ValueError("dictionary log does not fit the storage budget")
    return payload


def _unpack_dictionary(payload: bytes, dictionary: tuple[str, ...], budget: int) -> tuple[str, ...]:
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if not payload:
        return ()
    if len(payload) > budget or payload[0] not in (0, _ZLIB):
        raise ValueError("invalid dictionary payload mode")
    body = _decompress(payload[1:]) if payload[0] == _ZLIB else payload[1:]
    return _parse_dictionary_body(body, dictionary)


def _fit_dictionary_suffix(records: tuple[str, ...], dictionary: tuple[str, ...], budget: int) -> bytes:
    best: bytes | None = None
    best_count = -1
    for start in range(len(records)):
        candidate = records[start:]
        try:
            payload = _dictionary_payload(candidate, dictionary, budget)
        except ValueError:
            continue
        if len(candidate) > best_count:
            best, best_count = payload, len(candidate)
    if best is not None:
        return best
    raw = records[-1].encode("utf-8")
    best = None
    best_length = -1
    for length in range(len(raw), -1, -1):
        candidate = _valid_utf8_suffix(raw[-length:]) if length else b""
        if not candidate or len(candidate) <= best_length:
            continue
        try:
            payload = _dictionary_payload((_decode_utf8(candidate),), dictionary, budget)
        except ValueError:
            continue
        best, best_length = payload, len(candidate)
    return best or b""


@dataclass(frozen=True, slots=True)
class TextStore:
    """A deterministic store whose only retained state is its payload."""

    budget: int
    codec: str
    dictionary: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.budget, bool) or not isinstance(self.budget, int):
            raise TypeError("budget must be an integer")
        if self.budget < 8 or self.budget > 65_535:
            raise ValueError("budget must be between 8 and 65535 bytes")
        if self.codec not in SUPPORTED_CODECS:
            raise ValueError(f"unsupported codec: {self.codec!r}")
        if self.codec != "dictionary_recent" and self.dictionary:
            raise ValueError("only dictionary_recent accepts a shared dictionary")
        if len(self.dictionary) > MAX_DICTIONARY_ENTRIES or len(set(self.dictionary)) != len(self.dictionary):
            raise ValueError("shared dictionary entries must be unique and bounded")
        for record in self.dictionary:
            _validate_record(record)

    def empty(self) -> bytes:
        return b""

    def _payload(self, payload: bytes | StoredText) -> bytes:
        if isinstance(payload, StoredText):
            if payload.codec != self.codec:
                raise ValueError("stored payload codec differs")
            payload = payload.payload
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes or StoredText")
        if len(payload) > self.budget:
            raise ValueError("payload exceeds the storage budget")
        return payload

    def records(self, payload: bytes | StoredText) -> tuple[str, ...]:
        payload = self._payload(payload)
        if self.codec == "compressed_recent":
            text = _decode_utf8(_decompress(payload)) if payload else ""
            return tuple(text.splitlines()) if text else ()
        if self.codec == "compressed_diverse":
            return _unpack_log(payload, self.budget)
        return _unpack_dictionary(payload, self.dictionary, self.budget)

    def decode(self, payload: bytes | StoredText) -> str:
        payload = self._payload(payload)
        if self.codec == "compressed_recent":
            return _decode_utf8(_decompress(payload)) if payload else ""
        return "\n".join(self.records(payload))

    def update(self, previous: bytes | StoredText, record: str) -> bytes:
        record = _validate_record(record)
        previous = self._payload(previous)
        if self.codec == "compressed_recent":
            prior = self.decode(previous)
            combined = f"{prior}\n{record}" if prior else record
            return _fit_zlib_suffix(combined, self.budget)
        if self.codec == "compressed_diverse":
            prior = _unpack_log(previous, self.budget)
            selected = _diverse_records((*prior, record), self.budget)
            try:
                return _log_payload(selected, self.budget)
            except ValueError:
                return _fit_diverse_single(record, self.budget)
        prior = _unpack_dictionary(previous, self.dictionary, self.budget)
        return _fit_dictionary_suffix((*prior, record), self.dictionary, self.budget)

    def pack(self, payload: bytes) -> StoredText:
        return StoredText(self._payload(payload), self.codec)

    def shared_dictionary_bytes(self) -> int:
        """Return shared codebook bytes, excluded from each payload budget."""

        if not self.dictionary:
            return 0
        return 2 + sum(2 + len(_varint(len(record.encode("utf-8")))) + len(record.encode("utf-8"))
                       for record in self.dictionary)
