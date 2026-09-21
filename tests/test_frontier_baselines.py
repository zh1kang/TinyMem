from __future__ import annotations

import inspect

import pytest

from tinymem.studies.frontier.baselines import (
    SUPPORTED_CODECS,
    TextStore,
    build_statement_dictionary,
)

RECORDS = (
    "Mary moved to the bathroom.",
    "John went to the kitchen.",
    "Sandra travelled to the garden.",
    "Mary moved to the office.",
)


@pytest.mark.parametrize("codec", SUPPORTED_CODECS)
def test_empty_and_incremental_payloads_are_bounded(codec: str) -> None:
    dictionary = RECORDS if codec == "dictionary_recent" else ()
    store = TextStore(256, codec, dictionary)
    payload = store.empty()
    assert payload == b""
    for record in RECORDS:
        payload = store.update(payload, record)
        assert isinstance(payload, bytes)
        assert len(payload) <= store.budget
        assert store.decode(payload)
    assert store.decode(store.pack(payload)) == store.decode(payload)


def test_query_blind_api_has_no_question_argument() -> None:
    assert list(inspect.signature(TextStore.update).parameters) == [
        "self", "previous", "record",
    ]
    assert list(inspect.signature(TextStore.decode).parameters) == ["self", "payload"]


def test_compressed_recent_round_trips_unicode_and_keeps_newest_suffix() -> None:
    store = TextStore(256, "compressed_recent")
    payload = b""
    for record in ("αβγ moved to the garden.", "最新 fact: John went to the office."):
        payload = store.update(payload, record)
    decoded = store.decode(payload)
    assert "最新 fact" in decoded
    assert decoded.endswith("office.")
    assert len(payload) <= 256


def test_compressed_recent_uses_actual_fit_for_nonmonotonic_compression() -> None:
    store = TextStore(64, "compressed_recent")
    payload = b""
    for record in ("a " * 30, "b " * 30, "c " * 30):
        payload = store.update(payload, record)
    assert len(payload) <= 64
    assert store.decode(payload).rstrip().endswith("c")


def test_dictionary_uses_ids_and_preserves_unknown_literals() -> None:
    store = TextStore(256, "dictionary_recent", RECORDS[:2])
    payload = store.update(b"", RECORDS[0])
    payload = store.update(payload, "A new statement unseen during training.")
    records = store.records(payload)
    assert records[-1] == "A new statement unseen during training."
    assert store.decode(payload) == "\n".join(records)
    assert store.shared_dictionary_bytes() > 0


def test_dictionary_known_ids_are_compact() -> None:
    store = TextStore(64, "dictionary_recent", RECORDS)
    one = store.update(b"", RECORDS[0])
    ten = b""
    for record in (RECORDS * 3)[:10]:
        ten = store.update(ten, record)
    assert len(one) <= 3
    assert len(ten) <= 21


def test_dictionary_builder_is_deterministic_and_deduplicates_training_texts() -> None:
    dictionary = build_statement_dictionary((RECORDS[1], RECORDS[0], RECORDS[1]))
    assert dictionary == tuple(sorted((RECORDS[0], RECORDS[1]), key=lambda value: value.encode("utf-8")))
    with pytest.raises(TypeError):
        build_statement_dictionary(RECORDS[0])  # type: ignore[arg-type]


def test_dictionary_recent_drops_old_suffix_records_only_when_needed() -> None:
    store = TextStore(64, "dictionary_recent", RECORDS[:1])
    payload = b""
    for record in RECORDS:
        payload = store.update(payload, record)
    records = store.records(payload)
    assert records
    assert records[-1] == RECORDS[-1]
    assert len(payload) <= 64


def test_diverse_store_keeps_latest_and_novel_record() -> None:
    store = TextStore(128, "compressed_diverse")
    payload = b""
    for record in (
        "Mary moved to the bathroom.",
        "Mary moved to the bathroom.",
        "Sandra travelled to the garden.",
        "John went to the office.",
    ):
        payload = store.update(payload, record)
    records = store.records(payload)
    assert records[-1] == "John went to the office."
    assert "Sandra travelled to the garden." in records
    assert len(payload) <= 128


@pytest.mark.parametrize("codec", SUPPORTED_CODECS)
def test_tampering_is_rejected(codec: str) -> None:
    dictionary = RECORDS if codec == "dictionary_recent" else ()
    store = TextStore(256, codec, dictionary)
    payload = store.update(b"", RECORDS[0])
    tampered = bytearray(payload)
    if codec == "dictionary_recent":
        tampered[0] = 2
    else:
        tampered[-1] ^= 1
    with pytest.raises(ValueError):
        store.decode(bytes(tampered))


def test_invalid_records_and_configuration_fail_at_boundary() -> None:
    with pytest.raises(ValueError):
        TextStore(7, "compressed_recent")
    with pytest.raises(ValueError):
        TextStore(64, "other")
    store = TextStore(64, "compressed_recent")
    with pytest.raises(TypeError):
        store.update(b"", 3)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        store.update(b"", "word " * 129)
    with pytest.raises(ValueError):
        store.decode(b"not a payload")
