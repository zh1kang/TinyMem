import hashlib

import pytest
import torch

from tinymem.memory.fingerprint_facts import FingerprintFactRetention, entity_fingerprint
from tinymem.memory.packed_tokens import PackedTokenState


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_fixed_payload_updates_and_absent_lookup(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    memory = FingerprintFactRetention(66)
    state = memory.empty(device=device)
    state = memory.append_sentence(state, "Mary moved to the kitchen.")
    previous = state.payload.clone()
    updated = memory.append_sentence(state, "Mary went to the office.")
    assert memory.lookup(updated, "Mary") == "office"
    assert memory.lookup(updated, "John") == "unknown"
    assert memory.lookup(state, "Mary") == "kitchen"
    assert torch.equal(state.payload, previous)
    assert updated.nbytes == updated.payload.untyped_storage().nbytes() == 66
    assert set(vars(updated)) == {"payload"}
    assert FingerprintFactRetention(66).lookup(updated, "Mary") == "office"


def test_eviction_uses_fingerprint_update_recency():
    memory = FingerprintFactRetention(5)
    assert memory.capacity == 2
    state = memory.empty()
    for sentence in ("Mary moved to the kitchen.", "John went to the office.",
                     "Mary moved to the garden.", "Sandra went to the bedroom."):
        state = memory.append_sentence(state, sentence)
    assert memory.lookup(state, "Mary") == "garden"
    assert memory.lookup(state, "Sandra") == "bedroom"
    assert memory.lookup(state, "John") == "unknown"


def test_collisions_overwrite_and_can_false_match_absent_name(monkeypatch):
    monkeypatch.setattr("tinymem.memory.fingerprint_facts.entity_fingerprint", lambda person: 123)
    memory = FingerprintFactRetention(66)
    state = memory.append_sentence(memory.empty(), "Mary moved to the kitchen.")
    state = memory.append_sentence(state, "John went to the office.")
    assert memory.lookup(state, "Mary") == "office"
    assert memory.lookup(state, "John") == "office"
    assert memory.lookup(state, "Absent") == "office"
    assert len(memory._unpack(state)) == 1


def test_hash_definition_is_fixed():
    assert entity_fingerprint("Mary") == int.from_bytes(hashlib.sha256(b"Mary").digest()[:2], "little")


@pytest.mark.parametrize("name", ["", " Mary", "Mary\nJohn", 1])
def test_invalid_name(name):
    with pytest.raises(ValueError):
        entity_fingerprint(name)


@pytest.mark.parametrize("budget", [0, -1, True, 2, 3.5])
def test_invalid_budget(budget):
    with pytest.raises(ValueError):
        FingerprintFactRetention(budget)


def test_malformed_state_rejected():
    memory = FingerprintFactRetention(66)
    for word in ((1 << 527), 31, 1 | (7 << (memory.length_bits + 16))):
        state = PackedTokenState(torch.tensor([list(word.to_bytes(66, "little"))], dtype=torch.uint8))
        with pytest.raises(ValueError):
            memory.lookup(state, "Mary")
    with pytest.raises(ValueError, match="format"):
        memory.lookup(FingerprintFactRetention(65).empty(), "Mary")


def test_room_outside_declared_set_rejected():
    memory = FingerprintFactRetention(66)
    with pytest.raises(ValueError, match="six declared"):
        memory.append_sentence(memory.empty(), "Mary moved to the attic.")
