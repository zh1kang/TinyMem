import pytest
import torch

from tinymem.memory.latest_fact_tokens import LatestFactTokenRetention


class CharacterTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text.encode("ascii"))

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        return bytes(ids).decode("ascii")


def text(policy, state):
    ids, valid = policy.materialize(state, pad_id=0)
    return policy.tokenizer.decode(ids[0, valid[0]].tolist(), skip_special_tokens=False)


def test_latest_fact_replaces_old_value_without_query_or_symbolic_storage():
    policy = LatestFactTokenRetention(100, 128, CharacterTokenizer())
    initial = policy.empty(1, device="cpu")
    first = policy.append_sentence(initial, "Mary went to the kitchen.")
    old = first.payload.clone()
    second = policy.append_sentence(first, "John moved to the bathroom.")
    result = policy.append_sentence(second, "Mary travelled to the hallway.")
    assert text(policy, result) == "John moved to the bathroom.\nMary travelled to the hallway.\n\n"
    assert torch.equal(first.payload, old) and not initial.payload.any()
    assert result.nbytes == initial.nbytes
    assert set(vars(result)) == {"payload"}
    assert set(vars(policy)) == {"capacity", "vocab_size", "bits_per_id", "length_bits", "payload_bytes", "tokenizer"}


def test_capacity_pressure_evicts_oldest_person_and_never_truncates_a_fact():
    policy = LatestFactTokenRetention(57, 128, CharacterTokenizer())
    state = policy.empty(1, device="cpu")
    for sentence in ("Mary went to the kitchen.", "John moved to the bathroom.", "Daniel moved to the garden."):
        state = policy.append_sentence(state, sentence)
    assert text(policy, state) == "John moved to the bathroom.\nDaniel moved to the garden.\n\n"
    assert state.nbytes == policy.payload_bytes


def test_oversized_update_removes_stale_same_person_but_keeps_other_facts():
    policy = LatestFactTokenRetention(60, 128, CharacterTokenizer())
    state = policy.empty(1, device="cpu")
    state = policy.append_sentence(state, "Mary went to the kitchen.")
    state = policy.append_sentence(state, "John moved to the bathroom.")
    state = policy.append_sentence(state, "Mary went to the " + "verylongroom" * 10 + ".")
    assert text(policy, state) == "John moved to the bathroom.\n\n"


def test_unrelated_prose_is_ignored_by_fixed_grammar_without_a_gold_fact_flag():
    policy = LatestFactTokenRetention(40, 128, CharacterTokenizer())
    state = policy.append_sentence(policy.empty(1, device="cpu"), "Mary went to the kitchen.")
    for _ in range(50):
        previous = state.payload.clone()
        state = policy.append_sentence(state, "The rain made a soft sound.")
        assert torch.equal(state.payload, previous)
        assert state.nbytes == policy.payload_bytes


@pytest.mark.parametrize("sentence", [
    "", "Mary moved to the kitchen.\nJohn moved to the garden.", "Where is Mary?", " Mary moved to the kitchen.",
    "Mary moved to the kitchen. John moved to the garden.",
    "Mary moved to the kitchen and John moved to the garden.",
    "Mary moved to the kitchen and John went to the garden.",
])
def test_invalid_sentence_boundary_is_rejected(sentence):
    policy = LatestFactTokenRetention(60, 128, CharacterTokenizer())
    with pytest.raises(ValueError):
        policy.append_sentence(policy.empty(1, device="cpu"), sentence)


@pytest.mark.parametrize("boundary", ["\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"])
def test_all_splitlines_boundaries_are_rejected_before_storage(boundary):
    policy = LatestFactTokenRetention(100, 128, CharacterTokenizer())
    with pytest.raises(ValueError, match="one complete sentence"):
        policy.append_sentence(policy.empty(1, device="cpu"), f"Mary{boundary} moved to the kitchen.")


def test_corrupt_or_duplicated_retained_sentence_format_is_rejected():
    policy = LatestFactTokenRetention(100, 128, CharacterTokenizer())
    for stored in ("Mary moved to the kitchen.", "arbitrary words\n\n", "Mary moved to the kitchen.\nMary moved to the garden.\n\n",
                   "Mary moved to the kitchen. John moved to the garden.\n\n",
                   "Mary moved to the kitchen and John moved to the garden.\n\n"):
        ids = torch.tensor([policy.tokenizer.encode(stored, add_special_tokens=False)])
        state = policy.append(policy.empty(1, device="cpu"), ids, torch.ones_like(ids, dtype=torch.bool))
        with pytest.raises(ValueError):
            policy.append_sentence(state, "John moved to the bedroom.")
    with pytest.raises(ValueError, match="single stream"):
        policy.append_sentence(policy.empty(2, device="cpu"), "John moved to the bedroom.")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
def test_sentence_policy_matches_cpu_on_mps():
    policy = LatestFactTokenRetention(60, 128, CharacterTokenizer())
    cpu, mps = (policy.empty(1, device=device) for device in ("cpu", "mps"))
    for sentence in ("Mary went to the kitchen.", "John moved to the bathroom.", "Mary travelled to the garden."):
        cpu = policy.append_sentence(cpu, sentence)
        mps = policy.append_sentence(mps, sentence)
    assert torch.equal(cpu.payload, mps.payload.cpu())
    assert text(policy, cpu) == text(policy, mps)
