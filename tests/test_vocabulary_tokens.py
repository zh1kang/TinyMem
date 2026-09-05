import pytest
import torch

from tinymem.memory.packed_tokens import PackedTokenState
from tinymem.memory.vocabulary_tokens import VocabularyTokenRetention


def retained(policy, state):
    ids, valid = policy.materialize(state, pad_id=0)
    return [row[mask].tolist() for row, mask in zip(ids, valid, strict=True)]


def test_short_codes_and_native_escapes_match_independent_integer_layout():
    policy = VocabularyTokenRetention(8, 16, [7, 2])
    assert policy.vocabulary == (2, 7) and policy.code_bits == 2 and policy.native_bits == 4
    ids = torch.tensor([[2, 15, 7, 0]])
    state = policy.append(policy.empty(1, device="cpu"), ids, torch.ones_like(ids, dtype=torch.bool))
    word, offset = 4, policy.length_bits
    for code, width in ((0, 2), (2, 2), (15, 4), (1, 2), (2, 2), (0, 4)):
        word |= code << offset
        offset += width
    assert bytes(state.payload[0].tolist()) == word.to_bytes(8, "little")
    assert retained(policy, state) == [[2, 15, 7, 0]]
    assert state.nbytes == 8 and policy.dictionary_serialized_bytes == 8


@pytest.mark.parametrize("budget", [4, 8, 19, 34])
@pytest.mark.parametrize("vocabulary", [[], [1], [1, 2, 3], list(range(21))])
def test_recent_suffix_is_lossless_partition_invariant_and_owns_only_payload(budget, vocabulary):
    policy = VocabularyTokenRetention(budget, 151936, vocabulary)
    generator = torch.Generator().manual_seed(29)
    ids = torch.randint(40, (2, 53), generator=generator)
    ids[:, ::7] = 151935
    valid = torch.rand(2, 53, generator=generator) > 0.2
    initial = policy.empty(2, device="cpu")
    once = policy.append(initial, ids, valid)
    state = initial
    for begin, end in ((0, 3), (3, 3), (3, 24), (24, 53)):
        old = state.payload.clone()
        previous = state
        state = policy.append(state, ids[:, begin:end], valid[:, begin:end])
        assert torch.equal(previous.payload, old)
    assert torch.equal(state.payload, once.payload) and not initial.payload.any()
    for row, values in enumerate(retained(policy, state)):
        expected = ids[row, valid[row]].tolist()
        short_width = max(1, len(vocabulary).bit_length())
        native_width = (151936 - 1).bit_length()
        while len(expected) >= 1 << policy.length_bits or policy.length_bits + sum(short_width + (0 if value in vocabulary else native_width) for value in expected) > budget * 8:
            expected.pop(0)
        assert values == expected
    assert state.nbytes == 2 * budget and set(vars(state)) == {"payload"}


def test_21_id_vocabulary_stores_29_known_qwen_ids_in_19_bytes():
    policy = VocabularyTokenRetention(19, 151936, list(range(21)))
    assert policy.capacity == 29 and policy.code_bits == 5 and policy.length_bits == 5
    assert policy.dictionary_serialized_bytes == 84
    ids = (torch.arange(29) % 21).unsqueeze(0)
    state = policy.append(policy.empty(1, device="cpu"), ids, torch.ones_like(ids, dtype=torch.bool))
    assert retained(policy, state) == [ids[0].tolist()] and state.nbytes == 19
    assert policy.fits(ids[0].tolist()) and not policy.fits([0] * 30)
    assert not policy.fits([151935] * 29)


def test_length_header_cannot_overflow_even_when_token_bits_fit():
    policy = VocabularyTokenRetention(33, 151936, [0])
    assert policy.capacity == 255 and policy.length_bits == 8
    assert policy.fits([0] * 255) and not policy.fits([0] * 256)
    ids = torch.zeros(1, 600, dtype=torch.long)
    state = policy.append(policy.empty(1, device="cpu"), ids, torch.ones_like(ids, dtype=torch.bool))
    assert retained(policy, state) == [[0] * 255]
    assert state.payload[0, 0] == 255 and not state.payload[0, 1:].any()


def test_malformed_code_escape_length_and_padding_fail_closed():
    policy = VocabularyTokenRetention(4, 7, [2, 4])
    for word, message in (
        (policy.capacity + 1, "capacity"),
        (1 | (3 << policy.length_bits), "unknown vocabulary"),
        (1 | (2 << policy.length_bits) | (7 << (policy.length_bits + 2)), "unseen valid"),
        (1 | (2 << policy.length_bits) | (2 << (policy.length_bits + 2)), "unseen valid"),
        (1 << policy.length_bits, "unused payload"),
    ):
        state = PackedTokenState(torch.tensor([list(word.to_bytes(4, "little"))], dtype=torch.uint8))
        with pytest.raises(ValueError, match=message):
            policy.materialize(state, pad_id=0)
    tiny = VocabularyTokenRetention(1, 151936, [0])
    escaped = PackedTokenState(torch.tensor([[1 | (1 << tiny.length_bits)]], dtype=torch.uint8))
    with pytest.raises(ValueError, match="truncated native-ID"):
        tiny.materialize(escaped, pad_id=0)


@pytest.mark.parametrize("vocabulary", [[1, 1], [True], [-1], [8], [1.0]])
def test_invalid_shared_vocabulary_is_rejected(vocabulary):
    with pytest.raises(ValueError):
        VocabularyTokenRetention(4, 8, vocabulary)


def test_invalid_inputs_and_format_are_rejected():
    policy = VocabularyTokenRetention(4, 8, [1, 2])
    with pytest.raises(ValueError, match="byte_budget"):
        VocabularyTokenRetention(True, 8, [1])
    with pytest.raises(ValueError, match="vocab_size"):
        VocabularyTokenRetention(4, 0, [1])
    with pytest.raises(ValueError, match="batch_size"):
        policy.empty(False, device="cpu")
    with pytest.raises(ValueError, match="native integers"):
        policy.fits([True])
    state = policy.empty(1, device="cpu")
    with pytest.raises(ValueError, match="native vocabulary"):
        policy.append(state, torch.tensor([[8]]), torch.tensor([[True]]))
    with pytest.raises(TypeError, match="int32"):
        policy.append(state, torch.tensor([[1.0]]), torch.tensor([[True]]))
    with pytest.raises(ValueError, match="boolean"):
        policy.append(state, torch.tensor([[1]]), torch.tensor([[1]]))
    with pytest.raises(ValueError, match="byte format"):
        policy.materialize(PackedTokenState(torch.zeros(1, 5, dtype=torch.uint8)), pad_id=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
def test_mps_matches_cpu_with_known_and_escaped_native_ids():
    policy = VocabularyTokenRetention(19, 151936, [2, 7, 10])
    ids = torch.tensor([[2, 151935, 7, 100000, 10] * 5])
    valid = torch.ones_like(ids, dtype=torch.bool)
    cpu = policy.append(policy.empty(1, device="cpu"), ids, valid)
    mps = policy.append(policy.empty(1, device="mps"), ids.to("mps"), valid.to("mps"))
    assert torch.equal(cpu.payload, mps.payload.cpu())
    assert retained(policy, cpu) == retained(policy, mps)
