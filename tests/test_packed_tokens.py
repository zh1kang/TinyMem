import io

import pytest
import torch

from tinymem.memory.packed_tokens import PackedTokenRetention, PackedTokenState


@pytest.mark.parametrize("vocab", [1, 2, 3, 16, 255, 256, 257, 32768, 151936, 2**31])
@pytest.mark.parametrize("capacity", [1, 7, 25])
def test_packed_ids_match_independent_integer_codec(vocab, capacity):
    generator = torch.Generator().manual_seed(13)
    policy = PackedTokenRetention(capacity, vocab)
    ids = torch.randint(vocab, (2, capacity), generator=generator)
    ids[0, -1] = vocab - 1
    state = policy.append(policy.empty(2, device="cpu"), ids, torch.ones_like(ids, dtype=torch.bool))
    for row in range(2):
        word = capacity | sum(value << (policy.length_bits + index * policy.bits_per_id) for index, value in enumerate(ids[row].tolist()))
        expected = word.to_bytes(policy.payload_bytes, "little")
        assert bytes(state.payload[row].tolist()) == expected
    actual, valid = policy.materialize(state, pad_id=0)
    assert torch.equal(actual, ids)
    assert valid.all()
    assert state.nbytes == 2 * policy.payload_bytes


def test_qwen_ids_retain_29_tokens_in_66_actual_bytes():
    policy = PackedTokenRetention(29, 151936)
    assert policy.bits_per_id == 18
    ids = torch.arange(100, 180).unsqueeze(0)
    state = policy.append(policy.empty(1, device="cpu"), ids, torch.ones_like(ids, dtype=torch.bool))
    assert state.nbytes == 66
    assert policy.materialize(state, pad_id=0)[0].tolist() == [list(range(151, 180))]
    assert PackedTokenRetention(30, 151936).empty(1, device="cpu").nbytes == 69
    assert set(vars(state)) == {"payload"}
    assert all(not isinstance(value, torch.Tensor) for value in vars(policy).values())


@pytest.mark.parametrize("capacity", [2, 3, 4, 8, 15, 16, 31, 32, 63, 64])
def test_packed_length_header_handles_width_boundaries(capacity):
    policy = PackedTokenRetention(capacity, 257)
    for length in (0, 1, capacity - 1, capacity):
        ids = torch.full((1, length), 256, dtype=torch.int32)
        state = policy.append(policy.empty(1, device="cpu"), ids, torch.ones_like(ids, dtype=torch.bool))
        decoded, valid = policy.materialize(state, pad_id=0)
        assert valid.sum() == length
        assert (decoded[valid] == 256).all()
        expected_word = length | sum(256 << (capacity.bit_length() + index * 9) for index in range(length))
        assert bytes(state.payload[0].tolist()) == expected_word.to_bytes(policy.payload_bytes, "little")


def test_partial_masked_append_is_out_of_place_and_partition_invariant():
    policy = PackedTokenRetention(7, 151936)
    generator = torch.Generator().manual_seed(37)
    ids = torch.randint(policy.vocab_size, (3, 39), generator=generator)
    valid = torch.rand(3, 39, generator=generator) > 0.4
    valid[2] = False
    initial = policy.empty(3, device="cpu")
    state = initial
    for start, end in ((0, 3), (3, 19), (19, 19), (19, 39)):
        previous, old_bytes = state, state.payload.clone()
        state = policy.append(state, ids[:, start:end], valid[:, start:end])
        if start == 0:
            first, first_bytes = state, state.payload.clone()
        assert initial.payload.count_nonzero() == 0
        assert state.nbytes == initial.nbytes
        assert torch.equal(previous.payload, old_bytes)
    once = policy.append(initial, ids, valid)
    assert torch.equal(state.payload, once.payload)
    assert torch.equal(first.payload, first_bytes)
    actual, mask = policy.materialize(state, pad_id=151935)
    for row in range(3):
        expected = ids[row, valid[row]][-7:]
        assert torch.equal(actual[row, mask[row]], expected)
    assert actual[2].tolist() == [151935] * 7
    assert not mask[2].any()


def test_materialization_and_roundtrip_do_not_retain_a_second_history():
    policy = PackedTokenRetention(3, 151936)
    state = policy.append(policy.empty(1, device="cpu"), torch.tensor([[17, 151935]]), torch.ones(1, 2, dtype=torch.bool))
    payload = state.payload.clone()
    decoded, valid = policy.materialize(state, pad_id=0)
    decoded.zero_()
    valid.zero_()
    assert torch.equal(state.payload, payload)
    buffer = io.BytesIO()
    torch.save({"payload": state.payload}, buffer)
    buffer.seek(0)
    restored = PackedTokenState(**torch.load(buffer, weights_only=True))
    assert torch.equal(policy.materialize(restored, pad_id=0)[0], torch.tensor([[17, 151935, 0]]))
    assert restored.nbytes == state.nbytes == 7


def test_large_backing_allocation_is_not_hidden_by_packed_view():
    payload = torch.zeros(1, 100, dtype=torch.uint8)
    state = PackedTokenState(payload[:, :7])
    assert state.nbytes == 100
    assert PackedTokenRetention(3, 151936).materialize(state, pad_id=0)[0].shape == (1, 3)


@pytest.mark.parametrize("ids", [[-1], [151936], [2**31]])
def test_invalid_input_cannot_wrap_into_a_different_token(ids):
    policy = PackedTokenRetention(3, 151936)
    with pytest.raises(ValueError, match="vocabulary"):
        policy.append(policy.empty(1, device="cpu"), torch.tensor([ids]), torch.ones(1, 1, dtype=torch.bool))


def test_malformed_packed_payloads_fail_closed():
    policy = PackedTokenRetention(2, 3)
    for payload, message in ((13, "vocabulary"), (64, "unused payload"), (4, "unused token"), (3, "capacity")):
        state = PackedTokenState(torch.tensor([[payload]], dtype=torch.uint8))
        with pytest.raises(ValueError, match=message):
            policy.materialize(state, pad_id=0)
        with pytest.raises(ValueError, match=message):
            policy.append(state, torch.zeros(1, 0, dtype=torch.int64), torch.zeros(1, 0, dtype=torch.bool))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
def test_mps_packing_matches_cpu_and_exact_state_bytes():
    policy = PackedTokenRetention(29, 151936)
    ids = torch.tensor([[0, 151935, 65535, 65536, 17, 100000]])
    mask = torch.ones_like(ids, dtype=torch.bool)
    cpu = policy.append(policy.empty(1, device="cpu"), ids, mask)
    gpu = policy.append(policy.empty(1, device="mps"), ids.to("mps"), mask.to("mps"))
    assert torch.equal(cpu.payload, gpu.payload.cpu())
    assert torch.equal(policy.materialize(cpu, pad_id=0)[0], policy.materialize(gpu, pad_id=0)[0].cpu())
    assert gpu.nbytes == cpu.nbytes == 66
