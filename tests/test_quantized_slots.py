"""Behavioral tests for the fixed-grid recurrent slot memory."""

from __future__ import annotations

from io import BytesIO

import pytest
import torch

from tinymem.memory.quantized_slots import QuantizedSlotMemory


def _memory(slots: int = 2) -> QuantizedSlotMemory:
    torch.manual_seed(7)
    return QuantizedSlotMemory(16, slots=slots, memory_width=32, hidden_width=32, heads=4)


def _inputs(batch: int = 1, tokens: int = 5):
    hidden = torch.randn(batch, tokens, 16)
    valid = torch.ones(batch, tokens, dtype=torch.bool)
    return hidden, valid


@pytest.mark.parametrize("slots", [2, 8, 32])
def test_empty_shape_and_persistent_byte_count(slots: int) -> None:
    memory = _memory(slots)
    state = memory.empty(3)
    assert state.shape == (3, slots, 32)
    assert state.dtype == torch.float32
    assert torch.equal(state, torch.zeros_like(state))
    assert memory.persistent_bytes == slots * 32


def test_forward_is_on_signed_int8_grid_and_pack_roundtrips_exactly() -> None:
    memory = _memory()
    state = memory(memory.empty(1), *_inputs())
    scaled = state * 127
    assert torch.equal(scaled, torch.round(scaled))
    payload = memory.pack(state)
    assert isinstance(payload, bytes)
    assert len(payload) == 2 * 32
    restored = memory.unpack(payload)
    assert torch.equal(restored, state)
    assert memory.pack(restored) == payload


@pytest.mark.parametrize("training", [False, True])
def test_pack_unpack_preserves_every_continuing_quantized_write(training: bool) -> None:
    memory = _memory()
    memory.train(training)
    state = memory.empty(1)
    for _ in range(20):
        hidden, valid = _inputs(tokens=4)
        direct = memory(state, hidden, valid)
        resumed = memory(memory.unpack(memory.pack(state)), hidden, valid)
        assert torch.equal(direct, resumed)
        assert torch.equal(direct * 127, torch.round(direct * 127))
        state = direct


def test_pack_rejects_out_of_bounds_and_off_grid_values() -> None:
    memory = _memory()
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        memory.pack(torch.full((1, 2, 32), 1.01))
    with pytest.raises(ValueError, match="grid"):
        memory.pack(torch.full((1, 2, 32), 0.123))
    with pytest.raises(ValueError, match="exactly"):
        memory.unpack(b"\x00")
    with pytest.raises(ValueError, match="reserved"):
        memory.unpack(bytes([128]) * memory.persistent_bytes)


def test_all_padding_is_exact_no_op_and_masked_noise_is_ignored() -> None:
    memory = _memory()
    initial = memory(memory.empty(1), *_inputs(tokens=3))
    padded = torch.zeros(1, 4, 16)
    no_op = memory(initial, padded, torch.zeros(1, 4, dtype=torch.bool))
    assert torch.equal(no_op, initial)

    hidden = torch.randn(1, 4, 16)
    valid = torch.tensor([[True, False, True, False]])
    noisy = hidden.clone()
    noisy[:, ~valid[0]] = 1e12
    clean = hidden.clone()
    clean[:, ~valid[0]] = -1e12
    first = memory(initial, noisy, valid)
    second = memory(initial, clean, valid)
    assert torch.equal(first, second)


def test_writes_are_causal_and_reads_do_not_modify_state() -> None:
    memory = _memory()
    state = memory.empty(1)
    first, valid = _inputs(tokens=3)
    future_a = torch.randn(1, 3, 16)
    future_b = torch.randn(1, 3, 16)
    after_first_a = memory(state, first, valid)
    after_first_b = memory(state, first, valid)
    assert torch.equal(after_first_a, after_first_b)
    before_read = after_first_a.clone()
    _ = memory.memory_vectors(after_first_a)
    assert torch.equal(after_first_a, before_read)
    assert not torch.equal(memory(after_first_a, future_a, valid), memory(after_first_a, future_b, valid))


def test_gradients_reach_earlier_writes_and_all_trainable_parameters() -> None:
    memory = _memory()
    state = memory.empty(1)
    hidden_a, valid = _inputs(tokens=3)
    hidden_b, _ = _inputs(tokens=3)
    first = memory(state, hidden_a, valid)
    first.retain_grad()
    second = memory(first, hidden_b, valid)
    objective = memory.memory_vectors(second).square().mean()
    objective.backward()
    assert first.grad is not None
    assert float(first.grad.abs().sum()) > 0
    parameters = tuple(memory.parameters())
    assert parameters
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)


def test_gate_initializes_to_coordinatewise_half() -> None:
    memory = _memory()
    values = memory.gate_projection(torch.zeros(2, 2, 32 + 32))
    assert torch.equal(torch.sigmoid(values), torch.full_like(values, 0.5))


def test_checkpoint_restore_preserves_write_and_read_outputs() -> None:
    memory = _memory()
    state = memory(memory.empty(1), *_inputs())
    expected = memory.memory_vectors(state)
    buffer = BytesIO()
    torch.save(memory.state_dict(), buffer)
    restored = _memory()
    restored.load_state_dict(torch.load(BytesIO(buffer.getvalue()), weights_only=True), strict=True)
    actual_state = restored(restored.empty(1), *_inputs())
    torch.testing.assert_close(actual_state, state)
    torch.testing.assert_close(restored.memory_vectors(actual_state), expected)


def test_boundary_validation_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="divisible"):
        QuantizedSlotMemory(16, slots=2, hidden_width=30, heads=4)
    memory = _memory()
    with pytest.raises(ValueError, match="dimension"):
        memory(memory.empty(1), torch.zeros(1, 3, 15), torch.ones(1, 3, dtype=torch.bool))
    with pytest.raises(TypeError, match="FP32"):
        memory(memory.empty(1).double(), *_inputs())
    with pytest.raises(TypeError, match="bytes-like"):
        memory.unpack("invalid")
