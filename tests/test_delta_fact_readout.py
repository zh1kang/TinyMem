"""The phase-two reader must preserve matrix values and the byte boundary."""

import pytest
import torch

from test_readout_runner import tiny_reader
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.delta_fact_readout import SlotReadout, own_state, read_answer


@pytest.mark.parametrize("width", [8, 32])
def test_readout_accepts_finite_values_outside_unit_interval_and_keeps_gradients(width):
    bridge = SlotReadout(memory_width=width, reader_width=16)
    values = torch.full((2, 2, width), 2.0, requires_grad=True)
    state = LatentSlotState(values, torch.ones(2, 2, dtype=torch.bool))
    memory = bridge(state)
    assert memory.shape == (2, 2, 16)
    memory.square().sum().backward()
    assert torch.isfinite(values.grad).all() and values.grad.abs().sum() > 0
    assert torch.equal(values.detach(), torch.full_like(values, 2.0))


def test_inference_owns_only_one_detached_history_and_roundtrips(tiny_reader):
    from safetensors.torch import load, save

    bridge = SlotReadout(memory_width=8, reader_width=16).eval().requires_grad_(False)
    values = torch.randn(3, 2, 8, requires_grad=True)
    source = LatentSlotState(values, torch.ones(3, 2, dtype=torch.bool))
    frozen = own_state(source, 1)
    assert frozen.nbytes == 66
    assert not frozen.values.requires_grad
    restored = load(save({"values": frozen.values, "valid": frozen.valid}))
    restored = LatentSlotState(restored["values"], restored["valid"])
    args = (torch.tensor([1, 2]), torch.tensor([3, 4]))
    expected = read_answer(tiny_reader, bridge, frozen, *args, max_new_tokens=2)
    assert expected == read_answer(tiny_reader, bridge, restored, *args, max_new_tokens=2)
    assert expected["memory_positions"] == 2
    with torch.no_grad():
        values[1].add_(10)
    assert not torch.equal(values[1], frozen.values[0])


def test_inference_rejects_shared_backing_storage_and_trainable_reader(tiny_reader):
    bridge = SlotReadout(memory_width=8, reader_width=16).eval().requires_grad_(False)
    source = LatentSlotState(torch.zeros(3, 2, 8), torch.ones(3, 2, dtype=torch.bool))
    view = LatentSlotState(source.values[:1], source.valid[:1])
    args = (torch.tensor([1, 2]), torch.tensor([3, 4]))
    with pytest.raises(ValueError, match="own"):
        read_answer(tiny_reader, bridge, view, *args)
    tiny_reader.model.get_input_embeddings().weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        read_answer(tiny_reader, bridge, own_state(source, 0), *args)


def test_masked_slots_do_not_reach_the_reader(tiny_reader):
    bridge = SlotReadout(memory_width=8, reader_width=16).eval().requires_grad_(False)
    state = LatentSlotState(torch.full((1, 2, 8), float("nan")), torch.zeros(1, 2, dtype=torch.bool))
    assert torch.equal(bridge(state), torch.zeros(1, 2, 16))
    result = read_answer(tiny_reader, bridge, own_state(state, 0), torch.tensor([1]),
                         torch.tensor([2, 3]), max_new_tokens=1)
    assert result["memory_positions"] == 0


def test_readout_rejects_wrong_shape_nonfinite_valid_values_and_dtype():
    bridge = SlotReadout(memory_width=8, reader_width=16)
    with pytest.raises(ValueError, match="shape"):
        bridge(LatentSlotState(torch.zeros(1, 2, 32), torch.ones(1, 2, dtype=torch.bool)))
    with pytest.raises(ValueError, match="finite"):
        bridge(LatentSlotState(torch.full((1, 2, 8), float("inf")), torch.ones(1, 2, dtype=torch.bool)))
    with pytest.raises(TypeError, match="FP32"):
        bridge(LatentSlotState(torch.zeros(1, 2, 8, dtype=torch.float64), torch.ones(1, 2, dtype=torch.bool)))
