import pytest
import torch

from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.readout_controls import controlled_state, state_donors


def test_controls_own_detached_payloads_without_mutating_source():
    values = torch.linspace(-1, 1, 16).reshape(1, 2, 8).requires_grad_()
    source = LatentSlotState(values, torch.tensor([[True, False]]))
    for control in ("normal", "zero", "no_memory"):
        result = controlled_state(source, control)
        assert result.nbytes == 66
        assert not result.values.requires_grad and result.values.grad_fn is None
        assert result.values.data_ptr() != source.values.data_ptr()
        assert result.valid.data_ptr() != source.valid.data_ptr()
        if control == "normal":
            assert torch.equal(result.values, source.values)
        else:
            assert result.values.count_nonzero() == 0
        assert torch.equal(result.valid, source.valid if control != "no_memory" else torch.zeros_like(source.valid))
        result.values.zero_()
        result.valid.zero_()
        assert source.values.count_nonzero() == 16
        assert source.valid.tolist() == [[True, False]]
    with pytest.raises(ValueError, match="control"):
        controlled_state(source, "shuffled")


def test_donors_are_fixed_identity_based_derangement():
    ids = ("world-c", "world-a", "world-b")
    expected = {"world-a": "world-b", "world-b": "world-c", "world-c": "world-a"}
    assert state_donors(ids) == expected
    assert state_donors(tuple(reversed(ids))) == expected
    assert set(expected) == set(expected.values())
    assert all(target != donor for target, donor in expected.items())


@pytest.mark.parametrize("ids", [(), ("one",), ("one", "one"), ("", "two"), (1, "two"), "ab"])
def test_donors_reject_invalid_history_ids(ids):
    with pytest.raises(ValueError, match="history"):
        state_donors(ids)
