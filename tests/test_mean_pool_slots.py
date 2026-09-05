from copy import deepcopy

import pytest
import torch

from tinymem.memory.mean_pool_slots import MeanPoolSlotWriter
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.recurrent_memory import NativeRecurrentMemory


def test_masked_mean_is_computed_before_compression_and_fifo_append():
    writer = MeanPoolSlotWriter(16, 4, 2, hidden_width=8)
    hidden = torch.randn(1, 4, 16)
    hidden[:, 1::2] = float("nan")
    valid = torch.tensor([[True, False, True, False]])
    result = writer(writer.empty(1), hidden, valid)
    pooled = ((hidden[:, 0] + hidden[:, 2]) / 2).unsqueeze(1)
    expected = writer.output_projection(torch.nn.functional.gelu(writer.input_projection(pooled))).tanh()
    torch.testing.assert_close(result.values[:, -1:], expected)
    assert result.values[:, 0].count_nonzero() == 0
    assert result.valid.tolist() == [[False, True]]


@pytest.mark.parametrize("slots", [1, 2, 4])
def test_fifo_evicts_oldest_without_mutation_or_unbounded_storage(slots):
    writer = MeanPoolSlotWriter(16, 4, slots)
    state = writer.empty(1)
    history = []
    with torch.no_grad():
        for _ in range(10):
            old, saved = state, deepcopy(state)
            state = writer(state, torch.randn(1, 3, 16), torch.ones(1, 3, dtype=torch.bool))
            history.append(state.values[:, -1:].clone())
            expected = torch.cat(history[-slots:], dim=1)
            torch.testing.assert_close(state.values[:, -expected.shape[1]:], expected)
            torch.testing.assert_close(old.values, saved.values)
            assert torch.equal(old.valid, saved.valid)
            assert state.nbytes == slots * 17
            assert set(vars(state)) == {"values", "valid"}
            assert state.values.grad_fn is None
    assert not list(writer.buffers())
    assert not any(isinstance(value, torch.Tensor) for value in vars(writer).values())


def test_fifo_gradients_reach_surviving_slots_and_unmasked_inputs_only():
    writer = MeanPoolSlotWriter(16, 4, 3)
    values = torch.randn(1, 3, 4)
    values[:, 1] = float("nan")
    values.requires_grad_()
    state = LatentSlotState(values, torch.tensor([[True, False, True]]))
    hidden = torch.randn(1, 4, 16)
    hidden[:, 1::2] = float("nan")
    hidden.requires_grad_()
    result = writer(state, hidden, torch.tensor([[True, False, True, False]]))
    result.values.sum().backward()
    assert torch.isfinite(result.values).all()
    assert result.valid.tolist() == [[False, True, True]]
    assert values.grad[:, :2].count_nonzero() == 0
    torch.testing.assert_close(values.grad[:, 2], torch.ones_like(values[:, 2]))
    assert torch.isfinite(hidden.grad).all() and hidden.grad[:, 1::2].count_nonzero() == 0
    assert hidden.grad[:, ::2].abs().sum() > 0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in writer.parameters())


def test_empty_rows_keep_their_state_validity_and_identity_gradient():
    writer = MeanPoolSlotWriter(16, 4, 2)
    values = torch.randn(2, 2, 4, requires_grad=True)
    state = LatentSlotState(values, torch.tensor([[True, False], [True, True]]))
    hidden = torch.randn(2, 3, 16)
    hidden[0] = float("nan")
    hidden.requires_grad_()
    result = writer(state, hidden, torch.tensor([[False, False, False], [True, True, True]]))
    torch.testing.assert_close(result.values[0], values[0])
    assert torch.equal(result.valid[0], state.valid[0])
    result.values.sum().backward()
    torch.testing.assert_close(values.grad[0], torch.ones_like(values[0]))
    assert hidden.grad[0].count_nonzero() == 0
    assert hidden.grad[1].abs().sum() > 0


@pytest.mark.parametrize("field", ["reader_width", "memory_width", "slots", "hidden_width"])
@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_mean_pool_dimensions_are_positive_integers(field, invalid):
    arguments = dict(reader_width=16, memory_width=4, slots=2, hidden_width=8)
    arguments[field] = invalid
    with pytest.raises(ValueError, match=field):
        MeanPoolSlotWriter(**arguments)


def test_mean_pool_rejects_incompatible_inputs():
    writer = MeanPoolSlotWriter(16, 4, 2)
    state = writer.empty(1)
    with pytest.raises(ValueError, match="batch_size"):
        writer.empty(True)
    with pytest.raises(ValueError, match="nonempty tokens"):
        writer(state, torch.empty(1, 0, 16), torch.empty(1, 0, dtype=torch.bool))
    with pytest.raises(ValueError, match="boolean"):
        writer(state, torch.zeros(1, 2, 16), torch.ones(1, 2))
    with pytest.raises(ValueError, match="state shape"):
        writer(writer.empty(2), torch.zeros(1, 2, 16), torch.ones(1, 2, dtype=torch.bool))
    with pytest.raises(TypeError, match="dtype"):
        writer(state, torch.zeros(1, 2, 16, dtype=torch.float64), torch.ones(1, 2, dtype=torch.bool))


def test_native_mean_pool_choice_has_the_same_retained_budget():
    model = NativeRecurrentMemory(2048, memory_width=8, slots=2, segment_length=32,
                                  writer_kind="mean_pool", aggregation_width=64)
    assert isinstance(model.writer, MeanPoolSlotWriter)
    assert model.writer_kind == "mean_pool"
    assert model.writer.empty(1).nbytes == 66
    assert sum(parameter.numel() for parameter in model.parameters()) == 148040
