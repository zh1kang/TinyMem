from copy import deepcopy

import pytest
import torch

from tinymem.memory.query_pool_slots import QueryPoolSlotWriter
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.recurrent_memory import NativeRecurrentMemory


def test_full_width_pooling_happens_before_nonlinear_compression():
    writer = QueryPoolSlotWriter(16, 4, 2, hidden_width=8)
    with torch.no_grad():
        writer.queries.zero_()
    hidden = torch.randn(1, 4, 16)
    valid = torch.tensor([[True, False, True, False]])
    pooled = []
    hook = writer.input_projection.register_forward_pre_hook(lambda module, args: pooled.append(args[0].detach().clone()))
    try:
        result = writer(writer.empty(1), hidden, valid)
    finally:
        hook.remove()
    expected = ((hidden[:, 0] + hidden[:, 2]) / 2).unsqueeze(1).expand(-1, 2, -1)
    torch.testing.assert_close(pooled[0], expected)
    compressed = writer.output_projection(torch.nn.functional.gelu(writer.input_projection(expected)))
    gate = torch.sigmoid(writer.gate(torch.cat((torch.zeros_like(compressed), compressed), dim=-1)))
    torch.testing.assert_close(result.values, gate * compressed.tanh())


def test_wide_temporary_features_do_not_increase_retained_bytes_or_mutate_state():
    writer = QueryPoolSlotWriter(32, 8, 2, hidden_width=64)
    state = writer.empty(2)
    initial = deepcopy(state)
    original = state
    valid = torch.tensor([[True, True, True], [False, False, False]])
    with torch.no_grad():
        for _ in range(20):
            previous = state
            state = writer(state, torch.randn(2, 3, 32), valid)
            torch.testing.assert_close(state.values[1], previous.values[1])
            assert state.nbytes == 132
            assert state.values.shape == (2, 2, 8)
            assert set(vars(state)) == {"values", "valid"}
            assert state.values.grad_fn is None
    torch.testing.assert_close(original.values, initial.values)
    assert state.valid.tolist() == [[True, True], [False, False]]
    assert not any(isinstance(value, torch.Tensor) for value in vars(writer).values())
    assert not list(writer.buffers())


def test_invalid_slots_and_padding_cannot_affect_writes_or_gradients():
    writer = QueryPoolSlotWriter(16, 4, 2, hidden_width=8)
    values = torch.randn(1, 2, 4)
    values[:, 1] = float("nan")
    values.requires_grad_()
    state = LatentSlotState(values, torch.tensor([[True, False]]))
    hidden = torch.randn(1, 4, 16)
    hidden[:, 1::2] = float("nan")
    hidden.requires_grad_()
    valid = torch.tensor([[True, False, True, False]])
    result = writer(state, hidden, valid)
    cleaned = writer(LatentSlotState(values.nan_to_num(), state.valid), hidden.nan_to_num(), valid)
    torch.testing.assert_close(result.values, cleaned.values)
    result.values.square().sum().backward()
    assert torch.isfinite(values.grad).all() and values.grad[:, 1].count_nonzero() == 0
    assert torch.isfinite(hidden.grad).all() and hidden.grad[:, 1::2].count_nonzero() == 0
    assert hidden.grad[:, ::2].abs().sum() > 0
    assert all(torch.isfinite(parameter.grad).all() for parameter in writer.parameters())


def test_empty_segment_preserves_populated_state_and_its_identity_gradient():
    writer = QueryPoolSlotWriter(16, 4, 2, hidden_width=8)
    values = torch.randn(1, 2, 4, requires_grad=True)
    state = LatentSlotState(values, torch.tensor([[True, False]]))
    hidden = torch.full((1, 3, 16), float("nan"), requires_grad=True)
    result = writer(state, hidden, torch.zeros(1, 3, dtype=torch.bool))
    torch.testing.assert_close(result.values, values)
    assert torch.equal(result.valid, state.valid)
    result.values.sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))
    assert hidden.grad.count_nonzero() == 0
    assert all(parameter.grad is not None and parameter.grad.count_nonzero() == 0 for parameter in writer.parameters())


@pytest.mark.parametrize("field", ["reader_width", "memory_width", "slots", "hidden_width"])
@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_query_pool_dimensions_are_positive_integers(field, invalid):
    arguments = dict(reader_width=16, memory_width=4, slots=2, hidden_width=8)
    arguments[field] = invalid
    with pytest.raises(ValueError, match=field):
        QueryPoolSlotWriter(**arguments)


def test_query_pool_rejects_incompatible_inputs():
    writer = QueryPoolSlotWriter(16, 4, 2)
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


def test_native_writer_choice_is_explicit_and_keeps_the_default_initialization():
    arguments = dict(reader_width=16, memory_width=4, slots=2, segment_length=4)
    torch.manual_seed(17)
    default = NativeRecurrentMemory(**arguments)
    torch.manual_seed(17)
    explicit = NativeRecurrentMemory(**arguments, writer_kind="narrow")
    assert default.writer_kind == explicit.writer_kind == "narrow"
    for name, value in default.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[name])
    pooled = NativeRecurrentMemory(**arguments, writer_kind="query_pool", aggregation_width=8)
    assert isinstance(pooled.writer, QueryPoolSlotWriter) and pooled.writer.hidden_width == 8
    assert pooled.writer.empty(1).nbytes == default.writer.empty(1).nbytes
    with pytest.raises(ValueError, match="writer_kind"):
        NativeRecurrentMemory(**arguments, writer_kind="other")
    with pytest.raises(ValueError, match="only supported"):
        NativeRecurrentMemory(**arguments, aggregation_width=8)
