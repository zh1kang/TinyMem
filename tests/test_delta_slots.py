import pytest
import torch
from safetensors.torch import load_file, save_file

from tinymem.memory.delta_slots import DeltaSlotWriter, delta_update
from tinymem.memory.recurrent_slots import LatentSlotState


def test_delta_update_matches_independent_outer_product_oracle_without_mutating_input():
    matrix = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    key = torch.tensor([[0.6, 0.8]])
    value = torch.tensor([[5.0, 6.0]])
    beta = torch.tensor([0.25])

    result = delta_update(matrix, key, value, beta)

    expected = torch.tensor([[[1.3, 2.24], [3.4, 4.32]]])
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(matrix, torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_delta_update_supports_float_dtypes_and_batched_oracle(dtype):
    generator = torch.Generator().manual_seed(4)
    matrix = torch.randn(3, 4, 5, generator=generator, dtype=dtype)
    key = torch.randn(3, 4, generator=generator, dtype=dtype)
    key = 0.5 * key / key.norm(dim=-1, keepdim=True)
    value = torch.randn(3, 5, generator=generator, dtype=dtype)
    beta = torch.tensor([0.0, 0.25, 1.0], dtype=dtype)

    actual = delta_update(matrix, key, value, beta)
    prediction = torch.bmm(key.unsqueeze(1), matrix).squeeze(1)
    expected = matrix + key.unsqueeze(-1) * (beta[:, None] * (value - prediction)).unsqueeze(1)
    torch.testing.assert_close(actual, expected)
    assert actual.dtype is dtype
    assert actual.data_ptr() != matrix.data_ptr()


def test_delta_update_is_differentiable_against_independent_scalar_oracle():
    matrix = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    key = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    key = 0.5 * key / key.norm(dim=-1, keepdim=True)
    value = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    beta = torch.tensor([0.2, 0.8], dtype=torch.float64, requires_grad=True)

    def function(matrix_, key_, value_, beta_):
        return delta_update(matrix_, key_, value_, beta_)

    assert torch.autograd.gradcheck(function, (matrix, key, value, beta), eps=1e-6, atol=1e-5)


def test_delta_update_preserves_an_orthogonal_association_and_has_no_decay():
    matrix = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    key = torch.tensor([[1.0, 0.0]])
    value = torch.tensor([[5.0, 6.0]])
    updated = delta_update(matrix, key, value, torch.tensor([1.0]))

    torch.testing.assert_close(updated[0, 0], value[0], rtol=0, atol=0)
    torch.testing.assert_close(updated[0, 1], matrix[0, 1], rtol=0, atol=0)
    torch.testing.assert_close(
        delta_update(matrix, key, value, torch.tensor([0.0])), matrix, rtol=0, atol=0,
    )


def test_delta_update_cross_key_effect_matches_readout_calculation():
    matrix = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    read_key = torch.tensor([[0.6, 0.8]])
    update_key = torch.tensor([[1.0, 0.0]])
    updated = delta_update(matrix, update_key, torch.tensor([[5.0, 6.0]]), torch.tensor([1.0]))

    old_read = torch.bmm(read_key.unsqueeze(1), matrix).squeeze(1)
    new_read = torch.bmm(read_key.unsqueeze(1), updated).squeeze(1)
    torch.testing.assert_close(new_read - old_read, torch.tensor([[2.4, 2.4]]), rtol=0, atol=0)
    torch.testing.assert_close(updated[0, 1], matrix[0, 1], rtol=0, atol=0)


def test_delta_update_repetition_of_known_association_is_exactly_unchanged():
    matrix = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    key = torch.tensor([[0.6, 0.8]])
    value = torch.bmm(key.unsqueeze(1), matrix).squeeze(1)

    updated = delta_update(matrix, key, value, torch.tensor([0.75]))

    torch.testing.assert_close(updated, matrix, rtol=0, atol=0)


def test_delta_update_partial_beta_repetition_converges_to_target_and_matches_interference_oracle():
    matrix = torch.zeros(1, 2, 2)
    key = torch.tensor([[0.8, 0.6]])
    value = torch.tensor([[2.0, -1.0]])
    beta = torch.tensor([0.25])
    first = delta_update(matrix, key, value, beta)
    independent = matrix + key.unsqueeze(-1) * (
        beta[:, None] * (value - torch.bmm(key.unsqueeze(1), matrix).squeeze(1))
    ).unsqueeze(1)
    torch.testing.assert_close(first, independent)
    current = matrix
    for _ in range(50):
        current = delta_update(current, key, value, beta)
    assert (value - torch.bmm(key.unsqueeze(1), current).squeeze(1)).norm() < 1e-5


@pytest.mark.parametrize(
    ("matrix", "key", "value", "beta", "error"),
    [
        (torch.zeros(1, 2, 3), torch.zeros(1, 2), torch.zeros(1, 3), torch.zeros(2), "beta"),
        (torch.zeros(1, 2, 3), torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1), "key"),
        (torch.zeros(1, 2, 3), torch.tensor([[2.0, 0.0]]), torch.zeros(1, 3), torch.zeros(1), "norm"),
        (torch.zeros(1, 2, 3), torch.zeros(1, 2), torch.zeros(1, 3), torch.tensor([-0.1]), "[0, 1]"),
        (torch.zeros(1, 2, 3), torch.zeros(1, 2), torch.zeros(1, 3), torch.tensor([1.1]), "[0, 1]"),
    ],
)
def test_delta_update_rejects_invalid_boundaries(matrix, key, value, beta, error):
    with pytest.raises((ValueError, TypeError), match=error):
        delta_update(matrix, key, value, beta)


def test_writer_has_two_fp32_slots_and_bounded_state():
    writer = DeltaSlotWriter(16, 8, key_width=4, hidden_width=12)
    state = writer.empty(3)

    assert (writer.reader_width, writer.memory_width, writer.slots) == (16, 8, 2)
    assert (writer.key_width, writer.value_width, writer.hidden_width) == (4, 4, 12)
    assert writer.attention_query.shape == (16,)
    assert state.values.shape == (3, 2, 8)
    assert state.values.dtype == torch.float32
    assert state.valid.shape == (3, 2)
    assert state.nbytes == 3 * (2 * 8 * 4 + 2)
    assert not list(writer.buffers())
    assert not any(isinstance(value, torch.Tensor) for value in vars(writer).values())


def test_fixed_beta_preserves_shared_initialization_and_removes_gate_parameters():
    torch.manual_seed(31)
    learned = DeltaSlotWriter(8, 32, key_width=8)
    torch.manual_seed(31)
    fixed = DeltaSlotWriter(8, 32, key_width=8, fixed_beta=0.75)

    assert fixed.fixed_beta == 0.75
    assert "beta_projection.weight" in learned.state_dict()
    assert "beta_projection.bias" in learned.state_dict()
    assert "beta_projection.weight" not in fixed.state_dict()
    assert "beta_projection.bias" not in fixed.state_dict()
    assert sum(parameter.numel() for parameter in learned.parameters()) - sum(
        parameter.numel() for parameter in fixed.parameters()
    ) == 65
    for name, tensor in fixed.state_dict().items():
        torch.testing.assert_close(tensor, learned.state_dict()[name], rtol=0, atol=0)


def test_disabled_normalization_is_bitwise_default_and_does_not_consume_rng():
    torch.manual_seed(34)
    default = DeltaSlotWriter(8, 32, key_width=8)
    default_rng = torch.random.get_rng_state()
    torch.manual_seed(34)
    disabled = DeltaSlotWriter(8, 32, key_width=8, normalize_hidden=False)
    disabled_rng = torch.random.get_rng_state()

    assert disabled.normalize_hidden is False
    assert default.state_dict().keys() == disabled.state_dict().keys()
    for name, tensor in default.state_dict().items():
        torch.testing.assert_close(tensor, disabled.state_dict()[name], rtol=0, atol=0)
    assert torch.equal(default_rng, disabled_rng)
    hidden = torch.randn(2, 3, 8)
    valid = torch.ones(2, 3, dtype=torch.bool)
    default_output = default(default.empty(2), hidden, valid)
    disabled_output = disabled(disabled.empty(2), hidden, valid)
    torch.testing.assert_close(default_output.values, disabled_output.values, rtol=0, atol=0)
    assert torch.equal(default_output.valid, disabled_output.valid)

    mask = valid.unsqueeze(-1)
    safe_hidden = torch.where(mask, hidden, torch.zeros_like(hidden))
    scores = torch.sum(safe_hidden * default.attention_query, dim=-1)
    scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    weights = scores.softmax(dim=-1) * valid
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
    pooled = torch.sum(weights.unsqueeze(-1) * safe_hidden, dim=1)
    features = torch.nn.functional.gelu(default.input_projection(pooled))
    key = torch.nn.functional.normalize(default.key_projection(features), dim=-1, eps=1e-6)
    value = torch.tanh(default.value_projection(features))
    beta = torch.sigmoid(default.beta_projection(features)).squeeze(-1)
    expected = delta_update(
        torch.zeros(2, default.key_width, default.value_width), key, value, beta,
    ).reshape_as(default_output.values)
    torch.testing.assert_close(default_output.values, expected, rtol=0, atol=0)


def test_normalization_validates_bool_and_handles_zero_variance_backward():
    with pytest.raises(TypeError, match="normalize_hidden"):
        DeltaSlotWriter(8, 32, key_width=8, normalize_hidden=1)

    torch.manual_seed(35)
    writer = DeltaSlotWriter(8, 32, key_width=8, fixed_beta=0.75, normalize_hidden=True)
    with torch.no_grad():
        writer.input_projection.weight.zero_()
        writer.input_projection.bias.fill_(3.0)
    hidden = torch.randn(2, 3, 8, requires_grad=True)
    valid = torch.ones(2, 3, dtype=torch.bool)
    result = writer(writer.empty(2), hidden, valid)
    result.values.square().sum().backward()

    assert torch.isfinite(result.values).all()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in writer.parameters())


def test_enabled_normalization_adds_no_parameters_or_constructor_rng_draws():
    torch.manual_seed(350)
    fixed = DeltaSlotWriter(8, 32, key_width=8, fixed_beta=0.75)
    fixed_rng = torch.random.get_rng_state()
    torch.manual_seed(350)
    normalized = DeltaSlotWriter(8, 32, key_width=8, fixed_beta=0.75, normalize_hidden=True)
    normalized_rng = torch.random.get_rng_state()

    assert normalized.state_dict().keys() == fixed.state_dict().keys()
    for name, tensor in fixed.state_dict().items():
        torch.testing.assert_close(tensor, normalized.state_dict()[name], rtol=0, atol=0)
    assert torch.equal(fixed_rng, normalized_rng)


def test_normalization_keeps_heterogeneous_negative_projection_active():
    torch.manual_seed(36)
    writer = DeltaSlotWriter(8, 32, key_width=8, fixed_beta=0.75, normalize_hidden=True)
    with torch.no_grad():
        writer.input_projection.weight.zero_()
        coordinates = torch.arange(writer.hidden_width, dtype=torch.float32)
        writer.input_projection.weight[:, 0] = 0.15 * torch.sin(coordinates)
        writer.input_projection.weight[:, 1] = 0.08 * torch.cos(0.7 * coordinates)
        writer.input_projection.bias.copy_(torch.linspace(-4.0, -1.0, writer.hidden_width))
    hidden = torch.zeros(1, 1, 8)
    hidden[..., 0] = -1.0
    hidden[..., 1] = 0.5
    hidden.requires_grad_()
    valid = torch.ones(1, 1, dtype=torch.bool)
    projected = writer.input_projection(hidden[:, 0])
    assert bool((projected < 0).all())
    assert projected.std() > 0

    result = writer(writer.empty(1), hidden, valid)
    result.values.square().sum().backward()

    assert result.values.abs().sum() > 0
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert hidden.grad.abs().sum() > 1e-5
    assert writer.input_projection.weight.grad is not None
    assert torch.isfinite(writer.input_projection.weight.grad).all()


def test_fixed_beta_is_constant_for_each_input_and_actual_parameters_receive_gradients():
    torch.manual_seed(32)
    writer = DeltaSlotWriter(8, 32, key_width=8, fixed_beta=0.75)
    first = torch.randn(2, 3, 8, requires_grad=True)
    second = torch.randn(2, 5, 8, requires_grad=True)
    first_valid = torch.ones(2, 3, dtype=torch.bool)
    second_valid = torch.ones(2, 5, dtype=torch.bool)

    first_beta = writer.encode_statement(first, first_valid)[2]
    second_beta = writer.encode_statement(second, second_valid)[2]
    torch.testing.assert_close(first_beta, torch.full((2,), 0.75), rtol=0, atol=0)
    torch.testing.assert_close(second_beta, torch.full((2,), 0.75), rtol=0, atol=0)

    output = writer(writer.empty(2), first, first_valid)
    output.values.square().sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in writer.parameters())
    assert first.grad is not None and torch.isfinite(first.grad).all()


def test_fixed_beta_checkpoint_round_trip_preserves_the_258_byte_state_shape(tmp_path):
    torch.manual_seed(33)
    writer = DeltaSlotWriter(8, 32, key_width=8, fixed_beta=0.75)
    state = writer.empty(3)
    assert state.values.shape == (3, 2, 32)
    assert state.valid.shape == (3, 2)
    assert state.nbytes == 3 * 258

    path = tmp_path / "fixed-writer.safetensors"
    save_file({name: tensor.detach().cpu() for name, tensor in writer.state_dict().items()}, str(path))
    restored = DeltaSlotWriter(8, 32, key_width=8, fixed_beta=0.75)
    restored.load_state_dict(load_file(str(path)), strict=True)
    hidden = torch.randn(3, 4, 8)
    valid = torch.ones(3, 4, dtype=torch.bool)
    torch.testing.assert_close(
        restored(restored.empty(3), hidden, valid).values,
        writer(writer.empty(3), hidden, valid).values,
        rtol=0,
        atol=0,
    )


def test_writer_encodes_only_current_statement_and_round_trips_safetensors_state(tmp_path):
    torch.manual_seed(7)
    writer = DeltaSlotWriter(8, 8, key_width=4)
    hidden = torch.randn(2, 5, 8)
    valid = torch.tensor([[True, True, False, False, False], [True, True, True, False, False]])
    key, value, beta = writer.encode_statement(hidden, valid)
    assert key.shape == (2, 4)
    assert value.shape == (2, 4)
    assert beta.shape == (2,)
    torch.testing.assert_close(key.norm(dim=-1), torch.ones(2), atol=1e-5, rtol=1e-5)
    assert torch.all((beta >= 0) & (beta <= 1))

    path = tmp_path / "writer.safetensors"
    save_file({name: tensor.detach().cpu() for name, tensor in writer.state_dict().items()}, str(path))
    other = DeltaSlotWriter(8, 8, key_width=4)
    other.load_state_dict(load_file(str(path)), strict=True)
    other_key, other_value, other_beta = other.encode_statement(hidden, valid)
    torch.testing.assert_close(other_key, key, rtol=0, atol=0)
    torch.testing.assert_close(other_value, value, rtol=0, atol=0)
    torch.testing.assert_close(other_beta, beta, rtol=0, atol=0)
    state = writer(writer.empty(2), hidden, valid)
    restored_state = other(other.empty(2), hidden, valid)
    torch.testing.assert_close(restored_state.values, state.values, rtol=0, atol=0)
    assert torch.equal(restored_state.valid, state.valid)


def test_writer_matches_delta_update_oracle_and_repetition_converges():
    torch.manual_seed(11)
    writer = DeltaSlotWriter(8, 8, key_width=4)
    state = writer.empty(2)
    hidden = torch.randn(2, 4, 8)
    valid = torch.ones(2, 4, dtype=torch.bool)
    key, value, beta = writer.encode_statement(hidden, valid)
    expected = delta_update(torch.zeros(2, 4, 4), key, value, beta).reshape(2, 2, 8)
    first = writer(state, hidden, valid)
    torch.testing.assert_close(first.values, expected)
    second = writer(first, hidden, valid)
    assert (second.values - first.values).norm() < (first.values - state.values).norm()
    repeated = writer(second, hidden, valid)
    assert (repeated.values - second.values).norm() < (second.values - first.values).norm()
    assert first.valid.tolist() == [[True, True], [True, True]]


def test_writer_ignores_nan_padding_and_preserves_all_empty_rows_with_identity_gradient():
    torch.manual_seed(13)
    writer = DeltaSlotWriter(8, 8, key_width=4)
    values = torch.randn(2, 2, 8, requires_grad=True)
    state = LatentSlotState(values, torch.tensor([[True, True], [False, False]]))
    hidden = torch.randn(2, 4, 8, requires_grad=True)
    hidden.data[0, 2:] = float("nan")
    hidden.data[1] = float("nan")
    valid = torch.tensor([[True, True, False, False], [False, False, False, False]])

    result = writer(state, hidden, valid)
    clean = writer(
        LatentSlotState(values.detach().clone(), state.valid),
        hidden.detach().nan_to_num(),
        valid,
    )
    torch.testing.assert_close(result.values[0], clean.values[0])
    torch.testing.assert_close(result.values[1], values[1])
    result.values.sum().backward()
    torch.testing.assert_close(values.grad[1], torch.ones_like(values.grad[1]))
    assert hidden.grad[0, 2:].count_nonzero() == 0
    assert hidden.grad[1].count_nonzero() == 0
    assert torch.isfinite(hidden.grad).all()


def test_writer_propagates_feature_gradients_and_rejects_disagreeing_state_flags():
    writer = DeltaSlotWriter(8, 8, key_width=4)
    hidden = torch.randn(1, 3, 8, requires_grad=True)
    valid = torch.ones(1, 3, dtype=torch.bool)
    output = writer(writer.empty(1), hidden, valid)
    output.values.square().sum().backward()
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in writer.parameters())
    assert writer.input_projection.weight.grad.abs().sum() > 0

    invalid = LatentSlotState(torch.zeros(1, 2, 8), torch.tensor([[True, False]]))
    with pytest.raises(ValueError, match="valid flags"):
        writer(invalid, hidden.detach(), valid)

    with pytest.raises(TypeError, match="writer parameters"):
        writer.double()(writer.empty(1), hidden.detach(), valid)


@pytest.mark.parametrize(
    "arguments",
    [
        (0, 8, 4),
        (8, 0, 4),
        (8, 8, 0),
        (8, 8, 3),
    ],
)
def test_writer_rejects_invalid_dimensions(arguments):
    with pytest.raises(ValueError):
        DeltaSlotWriter(arguments[0], arguments[1], key_width=arguments[2])
