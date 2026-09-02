import pytest
import torch

from tinymem.memory.discrete_compressor import DiscreteMemoryCompressor


def test_discrete_compressor_returns_fixed_quantized_slots() -> None:
    compressor = DiscreteMemoryCompressor(
        4,
        codebook_size=8,
        summary_slots=2,
    )
    compressor.eval()
    hidden = torch.randn(3, 5, 4)
    valid = torch.ones(3, 5, dtype=torch.bool)

    output = compressor.compress(hidden, valid)

    assert output.values.shape == (3, 2, 4)
    assert output.valid.shape == (3, 2)
    assert output.indices.shape == (3, 2)
    assert output.assignments.shape == (3, 2, 8)
    assert output.probabilities.shape == (3, 2, 8)
    assert output.logits.shape == (3, 2, 8)
    assert output.prequantized.shape == (3, 2, 4)
    torch.testing.assert_close(
        output.values,
        compressor.codebook.embedding(output.indices),
    )


def test_discrete_compressor_preserves_the_shared_forward_contract() -> None:
    compressor = DiscreteMemoryCompressor(
        3,
        codebook_size=4,
        summary_slots=2,
    )
    hidden = torch.randn(2, 4, 3)
    valid = torch.ones(2, 4, dtype=torch.bool)

    values, summary_valid = compressor(hidden, valid)

    assert values.shape == (2, 2, 3)
    assert summary_valid.shape == (2, 2)


def test_discrete_compressor_backpropagates_through_quantization() -> None:
    torch.manual_seed(17)
    compressor = DiscreteMemoryCompressor(
        4,
        codebook_size=6,
        summary_slots=2,
    )
    hidden = torch.randn(2, 5, 4, requires_grad=True)
    valid = torch.tensor(
        [[True, True, False, False, False], [True, True, True, True, True]]
    )

    output = compressor.compress(hidden, valid)
    output.values.square().sum().backward()

    parameters = (
        compressor.summarizer.queries,
        compressor.summarizer.projection.weight,
        compressor.logit_projection.weight,
        compressor.codebook.embedding.weight,
    )
    assert hidden.grad is not None
    assert torch.equal(hidden.grad[0, 2:], torch.zeros(3, 4))
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_discrete_compressor_zeros_empty_rows() -> None:
    compressor = DiscreteMemoryCompressor(
        3,
        codebook_size=4,
        summary_slots=2,
    )
    hidden = torch.randn(2, 4, 3)
    valid = torch.tensor(
        [[False, False, False, False], [True, False, False, False]]
    )

    output = compressor.compress(hidden, valid)

    assert torch.equal(
        output.valid,
        torch.tensor([[False, False], [True, True]]),
    )
    assert torch.equal(output.indices[0], torch.tensor([-1, -1]))
    assert torch.equal(output.values[0], torch.zeros(2, 3))
    assert torch.equal(output.probabilities[0], torch.zeros(2, 4))


def test_discrete_compressor_temperature_is_checkpointed() -> None:
    compressor = DiscreteMemoryCompressor(
        2,
        codebook_size=4,
        summary_slots=1,
        temperature=2.0,
    )
    compressor.set_temperature(0.35)
    restored = DiscreteMemoryCompressor(
        2,
        codebook_size=4,
        summary_slots=1,
    )

    restored.load_state_dict(compressor.state_dict())

    assert restored.temperature == pytest.approx(0.35)


@pytest.mark.parametrize("name", ["codebook_size", "summary_slots"])
@pytest.mark.parametrize("value", [True, 0, -1, 2.5, "4"])
def test_discrete_compressor_rejects_invalid_sizes(
    name: str,
    value: object,
) -> None:
    arguments = {"codebook_size": 4, "summary_slots": 1, name: value}
    expected = (
        ValueError
        if isinstance(value, int) and not isinstance(value, bool)
        else TypeError
    )

    with pytest.raises(expected):
        DiscreteMemoryCompressor(3, **arguments)


@pytest.mark.parametrize(
    ("temperature", "error"),
    [
        (True, TypeError),
        ("1", TypeError),
        (0.0, ValueError),
        (-1.0, ValueError),
        (float("inf"), ValueError),
        (float("nan"), ValueError),
    ],
)
def test_discrete_compressor_rejects_invalid_temperature(
    temperature: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        DiscreteMemoryCompressor(
            3,
            codebook_size=4,
            summary_slots=1,
            temperature=temperature,
        )
