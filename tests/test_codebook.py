import pytest
import torch

from tinymem.memory.codebook import GumbelSoftmaxCodebook


def test_codebook_returns_discrete_slots_and_trace() -> None:
    torch.manual_seed(7)
    codebook = GumbelSoftmaxCodebook(model_width=4, codebook_size=3)
    logits = torch.randn(2, 2, 3)
    valid = torch.ones(2, 2, dtype=torch.bool)

    output = codebook(logits, valid, temperature=0.7)

    assert output.values.shape == (2, 2, 4)
    assert output.indices.shape == (2, 2)
    assert output.assignments.shape == (2, 2, 3)
    assert output.probabilities.shape == (2, 2, 3)
    assert output.indices.dtype == torch.long
    assert torch.equal(
        output.assignments.sum(dim=-1),
        torch.ones(2, 2),
    )
    assert torch.all((output.assignments == 0) | (output.assignments == 1))
    assert torch.equal(output.indices, output.assignments.argmax(dim=-1))
    torch.testing.assert_close(
        output.values,
        output.assignments @ codebook.embedding.weight,
    )


def test_codebook_straight_through_gradient_reaches_inputs_and_codes() -> None:
    torch.manual_seed(11)
    codebook = GumbelSoftmaxCodebook(model_width=3, codebook_size=4)
    logits = torch.randn(2, 2, 4, requires_grad=True)
    valid = torch.ones(2, 2, dtype=torch.bool)

    output = codebook(logits, valid, temperature=1.0)
    output.values.square().sum().backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) > 0
    assert codebook.embedding.weight.grad is not None
    assert torch.isfinite(codebook.embedding.weight.grad).all()
    assert torch.count_nonzero(codebook.embedding.weight.grad) > 0


def test_codebook_evaluation_uses_deterministic_argmax() -> None:
    codebook = GumbelSoftmaxCodebook(model_width=2, codebook_size=3)
    codebook.eval()
    with torch.no_grad():
        codebook.embedding.weight.copy_(
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        )
    logits = torch.tensor([[[1.0, 4.0, 2.0], [7.0, 0.0, 1.0]]])
    valid = torch.ones(1, 2, dtype=torch.bool)

    torch.manual_seed(1)
    first = codebook(logits, valid, temperature=0.5)
    torch.manual_seed(99)
    second = codebook(logits, valid, temperature=0.5)

    assert torch.equal(first.indices, torch.tensor([[1, 0]]))
    assert torch.equal(first.values, torch.tensor([[[3.0, 4.0], [1.0, 2.0]]]))
    assert torch.equal(first.indices, second.indices)
    assert torch.equal(first.values, second.values)
    torch.testing.assert_close(first.probabilities, second.probabilities)


def test_lower_temperature_sharpens_evaluation_probabilities() -> None:
    codebook = GumbelSoftmaxCodebook(model_width=2, codebook_size=3)
    codebook.eval()
    logits = torch.tensor([[[3.0, 2.0, 1.0]]])
    valid = torch.ones(1, 1, dtype=torch.bool)

    warm = codebook(logits, valid, temperature=2.0)
    cold = codebook(logits, valid, temperature=0.25)

    assert cold.probabilities.max() > warm.probabilities.max()


def test_codebook_soft_evaluation_uses_probability_mixture() -> None:
    codebook = GumbelSoftmaxCodebook(
        model_width=2,
        codebook_size=3,
        evaluation_mode="soft",
    )
    codebook.eval()
    logits = torch.tensor([[[3.0, 2.0, 1.0]]])
    valid = torch.ones(1, 1, dtype=torch.bool)

    output = codebook(logits, valid, temperature=1.0)

    torch.testing.assert_close(output.assignments, output.probabilities)
    torch.testing.assert_close(
        output.values,
        output.probabilities @ codebook.embedding.weight,
    )
    assert torch.count_nonzero(output.assignments) == 3


@pytest.mark.parametrize("mode", [None, 1, "discrete", ""])
def test_codebook_rejects_invalid_evaluation_mode(mode: object) -> None:
    expected = TypeError if not isinstance(mode, str) else ValueError

    with pytest.raises(expected):
        GumbelSoftmaxCodebook(
            model_width=2,
            codebook_size=3,
            evaluation_mode=mode,
        )


def test_training_probabilities_exclude_sampling_noise() -> None:
    codebook = GumbelSoftmaxCodebook(model_width=2, codebook_size=3)
    logits = torch.tensor([[[1.0, 2.0, 3.0]]])
    valid = torch.ones(1, 1, dtype=torch.bool)

    torch.manual_seed(1)
    first = codebook(logits, valid, temperature=1.0)
    torch.manual_seed(99)
    second = codebook(logits, valid, temperature=1.0)

    torch.testing.assert_close(first.probabilities, second.probabilities)


def test_codebook_zeros_invalid_slots_and_marks_their_indices() -> None:
    codebook = GumbelSoftmaxCodebook(model_width=3, codebook_size=4)
    logits = torch.randn(2, 2, 4, requires_grad=True)
    valid = torch.tensor([[True, False], [False, True]])

    output = codebook(logits, valid, temperature=1.0)
    output.values.sum().backward()

    assert torch.equal(output.indices[~valid], torch.full((2,), -1))
    assert torch.equal(
        output.values[~valid],
        torch.zeros_like(output.values[~valid]),
    )
    assert torch.equal(
        output.assignments[~valid],
        torch.zeros_like(output.assignments[~valid]),
    )
    assert torch.equal(
        output.probabilities[~valid],
        torch.zeros_like(output.probabilities[~valid]),
    )
    assert logits.grad is not None
    assert torch.equal(logits.grad[~valid], torch.zeros_like(logits.grad[~valid]))


@pytest.mark.parametrize("name", ["model_width", "codebook_size"])
@pytest.mark.parametrize("value", [True, 0, -1, 3.5, "4"])
def test_codebook_rejects_invalid_sizes(name: str, value: object) -> None:
    arguments = {"model_width": 4, "codebook_size": 8, name: value}
    expected = (
        ValueError
        if isinstance(value, int) and not isinstance(value, bool)
        else TypeError
    )

    with pytest.raises(expected):
        GumbelSoftmaxCodebook(**arguments)


@pytest.mark.parametrize(
    ("logits", "valid", "error"),
    [
        ([[[1.0, 2.0, 3.0]]], torch.tensor([[True]]), TypeError),
        (torch.ones(1, 3), torch.tensor([[True]]), ValueError),
        (
            torch.ones(1, 1, 3, dtype=torch.long),
            torch.tensor([[True]]),
            TypeError,
        ),
        (torch.ones(1, 1, 4), torch.tensor([[True]]), ValueError),
        (torch.ones(1, 1, 3), [[True]], TypeError),
        (torch.ones(1, 1, 3), torch.ones(1, 2, dtype=torch.bool), ValueError),
        (torch.ones(1, 1, 3), torch.ones(1, 1), TypeError),
    ],
)
def test_codebook_rejects_invalid_inputs(
    logits: object,
    valid: object,
    error: type[Exception],
) -> None:
    codebook = GumbelSoftmaxCodebook(model_width=2, codebook_size=3)

    with pytest.raises(error):
        codebook(logits, valid, temperature=1.0)


@pytest.mark.parametrize(
    ("temperature", "error"),
    [
        (True, TypeError),
        ("1.0", TypeError),
        (0.0, ValueError),
        (-1.0, ValueError),
        (float("inf"), ValueError),
        (float("nan"), ValueError),
    ],
)
def test_codebook_rejects_invalid_temperature(
    temperature: object,
    error: type[Exception],
) -> None:
    codebook = GumbelSoftmaxCodebook(model_width=2, codebook_size=3)

    with pytest.raises(error):
        codebook(
            torch.ones(1, 1, 3),
            torch.ones(1, 1, dtype=torch.bool),
            temperature=temperature,
        )
