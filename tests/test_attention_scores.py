import pytest
import torch

from tinymem.memory.attention_scores import attention_received


def test_attention_received_averages_heads_and_sums_queries() -> None:
    probabilities = torch.tensor(
        [
            [
                [[0.7, 0.3], [0.2, 0.8]],
                [[0.5, 0.5], [0.4, 0.6]],
            ]
        ],
        dtype=torch.float64,
    )

    scores = attention_received(probabilities)

    assert scores.dtype == torch.float32
    assert scores.shape == (1, 2)
    torch.testing.assert_close(scores, torch.tensor([[0.9, 1.1]]))


def test_attention_received_preserves_batch_rows() -> None:
    probabilities = torch.tensor(
        [
            [[[[0.8, 0.2]]]],
            [[[[0.1, 0.9]]]],
        ]
    ).reshape(2, 1, 1, 2)

    scores = attention_received(probabilities)

    torch.testing.assert_close(
        scores,
        torch.tensor([[0.8, 0.2], [0.1, 0.9]]),
    )


def test_attention_received_total_mass_equals_query_count() -> None:
    logits = torch.randn(3, 4, 5, 6)
    probabilities = torch.softmax(logits, dim=-1)

    scores = attention_received(probabilities)

    torch.testing.assert_close(scores.sum(dim=-1), torch.full((3,), 5.0))


def test_attention_received_rejects_non_tensor_input() -> None:
    with pytest.raises(TypeError, match="torch.Tensor"):
        attention_received([[1.0]])


def test_attention_received_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        attention_received(torch.ones(1, 2, 3))


def test_attention_received_rejects_integer_values() -> None:
    with pytest.raises(TypeError, match="floating-point"):
        attention_received(torch.ones(1, 1, 1, 1, dtype=torch.long))


def test_attention_received_rejects_empty_dimensions() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        attention_received(torch.empty(1, 1, 0, 2))


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_attention_received_rejects_nonfinite_values(
    invalid_value: float,
) -> None:
    probabilities = torch.tensor([[[[invalid_value, 0.0]]]])

    with pytest.raises(ValueError, match="finite"):
        attention_received(probabilities)


def test_attention_received_rejects_negative_values() -> None:
    probabilities = torch.tensor([[[[-0.1, 1.1]]]])

    with pytest.raises(ValueError, match="nonnegative"):
        attention_received(probabilities)


def test_attention_received_requires_normalized_probabilities() -> None:
    probabilities = torch.tensor([[[[0.2, 0.2]]]])

    with pytest.raises(ValueError, match="sum to one"):
        attention_received(probabilities)
