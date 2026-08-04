import pytest
import torch

from tinymem.model.attention import CausalSelfAttention


def test_attention_preserves_shape() -> None:
    attention = CausalSelfAttention(
        d_model=16,
        n_heads=4,
        max_position_embeddings=32,
    )
    inputs = torch.randn(2, 7, 16)

    outputs = attention(inputs)

    assert outputs.shape == inputs.shape


@pytest.mark.parametrize(
    ("d_model", "n_heads"),
    [(0, 2), (16, 0), (15, 4), (16, True), (True, 4)],
)
def test_attention_rejects_invalid_dimensions(d_model: object, n_heads: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_position_embeddings=32,
        )


@pytest.mark.parametrize("max_position_embeddings", [0, -1, True, 32.0])
def test_attention_rejects_invalid_max_positions(
    max_position_embeddings: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        CausalSelfAttention(
            d_model=16,
            n_heads=4,
            max_position_embeddings=max_position_embeddings,
        )


@pytest.mark.parametrize("dropout", [-0.1, 1.0, True, "0.1"])
def test_attention_rejects_invalid_dropout(dropout: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        CausalSelfAttention(
            d_model=16,
            n_heads=4,
            max_position_embeddings=32,
            dropout=dropout,
        )


def test_attention_rejects_wrong_input_shape() -> None:
    with pytest.raises((TypeError, ValueError)):
        CausalSelfAttention(16, 4, 32)(torch.randn(2, 7))


@pytest.mark.parametrize("inputs", [torch.ones(2, 7, 16, dtype=torch.long), torch.ones(2, 7, 15)])
def test_attention_rejects_invalid_input_tensor(inputs: torch.Tensor) -> None:
    with pytest.raises((TypeError, ValueError)):
        CausalSelfAttention(16, 4, 32)(inputs)


def test_attention_rejects_future_position_overflow() -> None:
    attention = CausalSelfAttention(16, 4, 4)

    with pytest.raises(ValueError, match="position"):
        attention(torch.randn(1, 5, 16))
