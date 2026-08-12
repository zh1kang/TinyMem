import pytest
import torch

from tinymem.model.block import TransformerBlock


def make_block() -> TransformerBlock:
    return TransformerBlock(
        d_model=16,
        n_heads=4,
        d_ff=32,
        max_position_embeddings=32,
    )


@pytest.mark.parametrize(
    ("d_model", "n_heads", "d_ff"),
    [(0, 4, 32), (16, 0, 32), (15, 4, 32), (16, 4, 0), (True, 4, 32)],
)
def test_block_rejects_invalid_dimensions(
    d_model: object,
    n_heads: object,
    d_ff: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        TransformerBlock(d_model, n_heads, d_ff, 32)


@pytest.mark.parametrize("dropout", [-0.1, 1.0, True, "0.1"])
def test_block_rejects_invalid_dropout(dropout: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        TransformerBlock(16, 4, 32, 32, dropout)


def test_block_preserves_shape() -> None:
    block = make_block()
    outputs = block(torch.randn(2, 7, 16))

    assert outputs.shape == (2, 7, 16)


@pytest.mark.parametrize("inputs", [torch.randn(2, 16), torch.ones(2, 7, 16, dtype=torch.long)])
def test_block_rejects_invalid_input(inputs: torch.Tensor) -> None:
    with pytest.raises((TypeError, ValueError)):
        make_block()(inputs)


def test_block_rejects_position_overflow() -> None:
    block = TransformerBlock(16, 4, 32, 4)

    with pytest.raises(ValueError, match="position"):
        block(torch.randn(1, 5, 16))
