import pytest
import torch

from tinymem.model.rope import RotaryEmbedding


def test_rope_uses_standard_inverse_frequency_schedule() -> None:
    rope = RotaryEmbedding(
        head_dim=8,
        max_position_embeddings=16,
        base=10_000.0,
    )

    expected = torch.tensor([1.0, 0.1, 0.01, 0.001])

    torch.testing.assert_close(rope.inv_freq, expected)


def test_rope_preserves_shape_and_dtype() -> None:
    rope = RotaryEmbedding(head_dim=8, max_position_embeddings=16)
    inputs = torch.randn(2, 4, 6, 8)

    outputs = rope(inputs)

    assert outputs.shape == inputs.shape
    assert outputs.dtype == inputs.dtype


def test_rope_preserves_vector_norm() -> None:
    rope = RotaryEmbedding(head_dim=8, max_position_embeddings=16)
    inputs = torch.randn(2, 4, 6, 8)

    outputs = rope(inputs)

    assert torch.allclose(
        outputs.square().sum(dim=-1),
        inputs.square().sum(dim=-1),
        atol=1e-5,
    )


def test_rope_position_offset_changes_rotation() -> None:
    rope = RotaryEmbedding(head_dim=8, max_position_embeddings=16)
    inputs = torch.randn(1, 1, 3, 8)

    first = rope(inputs, position_offset=0)
    shifted = rope(inputs, position_offset=1)

    assert not torch.allclose(first, shifted)


@pytest.mark.parametrize("head_dim", [0, -2, 3, True, 4.0])
def test_rope_rejects_invalid_head_dim(head_dim: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        RotaryEmbedding(head_dim=head_dim, max_position_embeddings=16)


def test_rope_rejects_sequence_longer_than_cache() -> None:
    rope = RotaryEmbedding(head_dim=8, max_position_embeddings=4)

    with pytest.raises(ValueError, match="maximum position"):
        rope(torch.randn(1, 1, 5, 8))
