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


def test_rope_shared_position_ids_match_contiguous_offset() -> None:
    rope = RotaryEmbedding(head_dim=8, max_position_embeddings=16)
    inputs = torch.randn(2, 3, 4, 8)

    expected = rope(inputs, position_offset=5)
    actual = rope(inputs, position_ids=torch.tensor([5, 6, 7, 8]))

    torch.testing.assert_close(actual, expected)


def test_rope_supports_different_positions_for_each_batch_row() -> None:
    rope = RotaryEmbedding(head_dim=8, max_position_embeddings=16)
    inputs = torch.randn(2, 3, 2, 8)
    position_ids = torch.tensor([[2, 7], [11, 4]])

    actual = rope(inputs, position_ids=position_ids)

    expected_first = rope(inputs[0:1], position_ids=position_ids[0])
    expected_second = rope(inputs[1:2], position_ids=position_ids[1])
    expected = torch.cat((expected_first, expected_second), dim=0)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "position_ids",
    [
        [1, 2],
        torch.tensor([1.0, 2.0]),
        torch.tensor(1),
        torch.tensor([[[1, 2]]]),
        torch.tensor([1]),
        torch.tensor([[1, 2], [3, 4], [5, 6]]),
        torch.tensor([-1, 2]),
    ],
)
def test_rope_rejects_invalid_explicit_position_ids(position_ids: object) -> None:
    rope = RotaryEmbedding(head_dim=8, max_position_embeddings=16)

    with pytest.raises((TypeError, ValueError)):
        rope(torch.randn(2, 1, 2, 8), position_ids=position_ids)


def test_rope_rejects_offset_with_explicit_position_ids() -> None:
    rope = RotaryEmbedding(head_dim=8, max_position_embeddings=16)

    with pytest.raises(ValueError, match="position_offset must be zero"):
        rope(
            torch.randn(1, 1, 2, 8),
            position_offset=3,
            position_ids=torch.tensor([1, 2]),
        )


@pytest.mark.parametrize("head_dim", [0, -2, 3, True, 4.0])
def test_rope_rejects_invalid_head_dim(head_dim: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        RotaryEmbedding(head_dim=head_dim, max_position_embeddings=16)


def test_rope_rejects_sequence_longer_than_cache() -> None:
    rope = RotaryEmbedding(head_dim=8, max_position_embeddings=4)

    with pytest.raises(ValueError, match="maximum position"):
        rope(torch.randn(1, 1, 5, 8))
