import pytest
import torch

from tinymem.training.losses import next_token_cross_entropy


def test_next_token_loss_shifts_logits_and_targets() -> None:
    logits = torch.tensor(
        [
            [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        ]
    )
    targets = torch.tensor([[2, 1, 0]])

    loss = next_token_cross_entropy(logits, targets)
    expected = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 3),
        targets[:, 1:].reshape(-1),
    )

    assert torch.allclose(loss, expected)


def test_next_token_loss_ignores_padding_targets() -> None:
    logits = torch.zeros(1, 4, 3)
    targets = torch.tensor([[1, 2, -100, -100]])

    loss = next_token_cross_entropy(logits, targets)
    expected = torch.log(torch.tensor(3.0))

    assert torch.allclose(loss, expected)


@pytest.mark.parametrize(
    "logits, targets",
    [
        (torch.zeros(1, 1, 3), torch.zeros(1, 1, dtype=torch.long)),
        (torch.zeros(1, 3), torch.zeros(1, 3, dtype=torch.long)),
        (torch.zeros(1, 3, 3), torch.zeros(1, 2, dtype=torch.long)),
        (torch.zeros(1, 3, 3), torch.zeros(1, 3)),
    ],
)
def test_next_token_loss_rejects_invalid_shapes_or_dtypes(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        next_token_cross_entropy(logits, targets)


def test_next_token_loss_rejects_out_of_range_targets() -> None:
    logits = torch.zeros(1, 3, 3)
    targets = torch.tensor([[0, 1, 3]])

    with pytest.raises(ValueError, match="target IDs"):
        next_token_cross_entropy(logits, targets)


def test_next_token_loss_requires_a_nonignored_target() -> None:
    logits = torch.zeros(1, 3, 3)
    targets = torch.full((1, 3), -100)

    with pytest.raises(ValueError, match="non-ignored"):
        next_token_cross_entropy(logits, targets)
