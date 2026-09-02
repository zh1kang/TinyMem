import pytest
import torch

from tinymem.model.multi_token_prediction import MultiTokenPredictionHeads
from tinymem.training.multi_token_prediction import multi_token_cross_entropy


def test_heads_project_each_horizon_independently() -> None:
    heads = MultiTokenPredictionHeads(4, 7, (2, 3, 4))
    output = heads(torch.randn(2, 5, 4))

    assert tuple(output) == (2, 3, 4)
    assert all(logits.shape == (2, 5, 7) for logits in output.values())
    assert len({head.weight.data_ptr() for head in heads.heads}) == 3


@pytest.mark.parametrize("horizon", (2, 3, 4))
def test_loss_uses_exact_future_offset(horizon: int) -> None:
    target_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
    token_valid = torch.ones_like(target_ids, dtype=torch.bool)
    logits = torch.full((1, 6, 6), -20.0)
    for source in range(6 - horizon):
        logits[0, source, target_ids[0, source + horizon]] = 20.0

    result = multi_token_cross_entropy(
        {horizon: logits},
        target_ids,
        token_valid,
    )

    assert result.total.item() < 1e-6


def test_loss_excludes_right_padding_from_each_alignment() -> None:
    target_ids = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 0, 0]])
    token_valid = torch.tensor(
        [[True, True, True, True, True], [True, True, True, False, False]]
    )
    logits = torch.full((2, 5, 5), -20.0)
    for row in range(2):
        for source in range(3):
            target = source + 2
            if token_valid[row, source] and token_valid[row, target]:
                logits[row, source, target_ids[row, target]] = 20.0

    result = multi_token_cross_entropy({2: logits}, target_ids, token_valid)

    assert result.total.item() < 1e-6


def test_loss_backpropagates_through_every_head() -> None:
    heads = MultiTokenPredictionHeads(4, 8, (2, 3, 4))
    hidden = torch.randn(2, 7, 4, requires_grad=True)
    target_ids = torch.randint(0, 8, (2, 7))
    token_valid = torch.ones_like(target_ids, dtype=torch.bool)

    result = multi_token_cross_entropy(heads(hidden), target_ids, token_valid)
    result.total.backward()

    assert hidden.grad is not None
    assert all(head.weight.grad is not None for head in heads.heads)
