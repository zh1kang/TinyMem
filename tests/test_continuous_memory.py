import pytest
import torch

from tinymem.memory.continuous import (
    AttentionPoolMemoryCompressor,
    MeanPoolMemoryCompressor,
)


def make_identity_compressor(model_width: int) -> MeanPoolMemoryCompressor:
    compressor = MeanPoolMemoryCompressor(model_width)
    with torch.no_grad():
        compressor.projection.weight.copy_(torch.eye(model_width))
        compressor.projection.bias.zero_()
    return compressor


@pytest.mark.parametrize("expired_tokens", [1, 3, 7])
def test_mean_pool_compressor_has_fixed_output_shape(expired_tokens: int) -> None:
    compressor = MeanPoolMemoryCompressor(model_width=4)
    hidden = torch.randn(2, expired_tokens, 4)
    valid = torch.ones(2, expired_tokens, dtype=torch.bool)

    summary, summary_valid = compressor(hidden, valid)

    assert summary.shape == (2, 1, 4)
    assert summary_valid.shape == (2, 1)


def test_mean_pool_compressor_averages_only_valid_hidden_states() -> None:
    compressor = make_identity_compressor(model_width=2)
    hidden = torch.tensor(
        [
            [[2.0, 4.0], [100.0, 100.0], [6.0, 8.0]],
            [[3.0, 9.0], [5.0, 7.0], [9.0, 3.0]],
        ]
    )
    valid = torch.tensor(
        [
            [True, False, True],
            [False, True, False],
        ]
    )

    summary, summary_valid = compressor(hidden, valid)

    expected = torch.tensor([[[4.0, 6.0]], [[5.0, 7.0]]])
    torch.testing.assert_close(summary, expected)
    assert summary_valid.all()


def test_mean_pool_compressor_zeros_rows_without_valid_tokens() -> None:
    compressor = MeanPoolMemoryCompressor(model_width=3)
    hidden = torch.randn(2, 4, 3)
    valid = torch.tensor(
        [
            [False, False, False, False],
            [True, False, False, False],
        ]
    )

    summary, summary_valid = compressor(hidden, valid)

    assert torch.equal(summary_valid, torch.tensor([[False], [True]]))
    assert torch.equal(summary[0], torch.zeros_like(summary[0]))


def test_mean_pool_compressor_backpropagates_only_through_valid_states() -> None:
    compressor = make_identity_compressor(model_width=2)
    hidden = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]],
        requires_grad=True,
    )
    valid = torch.tensor([[True, False, True]])

    summary, _ = compressor(hidden, valid)
    summary.sum().backward()

    assert compressor.projection.weight.grad is not None
    assert torch.isfinite(compressor.projection.weight.grad).all()
    assert hidden.grad is not None
    assert (hidden.grad[0, 0] != 0).all()
    assert torch.equal(hidden.grad[0, 1], torch.zeros(2))
    assert (hidden.grad[0, 2] != 0).all()


def test_attention_pool_starts_as_a_masked_mean() -> None:
    compressor = AttentionPoolMemoryCompressor(model_width=2)
    with torch.no_grad():
        compressor.projection.weight.copy_(torch.eye(2))
        compressor.projection.bias.zero_()
    hidden = torch.tensor([[[2.0, 4.0], [100.0, 100.0], [6.0, 8.0]]])
    valid = torch.tensor([[True, False, True]])

    summary, summary_valid = compressor(hidden, valid)

    torch.testing.assert_close(summary, torch.tensor([[[4.0, 6.0]]]))
    assert summary_valid.all()


def test_attention_pool_learned_query_can_select_a_valid_state() -> None:
    compressor = AttentionPoolMemoryCompressor(model_width=2)
    with torch.no_grad():
        compressor.query.copy_(torch.tensor([4.0, 0.0]))
        compressor.projection.weight.copy_(torch.eye(2))
        compressor.projection.bias.zero_()
    hidden = torch.tensor([[[4.0, 0.0], [0.0, 3.0], [100.0, 100.0]]])
    valid = torch.tensor([[True, True, False]])

    summary, _ = compressor(hidden, valid)

    assert summary[0, 0, 0] > 3.9
    assert summary[0, 0, 1] < 0.1


def test_attention_pool_backpropagates_to_query_and_valid_states() -> None:
    compressor = AttentionPoolMemoryCompressor(model_width=2)
    hidden = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]],
        requires_grad=True,
    )
    valid = torch.tensor([[True, False, True]])

    summary, _ = compressor(hidden, valid)
    summary.square().sum().backward()

    assert compressor.query.grad is not None
    assert torch.count_nonzero(compressor.query.grad) > 0
    assert hidden.grad is not None
    assert torch.equal(hidden.grad[0, 1], torch.zeros(2))


def test_attention_pool_zeros_rows_without_valid_tokens() -> None:
    compressor = AttentionPoolMemoryCompressor(model_width=3)
    hidden = torch.randn(2, 4, 3)
    valid = torch.tensor(
        [[False, False, False, False], [True, False, False, False]]
    )

    summary, summary_valid = compressor(hidden, valid)

    assert torch.equal(summary_valid, torch.tensor([[False], [True]]))
    assert torch.equal(summary[0], torch.zeros_like(summary[0]))
    assert torch.isfinite(summary).all()


@pytest.mark.parametrize(
    ("hidden", "valid", "error"),
    [
        ([[[1.0]]], torch.tensor([[True]]), TypeError),
        (torch.ones(1, 2), torch.tensor([[True, True]]), ValueError),
        (
            torch.ones(1, 2, 3, dtype=torch.long),
            torch.ones(1, 2, dtype=torch.bool),
            TypeError,
        ),
        (torch.ones(1, 2, 3), [[True, True]], TypeError),
        (torch.ones(1, 2, 3), torch.ones(1, 3, dtype=torch.bool), ValueError),
        (torch.ones(1, 2, 3), torch.ones(1, 2), TypeError),
    ],
)
def test_mean_pool_compressor_rejects_invalid_inputs(
    hidden: object,
    valid: object,
    error: type[Exception],
) -> None:
    compressor = MeanPoolMemoryCompressor(model_width=3)

    with pytest.raises(error):
        compressor(hidden, valid)


@pytest.mark.parametrize("model_width", [True, 0, -1, 3.5, "3"])
def test_mean_pool_compressor_rejects_invalid_width(model_width: object) -> None:
    expected = (
        ValueError
        if isinstance(model_width, int) and not isinstance(model_width, bool)
        else TypeError
    )

    with pytest.raises(expected):
        MeanPoolMemoryCompressor(model_width)
