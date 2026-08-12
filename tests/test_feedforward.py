import pytest
import torch

from tinymem.model.feedforward import FeedForward


@pytest.mark.parametrize(
    ("d_model", "d_ff"),
    [(0, 32), (16, 0), (-1, 32), (16, -1), (True, 32), (16, True)],
)
def test_feedforward_rejects_invalid_dimensions(d_model: object, d_ff: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        FeedForward(d_model=d_model, d_ff=d_ff)


@pytest.mark.parametrize("dropout", [-0.1, 1.0, True, "0.1"])
def test_feedforward_rejects_invalid_dropout(dropout: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        FeedForward(d_model=16, d_ff=32, dropout=dropout)


def test_feedforward_preserves_shape() -> None:
    outputs = FeedForward(d_model=16, d_ff=32)(torch.randn(2, 7, 16))

    assert outputs.shape == (2, 7, 16)


def test_feedforward_rejects_wrong_input_shape() -> None:
    with pytest.raises((TypeError, ValueError)):
        FeedForward(d_model=16, d_ff=32)(torch.randn(2, 7))
