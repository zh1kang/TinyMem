import pytest
import torch

from tinymem.memory.write_gate import TokenSegmentWriteGate


def test_token_write_gate_returns_one_logit_per_row() -> None:
    gate = TokenSegmentWriteGate(4)
    embeddings = torch.randn(2, 5, 4)
    valid = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, False, False, False],
        ]
    )

    logits = gate(embeddings, valid)

    assert logits.shape == (2, 1)
    assert torch.isfinite(logits).all()
    assert torch.equal(logits, torch.ones_like(logits))


def test_token_write_gate_ignores_invalid_embedding_values() -> None:
    torch.manual_seed(37)
    gate = TokenSegmentWriteGate(4)
    embeddings = torch.randn(1, 5, 4)
    valid = torch.tensor([[True, True, False, False, False]])
    changed = embeddings.clone()
    changed[:, 2:] = 1_000

    expected = gate(embeddings, valid)
    actual = gate(changed, valid)

    torch.testing.assert_close(actual, expected)


def test_token_write_gate_handles_an_all_invalid_row() -> None:
    gate = TokenSegmentWriteGate(4)

    logits = gate(
        torch.randn(1, 3, 4),
        torch.zeros(1, 3, dtype=torch.bool),
    )

    assert logits.shape == (1, 1)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("model_width", [True, 0, -1, 1.5, "4"])
def test_token_write_gate_rejects_invalid_width(model_width: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        TokenSegmentWriteGate(model_width)  # type: ignore[arg-type]
