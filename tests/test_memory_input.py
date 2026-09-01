import pytest
import torch

from tinymem.model.memory_input import AttentionMemory


def test_attention_memory_accepts_aligned_tensors() -> None:
    memory = AttentionMemory(
        values=torch.randn(2, 3, 4),
        valid=torch.tensor([[True, True, False], [True, False, False]]),
        positions=torch.tensor([[2, 7, -1], [5, -1, -1]]),
    )

    assert memory.slot_count == 3


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("values", torch.zeros(2, 3), ValueError),
        ("values", torch.zeros(2, 3, 4, dtype=torch.long), TypeError),
        ("valid", torch.zeros(2, 2, dtype=torch.bool), ValueError),
        ("valid", torch.zeros(2, 3), TypeError),
        ("positions", torch.zeros(2, 2, dtype=torch.long), ValueError),
        ("positions", torch.zeros(2, 3), TypeError),
    ],
)
def test_attention_memory_rejects_invalid_tensors(
    field: str,
    value: torch.Tensor,
    error: type[Exception],
) -> None:
    arguments = {
        "values": torch.zeros(2, 3, 4),
        "valid": torch.ones(2, 3, dtype=torch.bool),
        "positions": torch.zeros(2, 3, dtype=torch.long),
    }
    arguments[field] = value

    with pytest.raises(error):
        AttentionMemory(**arguments)


def test_attention_memory_allows_negative_sentinels_only_when_invalid() -> None:
    AttentionMemory(
        values=torch.zeros(1, 2, 4),
        valid=torch.tensor([[True, False]]),
        positions=torch.tensor([[3, -1]]),
    )

    with pytest.raises(ValueError, match="valid memory positions"):
        AttentionMemory(
            values=torch.zeros(1, 2, 4),
            valid=torch.tensor([[True, True]]),
            positions=torch.tensor([[3, -1]]),
        )
