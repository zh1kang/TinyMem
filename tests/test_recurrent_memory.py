import pytest
import torch

from tinymem.memory.recurrent_memory import RecurrentMemoryBank


def test_recurrent_bank_evicts_oldest_slot_and_appends_summary() -> None:
    bank = RecurrentMemoryBank(capacity=3, model_width=2)
    memory = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])
    memory_valid = torch.tensor([[True, True, True]])
    summary = torch.tensor([[[4.0, 4.0]]])
    summary_valid = torch.tensor([[True]])

    next_memory, next_valid = bank(
        memory,
        memory_valid,
        summary,
        summary_valid,
    )

    expected_memory = torch.tensor([[[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]])
    assert torch.equal(next_memory, expected_memory)
    assert torch.equal(next_valid, torch.tensor([[True, True, True]]))


def test_recurrent_bank_appends_multiple_summaries_together() -> None:
    bank = RecurrentMemoryBank(capacity=4, model_width=1)
    memory = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    memory_valid = torch.ones(1, 4, dtype=torch.bool)
    summary = torch.tensor([[[5.0], [6.0]]])
    summary_valid = torch.ones(1, 2, dtype=torch.bool)

    next_memory, next_valid = bank(
        memory,
        memory_valid,
        summary,
        summary_valid,
    )

    assert torch.equal(
        next_memory,
        torch.tensor([[[3.0], [4.0], [5.0], [6.0]]]),
    )
    assert next_valid.all()


def test_recurrent_bank_keeps_rows_with_invalid_summaries_unchanged() -> None:
    bank = RecurrentMemoryBank(capacity=3, model_width=1)
    memory = torch.tensor(
        [
            [[1.0], [2.0], [3.0]],
            [[10.0], [20.0], [30.0]],
        ]
    )
    memory_valid = torch.tensor(
        [
            [True, True, True],
            [False, True, True],
        ]
    )
    summary = torch.tensor([[[4.0]], [[40.0]]])
    summary_valid = torch.tensor([[True], [False]])

    next_memory, next_valid = bank(
        memory,
        memory_valid,
        summary,
        summary_valid,
    )

    assert torch.equal(next_memory[0], torch.tensor([[2.0], [3.0], [4.0]]))
    assert torch.equal(next_memory[1], memory[1])
    assert torch.equal(next_valid[0], torch.tensor([True, True, True]))
    assert torch.equal(next_valid[1], memory_valid[1])


@pytest.mark.parametrize("capacity", [1, 3, 5])
def test_recurrent_bank_preserves_fixed_shape(capacity: int) -> None:
    bank = RecurrentMemoryBank(capacity=capacity, model_width=4)
    memory = torch.zeros(2, capacity, 4)
    memory_valid = torch.zeros(2, capacity, dtype=torch.bool)
    summary = torch.ones(2, 1, 4)
    summary_valid = torch.ones(2, 1, dtype=torch.bool)

    next_memory, next_valid = bank(
        memory,
        memory_valid,
        summary,
        summary_valid,
    )

    assert next_memory.shape == memory.shape
    assert next_valid.shape == memory_valid.shape


def test_recurrent_bank_does_not_modify_inputs() -> None:
    bank = RecurrentMemoryBank(capacity=2, model_width=2)
    memory = torch.randn(2, 2, 2)
    memory_valid = torch.tensor([[True, True], [False, True]])
    summary = torch.randn(2, 1, 2)
    summary_valid = torch.tensor([[True], [False]])
    original_memory = memory.clone()
    original_valid = memory_valid.clone()
    original_summary = summary.clone()
    original_summary_valid = summary_valid.clone()

    bank(memory, memory_valid, summary, summary_valid)

    assert torch.equal(memory, original_memory)
    assert torch.equal(memory_valid, original_valid)
    assert torch.equal(summary, original_summary)
    assert torch.equal(summary_valid, original_summary_valid)


def test_recurrent_bank_preserves_gradient_paths_for_selected_values() -> None:
    bank = RecurrentMemoryBank(capacity=3, model_width=1)
    memory = torch.tensor(
        [
            [[1.0], [2.0], [3.0]],
            [[10.0], [20.0], [30.0]],
        ],
        requires_grad=True,
    )
    memory_valid = torch.ones(2, 3, dtype=torch.bool)
    summary = torch.tensor([[[4.0]], [[40.0]]], requires_grad=True)
    summary_valid = torch.tensor([[True], [False]])

    next_memory, _ = bank(memory, memory_valid, summary, summary_valid)
    next_memory.sum().backward()

    assert memory.grad is not None
    assert summary.grad is not None
    assert torch.equal(memory.grad[0], torch.tensor([[0.0], [1.0], [1.0]]))
    assert torch.equal(summary.grad[0], torch.tensor([[1.0]]))
    assert torch.equal(memory.grad[1], torch.ones(3, 1))
    assert torch.equal(summary.grad[1], torch.zeros(1, 1))


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ({"memory": [[[1.0]]]}, TypeError),
        ({"memory": torch.ones(1, 1)}, ValueError),
        ({"memory": torch.ones(1, 2, 3)}, ValueError),
        ({"memory": torch.ones(1, 2, 2, dtype=torch.long)}, TypeError),
        ({"memory_valid": torch.ones(1, 3, dtype=torch.bool)}, ValueError),
        ({"memory_valid": torch.ones(1, 2)}, TypeError),
        ({"summary": torch.ones(1, 2, 2)}, ValueError),
        ({"summary": torch.ones(1, 0, 2)}, ValueError),
        ({"summary": torch.ones(1, 1, 2, dtype=torch.long)}, TypeError),
        ({"summary_valid": torch.ones(1, dtype=torch.bool)}, ValueError),
        ({"summary_valid": torch.ones(1, 1)}, TypeError),
        (
            {
                "summary": torch.ones(1, 2, 2),
                "summary_valid": torch.tensor([[True, False]]),
            },
            ValueError,
        ),
    ],
)
def test_recurrent_bank_rejects_invalid_inputs(
    arguments: dict[str, object],
    error: type[Exception],
) -> None:
    bank = RecurrentMemoryBank(capacity=2, model_width=2)
    inputs: dict[str, object] = {
        "memory": torch.zeros(1, 2, 2),
        "memory_valid": torch.zeros(1, 2, dtype=torch.bool),
        "summary": torch.zeros(1, 1, 2),
        "summary_valid": torch.ones(1, 1, dtype=torch.bool),
    }
    inputs.update(arguments)

    with pytest.raises(error):
        bank(**inputs)


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ({"capacity": True}, TypeError),
        ({"capacity": 0}, ValueError),
        ({"model_width": 2.5}, TypeError),
        ({"model_width": -1}, ValueError),
    ],
)
def test_recurrent_bank_rejects_invalid_dimensions(
    arguments: dict[str, object],
    error: type[Exception],
) -> None:
    dimensions: dict[str, object] = {"capacity": 2, "model_width": 2}
    dimensions.update(arguments)

    with pytest.raises(error):
        RecurrentMemoryBank(**dimensions)
