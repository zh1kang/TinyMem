import pytest
import torch

from tinymem.memory.oracle import OracleMemory
from tinymem.memory.state import MemoryState


def fill_row(
    state: MemoryState,
    batch_index: int,
    positions: list[int],
) -> None:
    count = len(positions)
    state.valid[batch_index, :count] = True
    state.positions[batch_index, :count] = torch.tensor(positions)
    assert state.token_ids is not None
    state.token_ids[batch_index, :count] = torch.tensor(positions)
    values = torch.tensor(positions, dtype=state.values.dtype).unsqueeze(-1)
    state.values[batch_index, :count] = values.repeat(1, state.model_width)
    if state.scores is not None:
        state.scores[batch_index, :count] = torch.tensor(
            positions,
            dtype=state.scores.dtype,
        )


def test_oracle_keeps_only_target_positions_in_chronological_order() -> None:
    policy = OracleMemory(
        capacity=3,
        target_positions=torch.tensor([[2, 8, 5]]),
    )
    state = policy.initialize(
        batch_size=1,
        model_width=2,
        with_scores=True,
    )
    candidates = MemoryState.empty(
        batch_size=1,
        capacity=5,
        model_width=2,
        with_scores=True,
    )
    fill_row(candidates, 0, [1, 8, 2, 4, 5])

    result = policy.update(state, candidates)

    assert torch.equal(result.positions, torch.tensor([[2, 5, 8]]))
    assert torch.equal(result.token_ids, result.positions)
    assert torch.equal(result.values[0, :, 0].to(torch.long), result.positions[0])
    assert result.scores is not None
    assert torch.equal(result.scores, result.positions.to(torch.float32))
    assert result.valid.all()


def test_oracle_accumulates_targets_as_they_expire() -> None:
    policy = OracleMemory(
        capacity=2,
        target_positions=torch.tensor([[3, 9]]),
    )
    state = policy.initialize(batch_size=1, model_width=1)
    first = MemoryState.empty(batch_size=1, capacity=2, model_width=1)
    fill_row(first, 0, [3, 4])
    second = MemoryState.empty(batch_size=1, capacity=2, model_width=1)
    fill_row(second, 0, [8, 9])

    state = policy.update(state, first)
    result = policy.update(state, second)

    assert torch.equal(result.positions, torch.tensor([[3, 9]]))
    assert result.valid.all()


def test_oracle_handles_batch_rows_and_padding_independently() -> None:
    policy = OracleMemory(
        capacity=2,
        target_positions=torch.tensor([[2, 4], [7, -1], [-1, -1]]),
    )
    state = policy.initialize(batch_size=3, model_width=1)
    candidates = MemoryState.empty(batch_size=3, capacity=3, model_width=1)
    fill_row(candidates, 0, [1, 2, 4])
    fill_row(candidates, 1, [7, 8])
    fill_row(candidates, 2, [9])

    result = policy.update(state, candidates)

    assert torch.equal(
        result.positions,
        torch.tensor([[2, 4], [7, -1], [-1, -1]]),
    )
    assert torch.equal(
        result.valid,
        torch.tensor([[True, True], [True, False], [False, False]]),
    )


def test_oracle_does_not_modify_inputs_or_expose_target_storage() -> None:
    targets = torch.tensor([[2]])
    policy = OracleMemory(capacity=1, target_positions=targets)
    state = policy.initialize(batch_size=1, model_width=1)
    candidates = MemoryState.empty(batch_size=1, capacity=1, model_width=1)
    fill_row(candidates, 0, [2])
    original_candidates = candidates.positions.clone()

    policy.update(state, candidates)
    targets.fill_(9)
    exposed = policy.target_positions
    exposed.fill_(7)

    assert torch.equal(candidates.positions, original_candidates)
    assert torch.equal(policy.target_positions, torch.tensor([[2]]))


@pytest.mark.parametrize(
    ("target_positions", "error"),
    [
        ([[1]], TypeError),
        (torch.tensor([1]), ValueError),
        (torch.tensor([[1.0]]), TypeError),
        (torch.tensor([[-2]]), ValueError),
        (torch.tensor([[1, 1]]), ValueError),
        (torch.tensor([[1, 2]]), ValueError),
    ],
)
def test_oracle_rejects_invalid_target_positions(
    target_positions: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        OracleMemory(capacity=1, target_positions=target_positions)


def test_oracle_requires_matching_batch_size() -> None:
    policy = OracleMemory(capacity=1, target_positions=torch.tensor([[1], [2]]))
    state = policy.initialize(batch_size=1, model_width=1)
    candidates = MemoryState.empty(batch_size=1, capacity=1, model_width=1)

    with pytest.raises(ValueError, match="batch size"):
        policy.update(state, candidates)


def test_oracle_requires_raw_token_ids() -> None:
    policy = OracleMemory(capacity=1, target_positions=torch.tensor([[1]]))
    state = policy.initialize(batch_size=1, model_width=1)
    candidates = MemoryState(
        values=torch.zeros(1, 1, 1),
        valid=torch.zeros(1, 1, dtype=torch.bool),
        positions=torch.full((1, 1), -1, dtype=torch.long),
        token_ids=None,
    )

    with pytest.raises(ValueError, match="token_ids"):
        policy.update(state, candidates)


def test_oracle_requires_matching_score_availability() -> None:
    policy = OracleMemory(capacity=1, target_positions=torch.tensor([[1]]))
    state = policy.initialize(batch_size=1, model_width=1, with_scores=True)
    candidates = MemoryState.empty(batch_size=1, capacity=1, model_width=1)

    with pytest.raises(ValueError, match="score availability"):
        policy.update(state, candidates)
