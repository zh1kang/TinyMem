import pytest
import torch

from tinymem.memory.recent_tokens import RecentTokenMemory
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


def test_recent_memory_keeps_newest_items_in_chronological_order() -> None:
    policy = RecentTokenMemory(capacity=3)
    state = policy.initialize(batch_size=1, model_width=2)
    candidates = MemoryState.empty(batch_size=1, capacity=3, model_width=2)
    fill_row(state, 0, [1, 5])
    fill_row(candidates, 0, [3, 6, 2])

    result = policy.update(state, candidates)

    assert result.token_ids is not None
    assert torch.equal(result.positions, torch.tensor([[3, 5, 6]]))
    assert torch.equal(result.token_ids, torch.tensor([[3, 5, 6]]))
    assert result.valid.all()


def test_recent_memory_handles_each_batch_mask_independently() -> None:
    policy = RecentTokenMemory(capacity=2)
    state = policy.initialize(batch_size=2, model_width=2)
    candidates = MemoryState.empty(batch_size=2, capacity=3, model_width=2)
    fill_row(candidates, 0, [1, 3])
    fill_row(candidates, 1, [4])

    result = policy.update(state, candidates)

    assert torch.equal(result.positions, torch.tensor([[1, 3], [4, -1]]))
    assert torch.equal(
        result.valid,
        torch.tensor([[True, True], [True, False]]),
    )


def test_recent_memory_does_not_modify_inputs() -> None:
    policy = RecentTokenMemory(capacity=2)
    state = policy.initialize(batch_size=1, model_width=2)
    candidates = MemoryState.empty(batch_size=1, capacity=2, model_width=2)
    fill_row(state, 0, [1])
    fill_row(candidates, 0, [2])
    original_state = state.positions.clone()
    original_candidates = candidates.positions.clone()

    policy.update(state, candidates)

    assert torch.equal(state.positions, original_state)
    assert torch.equal(candidates.positions, original_candidates)


def test_recent_memory_preserves_optional_scores_for_budget_accounting() -> None:
    policy = RecentTokenMemory(capacity=2)
    state = policy.initialize(batch_size=1, model_width=1, with_scores=True)
    candidates = MemoryState.empty(
        batch_size=1,
        capacity=2,
        model_width=1,
        with_scores=True,
    )
    fill_row(candidates, 0, [2, 4])
    assert candidates.scores is not None
    candidates.scores[0] = torch.tensor([0.25, 0.75])

    result = policy.update(state, candidates)

    assert result.scores is not None
    torch.testing.assert_close(result.scores, torch.tensor([[0.25, 0.75]]))


def test_recent_memory_requires_matching_score_availability() -> None:
    policy = RecentTokenMemory(capacity=1)
    state = policy.initialize(batch_size=1, model_width=1, with_scores=True)
    candidates = MemoryState.empty(batch_size=1, capacity=1, model_width=1)

    with pytest.raises(ValueError, match="score availability"):
        policy.update(state, candidates)


def test_recent_memory_requires_raw_token_ids() -> None:
    policy = RecentTokenMemory(capacity=2)
    state = policy.initialize(batch_size=1, model_width=2)
    candidates = MemoryState(
        values=torch.zeros(1, 2, 2),
        valid=torch.zeros(1, 2, dtype=torch.bool),
        positions=torch.full((1, 2), -1, dtype=torch.long),
        token_ids=None,
    )

    with pytest.raises(ValueError, match="token_ids"):
        policy.update(state, candidates)
