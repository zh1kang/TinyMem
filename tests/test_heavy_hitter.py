import pytest
import torch

from tinymem.memory.heavy_hitter import HeavyHitterMemory
from tinymem.memory.state import MemoryState


def fill_row(
    state: MemoryState,
    batch_index: int,
    positions: list[int],
    scores: list[float],
) -> None:
    count = len(positions)
    state.valid[batch_index, :count] = True
    state.positions[batch_index, :count] = torch.tensor(positions)
    assert state.token_ids is not None
    state.token_ids[batch_index, :count] = torch.tensor(positions)
    assert state.scores is not None
    state.scores[batch_index, :count] = torch.tensor(
        scores,
        dtype=state.scores.dtype,
    )
    values = torch.tensor(positions, dtype=state.values.dtype).unsqueeze(-1)
    state.values[batch_index, :count] = values.repeat(1, state.model_width)


def test_heavy_hitter_combines_high_scores_with_newest_remaining_entries() -> None:
    policy = HeavyHitterMemory(capacity=3, recent_slots=2)
    state = policy.initialize(batch_size=1, model_width=2, with_scores=True)
    candidates = MemoryState.empty(
        batch_size=1,
        capacity=4,
        model_width=2,
        with_scores=True,
    )
    fill_row(candidates, 0, [1, 9, 10, 11], [0.1, 0.9, 0.2, 0.3])

    result = policy.update(state, candidates)

    assert torch.equal(result.positions, torch.tensor([[9, 10, 11]]))
    assert torch.equal(result.token_ids, result.positions)
    assert torch.equal(result.values[0, :, 0].to(torch.long), result.positions[0])
    assert result.scores is not None
    torch.testing.assert_close(result.scores, torch.tensor([[0.9, 0.2, 0.3]]))
    assert result.valid.all()


def test_heavy_hitter_breaks_score_ties_by_recency() -> None:
    policy = HeavyHitterMemory(capacity=3, recent_slots=2)
    state = policy.initialize(batch_size=1, model_width=1, with_scores=True)
    candidates = MemoryState.empty(
        batch_size=1,
        capacity=4,
        model_width=1,
        with_scores=True,
    )
    fill_row(candidates, 0, [1, 9, 10, 11], [1.0, 1.0, 0.5, 0.4])

    result = policy.update(state, candidates)

    assert torch.equal(result.positions, torch.tensor([[9, 10, 11]]))


def test_heavy_hitter_handles_no_entries_remaining_after_heavy_selection() -> None:
    policy = HeavyHitterMemory(capacity=4, recent_slots=2)
    state = policy.initialize(batch_size=1, model_width=1, with_scores=True)
    candidates = MemoryState.empty(
        batch_size=1,
        capacity=2,
        model_width=1,
        with_scores=True,
    )
    fill_row(candidates, 0, [1, 2], [0.9, 0.8])

    result = policy.update(state, candidates)

    assert torch.equal(result.valid, torch.tensor([[True, True, False, False]]))
    assert torch.equal(result.positions, torch.tensor([[1, 2, -1, -1]]))
    assert result.scores is not None
    assert torch.isneginf(result.scores[0, 2:]).all()


def test_heavy_hitter_handles_batch_rows_independently() -> None:
    policy = HeavyHitterMemory(capacity=3, recent_slots=1)
    state = policy.initialize(batch_size=3, model_width=1, with_scores=True)
    candidates = MemoryState.empty(
        batch_size=3,
        capacity=4,
        model_width=1,
        with_scores=True,
    )
    fill_row(candidates, 0, [1, 2, 3, 4], [0.9, 0.8, 0.1, 0.2])
    fill_row(candidates, 1, [7], [0.5])

    result = policy.update(state, candidates)

    assert torch.equal(
        result.positions,
        torch.tensor([[1, 2, 4], [7, -1, -1], [-1, -1, -1]]),
    )
    assert torch.equal(
        result.valid,
        torch.tensor(
            [[True, True, True], [True, False, False], [False, False, False]]
        ),
    )


def test_heavy_hitter_does_not_modify_inputs() -> None:
    policy = HeavyHitterMemory(capacity=2, recent_slots=1)
    state = policy.initialize(batch_size=1, model_width=1, with_scores=True)
    candidates = MemoryState.empty(
        batch_size=1,
        capacity=2,
        model_width=1,
        with_scores=True,
    )
    fill_row(state, 0, [1], [0.5])
    fill_row(candidates, 0, [2], [0.8])
    original_state = (
        state.values.clone(),
        state.valid.clone(),
        state.positions.clone(),
        state.token_ids.clone(),
        state.scores.clone(),
    )
    original_candidates = (
        candidates.values.clone(),
        candidates.valid.clone(),
        candidates.positions.clone(),
        candidates.token_ids.clone(),
        candidates.scores.clone(),
    )

    policy.update(state, candidates)

    for tensor, original in zip(
        (state.values, state.valid, state.positions, state.token_ids, state.scores),
        original_state,
        strict=True,
    ):
        assert torch.equal(tensor, original)
    for tensor, original in zip(
        (
            candidates.values,
            candidates.valid,
            candidates.positions,
            candidates.token_ids,
            candidates.scores,
        ),
        original_candidates,
        strict=True,
    ):
        assert torch.equal(tensor, original)


@pytest.mark.parametrize("recent_slots", [True, 0, 4])
def test_heavy_hitter_rejects_invalid_recent_slots(recent_slots: int) -> None:
    expected_error = TypeError if isinstance(recent_slots, bool) else ValueError

    with pytest.raises(expected_error):
        HeavyHitterMemory(capacity=4, recent_slots=recent_slots)
