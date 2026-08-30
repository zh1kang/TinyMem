import pytest
import torch

from tinymem.memory.importance import ExtractiveImportanceMemory
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


def test_importance_keeps_highest_scores_and_breaks_ties_by_recency() -> None:
    policy = ExtractiveImportanceMemory(capacity=3)
    state = policy.initialize(batch_size=1, model_width=2, with_scores=True)
    candidates = MemoryState.empty(
        batch_size=1,
        capacity=3,
        model_width=2,
        with_scores=True,
    )
    fill_row(state, 0, [2, 7], [0.9, 0.4])
    fill_row(candidates, 0, [5, 9, 11], [0.8, 0.4, 0.2])

    result = policy.update(state, candidates)

    assert result.token_ids is not None
    assert result.scores is not None
    assert torch.equal(result.positions, torch.tensor([[2, 5, 9]]))
    assert torch.equal(result.token_ids, result.positions)
    assert torch.equal(result.values[0, :, 0].to(torch.long), result.positions[0])
    torch.testing.assert_close(
        result.scores,
        torch.tensor([[0.9, 0.8, 0.4]]),
    )
    assert result.valid.all()


def test_importance_handles_batch_rows_and_empty_slots_independently() -> None:
    policy = ExtractiveImportanceMemory(capacity=2)
    state = policy.initialize(batch_size=3, model_width=1, with_scores=True)
    candidates = MemoryState.empty(
        batch_size=3,
        capacity=3,
        model_width=1,
        with_scores=True,
    )
    fill_row(candidates, 0, [1, 3, 5], [0.1, 0.9, 0.8])
    fill_row(candidates, 1, [4], [0.5])

    result = policy.update(state, candidates)

    assert torch.equal(
        result.positions,
        torch.tensor([[3, 5], [4, -1], [-1, -1]]),
    )
    assert torch.equal(
        result.valid,
        torch.tensor(
            [[True, True], [True, False], [False, False]],
        ),
    )
    assert result.scores is not None
    assert torch.isneginf(result.scores[1, 1])
    assert torch.isneginf(result.scores[2]).all()


def test_importance_does_not_modify_inputs() -> None:
    policy = ExtractiveImportanceMemory(capacity=2)
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


@pytest.mark.parametrize("missing", ["state", "candidates"])
def test_importance_requires_scores(missing: str) -> None:
    policy = ExtractiveImportanceMemory(capacity=2)
    state = policy.initialize(
        batch_size=1,
        model_width=1,
        with_scores=missing != "state",
    )
    candidates = MemoryState.empty(
        batch_size=1,
        capacity=2,
        model_width=1,
        with_scores=missing != "candidates",
    )

    with pytest.raises(ValueError, match="scores"):
        policy.update(state, candidates)


def test_importance_requires_matching_score_dtypes() -> None:
    policy = ExtractiveImportanceMemory(capacity=2)
    state = policy.initialize(batch_size=1, model_width=1, with_scores=True)
    candidates = MemoryState(
        values=torch.zeros(1, 2, 1),
        valid=torch.zeros(1, 2, dtype=torch.bool),
        positions=torch.full((1, 2), -1, dtype=torch.long),
        token_ids=torch.full((1, 2), -1, dtype=torch.long),
        scores=torch.full((1, 2), float("-inf"), dtype=torch.float64),
    )

    with pytest.raises(ValueError, match="same dtype"):
        policy.update(state, candidates)


@pytest.mark.parametrize("invalid_score", [float("inf"), float("nan")])
def test_importance_requires_finite_scores_for_valid_entries(
    invalid_score: float,
) -> None:
    policy = ExtractiveImportanceMemory(capacity=1)
    state = policy.initialize(batch_size=1, model_width=1, with_scores=True)
    candidates = MemoryState.empty(
        batch_size=1,
        capacity=1,
        model_width=1,
        with_scores=True,
    )
    fill_row(candidates, 0, [1], [invalid_score])

    with pytest.raises(ValueError, match="finite scores"):
        policy.update(state, candidates)


def test_importance_requires_raw_token_ids() -> None:
    policy = ExtractiveImportanceMemory(capacity=1)
    state = policy.initialize(batch_size=1, model_width=1, with_scores=True)
    candidates = MemoryState(
        values=torch.zeros(1, 1, 1),
        valid=torch.zeros(1, 1, dtype=torch.bool),
        positions=torch.full((1, 1), -1, dtype=torch.long),
        token_ids=None,
        scores=torch.full((1, 1), float("-inf")),
    )

    with pytest.raises(ValueError, match="token_ids"):
        policy.update(state, candidates)
