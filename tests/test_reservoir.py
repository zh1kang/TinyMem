import pytest
import torch

from tinymem.memory.reservoir import RandomReservoirMemory
from tinymem.memory.state import MemoryState


def fill_row(
    state: MemoryState,
    batch_index: int,
    positions: list[int],
    scores: list[float] | None = None,
) -> None:
    count = len(positions)
    state.valid[batch_index, :count] = True
    state.positions[batch_index, :count] = torch.tensor(positions)
    assert state.token_ids is not None
    state.token_ids[batch_index, :count] = torch.tensor(positions)
    values = torch.tensor(positions, dtype=state.values.dtype).unsqueeze(-1)
    state.values[batch_index, :count] = values.repeat(1, state.model_width)
    if scores is not None:
        assert state.scores is not None
        state.scores[batch_index, :count] = torch.tensor(
            scores,
            dtype=state.scores.dtype,
        )


def test_reservoir_selects_valid_items_and_preserves_metadata_alignment() -> None:
    policy = RandomReservoirMemory(capacity=3)
    state = policy.initialize(batch_size=1, model_width=2, with_scores=True)
    candidates = MemoryState.empty(batch_size=1, capacity=4, model_width=2)
    fill_row(state, 0, [1, 2], [0.95, 0.90])
    fill_row(candidates, 0, [3, 4, 5, 6])

    result = policy.update(
        state,
        candidates,
        generator=torch.Generator().manual_seed(4),
    )

    assert result.valid.sum().item() == 3
    assert result.scores is not None
    assert torch.equal(result.positions[0, :3], result.token_ids[0, :3])
    assert torch.equal(result.values[0, :3, 0].to(torch.long), result.positions[0, :3])
    assert torch.all(result.positions[0, :3][1:] >= result.positions[0, :3][:-1])
    assert torch.isfinite(result.scores[0, :3]).all()


def test_reservoir_is_reproducible_with_a_seeded_generator() -> None:
    policy = RandomReservoirMemory(capacity=2)
    state = policy.initialize(batch_size=1, model_width=1, with_scores=True)
    candidates = MemoryState.empty(batch_size=1, capacity=4, model_width=1)
    fill_row(candidates, 0, [1, 2, 3, 4])

    first = policy.update(
        state,
        candidates,
        generator=torch.Generator().manual_seed(12),
    )
    second = policy.update(
        state,
        candidates,
        generator=torch.Generator().manual_seed(12),
    )

    assert torch.equal(first.positions, second.positions)
    assert torch.equal(first.token_ids, second.token_ids)
    assert first.scores is not None and second.scores is not None
    assert torch.equal(first.scores, second.scores)


def test_reservoir_keeps_existing_scores_across_updates() -> None:
    policy = RandomReservoirMemory(capacity=1)
    state = policy.initialize(batch_size=1, model_width=1, with_scores=True)
    candidates = MemoryState.empty(batch_size=1, capacity=1, model_width=1)
    fill_row(state, 0, [10], [1.0])
    fill_row(candidates, 0, [20])

    result = policy.update(
        state,
        candidates,
        generator=torch.Generator().manual_seed(0),
    )

    assert result.scores is not None
    assert result.positions.item() == 10
    assert result.scores.item() == pytest.approx(1.0)


def test_reservoir_excludes_invalid_candidates_and_keeps_empty_sentinels() -> None:
    policy = RandomReservoirMemory(capacity=2)
    state = policy.initialize(batch_size=1, model_width=1, with_scores=True)
    candidates = MemoryState.empty(batch_size=1, capacity=2, model_width=1)
    fill_row(candidates, 0, [7])

    result = policy.update(state, candidates)

    assert torch.equal(result.valid, torch.tensor([[True, False]]))
    assert torch.equal(result.positions, torch.tensor([[7, -1]]))
    assert torch.equal(result.token_ids, torch.tensor([[7, -1]]))
    assert result.scores is not None
    assert torch.isfinite(result.scores[0, 0])
    assert torch.isneginf(result.scores[0, 1])


def test_reservoir_does_not_modify_inputs() -> None:
    policy = RandomReservoirMemory(capacity=2)
    state = policy.initialize(batch_size=1, model_width=1, with_scores=True)
    candidates = MemoryState.empty(batch_size=1, capacity=2, model_width=1)
    fill_row(state, 0, [1], [0.5])
    fill_row(candidates, 0, [2])
    original_state = (state.values.clone(), state.positions.clone(), state.scores.clone())
    original_candidates = (candidates.values.clone(), candidates.positions.clone())

    policy.update(state, candidates, generator=torch.Generator().manual_seed(2))

    assert torch.equal(state.values, original_state[0])
    assert torch.equal(state.positions, original_state[1])
    assert torch.equal(state.scores, original_state[2])
    assert torch.equal(candidates.values, original_candidates[0])
    assert torch.equal(candidates.positions, original_candidates[1])


def test_reservoir_requires_scored_state() -> None:
    policy = RandomReservoirMemory(capacity=2)
    state = policy.initialize(batch_size=1, model_width=1)
    candidates = MemoryState.empty(batch_size=1, capacity=2, model_width=1)

    with pytest.raises(ValueError, match="scores"):
        policy.update(state, candidates)
