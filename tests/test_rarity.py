import pytest
import torch

from tinymem.memory.rarity import TokenRarityScorer, count_token_frequencies
from tinymem.memory.state import MemoryState


def test_count_token_frequencies_ignores_invalid_slots() -> None:
    token_ids = torch.tensor([[0, 1, -1], [1, 3, 2]])
    valid = torch.tensor(
        [[True, True, False], [True, True, True]],
    )

    counts = count_token_frequencies(token_ids, valid, vocab_size=4)

    assert torch.equal(counts, torch.tensor([1, 2, 1, 1]))


def test_count_token_frequencies_handles_no_valid_tokens() -> None:
    token_ids = torch.full((2, 3), -1, dtype=torch.long)
    valid = torch.zeros(2, 3, dtype=torch.bool)

    counts = count_token_frequencies(token_ids, valid, vocab_size=4)

    assert torch.equal(counts, torch.zeros(4, dtype=torch.long))


@pytest.mark.parametrize(
    "token_ids, valid, vocab_size, error",
    [
        (torch.zeros(2, dtype=torch.float32), torch.ones(2, dtype=torch.bool), 4, TypeError),
        (torch.zeros(2, dtype=torch.long), torch.ones(2), 4, TypeError),
        (torch.zeros(2, dtype=torch.long), torch.ones(3, dtype=torch.bool), 4, ValueError),
        (torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.bool), 0, ValueError),
    ],
)
def test_count_token_frequencies_rejects_invalid_inputs(
    token_ids: torch.Tensor,
    valid: torch.Tensor,
    vocab_size: int,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        count_token_frequencies(token_ids, valid, vocab_size=vocab_size)


def test_count_token_frequencies_rejects_valid_out_of_range_ids() -> None:
    token_ids = torch.tensor([0, 4])
    valid = torch.tensor([True, True])

    with pytest.raises(ValueError, match="valid token IDs"):
        count_token_frequencies(token_ids, valid, vocab_size=4)


def test_rarity_scorer_owns_a_float_copy_of_counts() -> None:
    counts = torch.tensor([1, 2, 3], dtype=torch.long)

    scorer = TokenRarityScorer(counts, smoothing=0.5)
    counts.zero_()

    assert torch.equal(scorer.token_counts, torch.tensor([1.0, 2.0, 3.0]))
    assert scorer.smoothing == 0.5
    assert scorer.vocab_size == 3
    assert scorer.total_count.item() == 6.0


@pytest.mark.parametrize(
    "counts, error",
    [
        (torch.tensor([], dtype=torch.long), ValueError),
        (torch.tensor([[1, 2]], dtype=torch.long), ValueError),
        (torch.tensor([1.0, 2.0]), TypeError),
        (torch.tensor([1, -1]), ValueError),
        (torch.tensor([0, 0]), ValueError),
    ],
)
def test_rarity_scorer_rejects_invalid_counts(
    counts: torch.Tensor,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        TokenRarityScorer(counts)


@pytest.mark.parametrize("smoothing", [0.0, -1.0, float("inf"), float("nan")])
def test_rarity_scorer_rejects_invalid_smoothing(smoothing: float) -> None:
    with pytest.raises(ValueError, match="smoothing"):
        TokenRarityScorer(torch.tensor([1, 2]), smoothing=smoothing)


def test_rarity_scorer_uses_smoothed_negative_log_probability() -> None:
    counts = torch.tensor([100, 10, 1, 0])
    scorer = TokenRarityScorer(counts, smoothing=1.0)
    token_ids = torch.tensor([[0, 1, 2, 3, -1]])
    valid = torch.tensor([[True, True, True, True, False]])

    scores = scorer.score(token_ids, valid)

    expected = -torch.log((counts.to(torch.float32) + 1.0) / 115.0)
    torch.testing.assert_close(scores[valid], expected)
    assert torch.isneginf(scores[~valid]).all()
    assert scores[0, 3] > scores[0, 2] > scores[0, 1] > scores[0, 0]


def test_rarity_scorer_handles_an_empty_valid_mask() -> None:
    scorer = TokenRarityScorer(torch.tensor([2, 1]))
    token_ids = torch.full((2, 3), -1, dtype=torch.long)
    valid = torch.zeros(2, 3, dtype=torch.bool)

    scores = scorer.score(token_ids, valid)

    assert scores.shape == token_ids.shape
    assert scores.dtype == torch.float32
    assert torch.isneginf(scores).all()


def test_rarity_scorer_rejects_valid_out_of_range_ids() -> None:
    scorer = TokenRarityScorer(torch.tensor([2, 1]))
    token_ids = torch.tensor([0, 2])
    valid = torch.tensor([True, True])

    with pytest.raises(ValueError, match="valid token IDs"):
        scorer.score(token_ids, valid)


@pytest.mark.parametrize(
    "token_ids, valid, error",
    [
        (torch.zeros(2), torch.ones(2, dtype=torch.bool), TypeError),
        (torch.zeros(2, dtype=torch.long), torch.ones(2), TypeError),
        (
            torch.zeros(2, dtype=torch.long),
            torch.ones(3, dtype=torch.bool),
            ValueError,
        ),
    ],
)
def test_rarity_scorer_rejects_invalid_score_inputs(
    token_ids: torch.Tensor,
    valid: torch.Tensor,
    error: type[Exception],
) -> None:
    scorer = TokenRarityScorer(torch.tensor([2, 1]))

    with pytest.raises(error):
        scorer.score(token_ids, valid)


def test_rarity_scorer_returns_a_fresh_scored_state() -> None:
    scorer = TokenRarityScorer(torch.tensor([4, 2, 1]))
    state = MemoryState.empty(batch_size=1, capacity=3, model_width=2)
    state.valid[0, :2] = True
    state.positions[0, :2] = torch.tensor([4, 7])
    assert state.token_ids is not None
    state.token_ids[0, :2] = torch.tensor([0, 2])
    state.values[0, 0] = torch.tensor([1.0, 2.0])
    state.values[0, 1] = torch.tensor([3.0, 4.0])

    result = scorer.score_state(state)

    assert result.scores is not None
    torch.testing.assert_close(
        result.scores[0, :2],
        -torch.log(torch.tensor([0.5, 0.2])),
    )
    assert torch.isneginf(result.scores[0, 2])
    assert torch.equal(result.values, state.values)
    assert torch.equal(result.valid, state.valid)
    assert torch.equal(result.positions, state.positions)
    assert torch.equal(result.token_ids, state.token_ids)
    assert result.values.data_ptr() != state.values.data_ptr()
    assert result.valid.data_ptr() != state.valid.data_ptr()
    assert result.positions.data_ptr() != state.positions.data_ptr()
    assert result.token_ids.data_ptr() != state.token_ids.data_ptr()
    assert state.scores is None
