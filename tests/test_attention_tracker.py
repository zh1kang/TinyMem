import pytest
import torch

from tinymem.memory.attention_tracker import (
    AttentionScoreState,
    CumulativeAttentionTracker,
)


def attention_prob(rows: list[list[float]]) -> torch.Tensor:
    return torch.tensor(rows).unsqueeze(1).unsqueeze(1)


def test_tracker_initializes_from_first_attention_call() -> None:
    tracker = CumulativeAttentionTracker()

    expired = tracker.update(
        attention_prob([[0.2, 0.3, 0.5]]),
        torch.tensor([2, 3, 4]),
    )

    assert tracker.initialized
    assert expired.positions.numel() == 0
    assert expired.scores.shape == (1, 0)
    assert torch.equal(tracker.state.positions, torch.tensor([2, 3, 4]))
    torch.testing.assert_close(
        tracker.state.scores,
        torch.tensor([[0.2, 0.3, 0.5]]),
    )


def test_tracker_accumulates_an_unchanged_window() -> None:
    tracker = CumulativeAttentionTracker()
    positions = torch.tensor([2, 3])
    tracker.update(attention_prob([[0.4, 0.6]]), positions)

    expired = tracker.update(attention_prob([[0.1, 0.9]]), positions)

    assert expired.positions.numel() == 0
    torch.testing.assert_close(
        tracker.state.scores,
        torch.tensor([[0.5, 1.5]]),
    )


def test_tracker_returns_expired_scores_when_window_slides() -> None:
    tracker = CumulativeAttentionTracker()
    tracker.update(
        attention_prob([[0.2, 0.3, 0.5]]),
        torch.tensor([2, 3, 4]),
    )

    expired = tracker.update(
        attention_prob([[0.1, 0.2, 0.7]]),
        torch.tensor([3, 4, 5]),
    )

    assert torch.equal(expired.positions, torch.tensor([2]))
    torch.testing.assert_close(expired.scores, torch.tensor([[0.2]]))
    assert torch.equal(tracker.state.positions, torch.tensor([3, 4, 5]))
    torch.testing.assert_close(
        tracker.state.scores,
        torch.tensor([[0.4, 0.7, 0.7]]),
    )


def test_tracker_handles_a_shrinking_window() -> None:
    tracker = CumulativeAttentionTracker()
    tracker.update(
        attention_prob([[0.2, 0.3, 0.5]]),
        torch.tensor([2, 3, 4]),
    )

    expired = tracker.update(
        attention_prob([[0.4, 0.6]]),
        torch.tensor([4, 5]),
    )

    assert torch.equal(expired.positions, torch.tensor([2, 3]))
    torch.testing.assert_close(expired.scores, torch.tensor([[0.2, 0.3]]))
    assert torch.equal(tracker.state.positions, torch.tensor([4, 5]))
    torch.testing.assert_close(
        tracker.state.scores,
        torch.tensor([[0.9, 0.6]]),
    )


def test_tracker_handles_a_growing_window() -> None:
    tracker = CumulativeAttentionTracker()
    tracker.update(
        attention_prob([[0.4, 0.6]]),
        torch.tensor([2, 3]),
    )

    expired = tracker.update(
        attention_prob([[0.2, 0.3, 0.5]]),
        torch.tensor([2, 3, 4]),
    )

    assert expired.positions.numel() == 0
    torch.testing.assert_close(
        tracker.state.scores,
        torch.tensor([[0.6, 0.9, 0.5]]),
    )


def test_tracker_preserves_batch_rows() -> None:
    tracker = CumulativeAttentionTracker()
    positions = torch.tensor([0, 1])
    tracker.update(
        attention_prob([[0.8, 0.2], [0.1, 0.9]]),
        positions,
    )

    tracker.update(
        attention_prob([[0.3, 0.7], [0.6, 0.4]]),
        positions,
    )

    torch.testing.assert_close(
        tracker.state.scores,
        torch.tensor([[1.1, 0.9], [0.7, 1.3]]),
    )


def test_tracker_scores_only_the_tracked_local_key_suffix() -> None:
    tracker = CumulativeAttentionTracker()
    probabilities = torch.tensor([[[[0.6, 0.1, 0.3]]]])

    tracker.update(probabilities, torch.tensor([8, 9]))

    assert torch.equal(tracker.state.positions, torch.tensor([8, 9]))
    torch.testing.assert_close(tracker.state.scores, torch.tensor([[0.1, 0.3]]))


def test_tracker_state_is_a_defensive_copy() -> None:
    tracker = CumulativeAttentionTracker()
    tracker.update(attention_prob([[1.0]]), torch.tensor([0]))

    copied_state = tracker.state
    copied_state.positions.fill_(9)
    copied_state.scores.fill_(9.0)

    assert torch.equal(tracker.state.positions, torch.tensor([0]))
    torch.testing.assert_close(tracker.state.scores, torch.tensor([[1.0]]))


def test_tracker_detaches_cumulative_accounting_from_autograd() -> None:
    tracker = CumulativeAttentionTracker()
    logits = torch.randn(1, 1, 1, 2, requires_grad=True)

    tracker.update(torch.softmax(logits, dim=-1), torch.tensor([0, 1]))

    assert not tracker.state.scores.requires_grad


def test_tracker_reset_clears_state() -> None:
    tracker = CumulativeAttentionTracker()
    tracker.update(attention_prob([[1.0]]), torch.tensor([0]))

    tracker.reset()

    assert not tracker.initialized
    with pytest.raises(RuntimeError, match="has not observed"):
        _ = tracker.state


def test_tracker_rejects_batch_size_changes() -> None:
    tracker = CumulativeAttentionTracker()
    tracker.update(attention_prob([[1.0]]), torch.tensor([0]))

    with pytest.raises(ValueError, match="batch size"):
        tracker.update(
            attention_prob([[1.0], [1.0]]),
            torch.tensor([1]),
        )


def test_tracker_rejects_backward_windows() -> None:
    tracker = CumulativeAttentionTracker()
    tracker.update(attention_prob([[0.5, 0.5]]), torch.tensor([2, 3]))

    with pytest.raises(ValueError, match="backward"):
        tracker.update(
            attention_prob([[0.5, 0.5]]),
            torch.tensor([1, 3]),
        )


@pytest.mark.parametrize(
    ("positions", "scores", "error"),
    [
        (torch.tensor([[0]]), torch.ones(1, 1), ValueError),
        (torch.tensor([0.0]), torch.ones(1, 1), TypeError),
        (torch.tensor([0, 1]), torch.ones(1, 1), ValueError),
        (torch.tensor([1, 1]), torch.ones(1, 2), ValueError),
        (torch.tensor([-1]), torch.ones(1, 1), ValueError),
        (torch.tensor([0]), torch.ones(1, 1, dtype=torch.float64), TypeError),
    ],
)
def test_attention_score_state_rejects_invalid_tensors(
    positions: torch.Tensor,
    scores: torch.Tensor,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        AttentionScoreState(positions=positions, scores=scores)
