import pytest
import torch

from tinymem.evaluation.answerability import (
    evaluate_answerability,
    expected_calibration_error,
    selective_prediction_point,
)
from tinymem.evaluation.continuous_memory import (
    drop_newest_memory,
    keep_oldest_memory,
    replace_newest_memory,
)
from tinymem.model.memory_input import AttentionMemory


def test_selective_prediction_point_tracks_abstention_quality() -> None:
    probabilities = torch.tensor([0.9, 0.8, 0.4, 0.1])
    correct = torch.tensor([True, False, False, False])
    answerable = torch.tensor([True, True, False, False])

    point = selective_prediction_point(
        probabilities,
        correct,
        answerable,
        threshold=0.5,
    )

    assert point.coverage == 0.5
    assert point.selective_accuracy == 0.5
    assert point.selective_risk == 0.5
    assert point.abstention_precision == 1.0
    assert point.abstention_recall == 1.0
    assert point.false_confidence_rate == 0.0


def test_answerability_evaluation_reports_calibration_and_curve() -> None:
    probabilities = torch.tensor([0.9, 0.8, 0.2, 0.1])
    correct = torch.tensor([True, True, False, False])
    answerable = torch.tensor([True, True, False, False])

    result = evaluate_answerability(
        probabilities,
        correct,
        answerable,
        thresholds=(0.25, 0.5, 0.75),
        calibration_bins=2,
    )

    assert result.brier_score == pytest.approx(0.025)
    assert result.expected_calibration_error == pytest.approx(0.15)
    assert result.mean_answerable_probability == pytest.approx(0.85)
    assert result.mean_unanswerable_probability == pytest.approx(0.15)
    assert len(result.points) == 3
    assert result.to_dict()["points"][1]["coverage"] == 0.5


def test_expected_calibration_error_requires_boolean_targets() -> None:
    with pytest.raises(TypeError, match="boolean"):
        expected_calibration_error(
            torch.tensor([0.5]),
            torch.tensor([1]),
        )


def test_slot_interventions_preserve_valid_memory_invariants() -> None:
    memory = AttentionMemory(
        values=torch.tensor(
            [
                [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
                [[0.0, 0.0], [3.0, 3.0], [4.0, 4.0]],
            ]
        ),
        valid=torch.tensor(
            [[False, True, True], [False, True, True]]
        ),
        positions=torch.tensor([[-1, 2, 5], [-1, 3, 6]]),
    )

    dropped = drop_newest_memory(memory)
    stale = keep_oldest_memory(memory)
    replaced = replace_newest_memory(memory)

    expected_valid = torch.tensor(
        [[False, True, False], [False, True, False]]
    )
    assert torch.equal(dropped.valid, expected_valid)
    assert torch.equal(stale.valid, expected_valid)
    assert torch.equal(dropped.positions, torch.tensor([[-1, 2, -1], [-1, 3, -1]]))
    assert torch.equal(replaced.values[0, 2], memory.values[1, 2])
    assert torch.equal(replaced.values[1, 2], memory.values[0, 2])
