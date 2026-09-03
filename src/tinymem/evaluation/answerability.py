"""Selective prediction metrics for answerability estimates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral, Real

import torch


@dataclass(frozen=True)
class SelectivePredictionPoint:
    """Summarize model behavior at one answerability threshold."""

    threshold: float
    coverage: float
    selective_accuracy: float
    selective_risk: float
    abstention_precision: float
    abstention_recall: float
    false_confidence_rate: float
    answered: int
    total: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class AnswerabilityEvaluation:
    """Hold threshold, calibration, and coverage-accuracy measurements."""

    brier_score: float
    expected_calibration_error: float
    mean_answerable_probability: float
    mean_unanswerable_probability: float
    points: tuple[SelectivePredictionPoint, ...]

    def to_dict(self) -> dict[str, object]:
        document = asdict(self)
        document["points"] = [point.to_dict() for point in self.points]
        return document


def _validated_inputs(
    probabilities: torch.Tensor,
    correct: torch.Tensor,
    answerable: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(probabilities, torch.Tensor):
        raise TypeError("probabilities must be a torch.Tensor")
    if not isinstance(correct, torch.Tensor):
        raise TypeError("correct must be a torch.Tensor")
    if not isinstance(answerable, torch.Tensor):
        raise TypeError("answerable must be a torch.Tensor")
    if probabilities.ndim != 1:
        raise ValueError("probabilities must have shape [examples]")
    if probabilities.numel() == 0:
        raise ValueError("probabilities must be nonempty")
    if not probabilities.is_floating_point():
        raise TypeError("probabilities must be a floating-point tensor")
    if correct.shape != probabilities.shape or answerable.shape != probabilities.shape:
        raise ValueError("correct and answerable must match probabilities")
    if correct.dtype != torch.bool or answerable.dtype != torch.bool:
        raise TypeError("correct and answerable must be boolean tensors")
    if (
        correct.device != probabilities.device
        or answerable.device != probabilities.device
    ):
        raise ValueError("all inputs must share a device")
    if not torch.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probabilities must be in [0, 1]")
    return probabilities, correct, answerable


def selective_prediction_point(
    probabilities: torch.Tensor,
    correct: torch.Tensor,
    answerable: torch.Tensor,
    *,
    threshold: float,
) -> SelectivePredictionPoint:
    """Compute coverage, selective risk, and abstention quality."""
    probabilities, correct, answerable = _validated_inputs(
        probabilities,
        correct,
        answerable,
    )
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise TypeError("threshold must be a real number")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    answered = probabilities >= threshold
    abstained = ~answered
    unanswerable = ~answerable
    answered_count = int(answered.sum())
    abstained_count = int(abstained.sum())
    unanswerable_count = int(unanswerable.sum())
    selective_accuracy = (
        float(correct[answered].float().mean()) if answered_count else 0.0
    )
    abstention_precision = (
        float(unanswerable[abstained].float().mean()) if abstained_count else 0.0
    )
    abstention_recall = (
        float((abstained & unanswerable).sum() / unanswerable_count)
        if unanswerable_count
        else 0.0
    )
    false_confidence_rate = (
        float((answered & unanswerable).sum() / unanswerable_count)
        if unanswerable_count
        else 0.0
    )
    return SelectivePredictionPoint(
        threshold=float(threshold),
        coverage=answered_count / probabilities.numel(),
        selective_accuracy=selective_accuracy,
        selective_risk=1.0 - selective_accuracy,
        abstention_precision=abstention_precision,
        abstention_recall=abstention_recall,
        false_confidence_rate=false_confidence_rate,
        answered=answered_count,
        total=probabilities.numel(),
    )


def expected_calibration_error(
    probabilities: torch.Tensor,
    answerable: torch.Tensor,
    *,
    bins: int = 10,
) -> float:
    """Return equal-width expected calibration error for answerability."""
    probabilities, _, answerable = _validated_inputs(
        probabilities,
        torch.zeros_like(answerable),
        answerable,
    )
    if isinstance(bins, bool) or not isinstance(bins, Integral):
        raise TypeError("bins must be an integer")
    if bins <= 0:
        raise ValueError("bins must be positive")

    boundaries = torch.linspace(
        0,
        1,
        int(bins) + 1,
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    error = probabilities.new_zeros(())
    for index in range(int(bins)):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        in_bin = (probabilities >= lower) & (
            probabilities <= upper
            if index == bins - 1
            else probabilities < upper
        )
        if in_bin.any():
            confidence = probabilities[in_bin].mean()
            frequency = answerable[in_bin].float().mean()
            error = error + in_bin.float().mean() * (confidence - frequency).abs()
    return float(error)


def evaluate_answerability(
    probabilities: torch.Tensor,
    correct: torch.Tensor,
    answerable: torch.Tensor,
    *,
    thresholds: tuple[float, ...] = (0.25, 0.5, 0.75),
    calibration_bins: int = 10,
) -> AnswerabilityEvaluation:
    """Evaluate calibration and the coverage-accuracy tradeoff."""
    probabilities, correct, answerable = _validated_inputs(
        probabilities,
        correct,
        answerable,
    )
    if not isinstance(thresholds, tuple) or not thresholds:
        raise ValueError("thresholds must be a nonempty tuple")
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not 0.0 <= value <= 1.0
        for value in thresholds
    ):
        raise ValueError("thresholds must contain values in [0, 1]")
    if tuple(sorted(set(thresholds))) != thresholds:
        raise ValueError("thresholds must be unique and increasing")

    answerable_probabilities = probabilities[answerable]
    unanswerable_probabilities = probabilities[~answerable]
    targets = answerable.to(dtype=probabilities.dtype)
    return AnswerabilityEvaluation(
        brier_score=float(((probabilities - targets) ** 2).mean()),
        expected_calibration_error=expected_calibration_error(
            probabilities,
            answerable,
            bins=calibration_bins,
        ),
        mean_answerable_probability=(
            float(answerable_probabilities.mean())
            if answerable_probabilities.numel()
            else 0.0
        ),
        mean_unanswerable_probability=(
            float(unanswerable_probabilities.mean())
            if unanswerable_probabilities.numel()
            else 0.0
        ),
        points=tuple(
            selective_prediction_point(
                probabilities,
                correct,
                answerable,
                threshold=threshold,
            )
            for threshold in thresholds
        ),
    )
