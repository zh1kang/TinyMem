"""Report boundaries must reject empty scientific cohorts."""
import pytest

from tinymem.research.readout_report import accuracy_metrics


def test_empty_cohort_is_not_reported_as_perfect_accuracy():
    with pytest.raises(ValueError):
        accuracy_metrics([])


def test_named_control_comparison_preserves_condition_names():
    from tinymem.research.readout_report import paired_control_comparison

    rows = [
        {"condition": condition, "seed": 17, "history_id": history,
         "case_id": category, "category": category, "correct": condition == "normal"}
        for condition in ("normal", "zero")
        for history in ("h1", "h2")
        for category in ("known", "missing")
    ]
    result = paired_control_comparison(rows, reference="zero", resamples=20)
    assert result["contrast"] == {"condition": "normal", "reference": "zero"}
    assert result["metrics"]["known"]["mean_difference"] == 1.0
