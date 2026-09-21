from copy import deepcopy

import pytest

from tinymem.studies.frontier.analysis import analyze


def _rows():
    rows = []
    for mode_number, mode in enumerate(("learned", "compressed_recent", "compressed_diverse", "dictionary_recent")):
        for budget in (64, 256, 1024):
            for level in (0, 2):
                for task_number in range(1, 6):
                    task = f"qa{task_number}"
                    for seed in (27001, 27002, 27003):
                        for question_number in range(2):
                            question = f"{task}-question-{question_number}"
                            rows.append({
                                "id": question, "story": f"{task}-story-{question_number}",
                                "task": task, "answer": "room", "mode": mode,
                                "budget": budget, "level": level,
                                "cell": mode_number * 3 + (64, 256, 1024).index(budget), "seed": seed,
                                "correct": mode == "compressed_recent",
                                "normalized_correct": mode == "compressed_recent",
                                "payload_bytes": 4 if mode == "learned" else 12,
                            })
    return rows


def test_analysis_reports_paired_constant_gain_and_is_order_invariant():
    rows = _rows()
    result = analyze(rows, expected_per_task=2, bootstrap_samples=20)
    comparison = result["comparisons"][0]
    metric = comparison["metrics"]["normalized_correct"]
    assert comparison["direction"] == "learned_minus_explicit"
    assert metric["mean_difference"] == -1.0
    assert metric["bootstrap"]["bounds"] == [-1.0, -1.0]
    assert metric["seed_difference"]["per_seed"] == {"27001": -1.0, "27002": -1.0, "27003": -1.0}
    assert analyze(list(reversed(rows)), expected_per_task=2, bootstrap_samples=20) == result
    assert result["summaries"][0]["normalized_accuracy"]["mean"] == 0.0
    assert result["summaries"][6]["normalized_accuracy"]["mean"] == 1.0


def test_analysis_excludes_controls_and_reports_all_declared_cells():
    rows = _rows()
    rows.append({
        "id": "control", "story": "control-story", "task": "qa1", "answer": "room",
        "mode": "zero", "budget": None, "level": 0, "cell": 0, "seed": 27001,
        "correct": False, "normalized_correct": False, "payload_bytes": None,
    })
    result = analyze(rows, expected_per_task=2, bootstrap_samples=20)
    assert result["controls_excluded"] == ["zero"]
    assert len(result["summaries"]) == 24
    assert len(result["comparisons"]) == 18


@pytest.mark.parametrize("mutation, message", [
    ("missing", "missing"), ("missing_codec", "four declared"),
    ("overbudget", "budget"), ("identity", "identities"),
])
def test_analysis_rejects_invalid_design(mutation, message):
    rows = _rows()
    if mutation == "missing":
        rows.pop()
    elif mutation == "missing_codec":
        rows = [row for row in rows if row["mode"] != "dictionary_recent"]
    elif mutation == "overbudget":
        rows[0]["payload_bytes"] = 65
    else:
        rows[1]["id"] = "different-question"
    with pytest.raises(ValueError, match=message):
        analyze(rows, expected_per_task=2, bootstrap_samples=20)


def test_analysis_does_not_mutate_input_rows():
    rows = _rows()
    original = deepcopy(rows)
    analyze(rows, expected_per_task=2, bootstrap_samples=20)
    assert rows == original


def test_analysis_reports_positive_learned_minus_explicit_gain():
    rows = _rows()
    for row in rows:
        row["correct"] = row["normalized_correct"] = row["mode"] == "learned"
    result = analyze(rows, expected_per_task=2, bootstrap_samples=20)
    assert result["comparisons"][0]["metrics"]["normalized_correct"]["mean_difference"] == 1.0


def test_analysis_keeps_seed_accuracy_separate():
    rows = _rows()
    for row in rows:
        row["correct"] = row["normalized_correct"] = row["mode"] == "learned" and row["seed"] == 27001
    result = analyze(rows, expected_per_task=2, bootstrap_samples=20)
    learned = result["summaries"][0]["normalized_accuracy"]["per_seed"]
    assert learned == {"27001": 1.0, "27002": 0.0, "27003": 0.0}
