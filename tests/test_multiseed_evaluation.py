import pytest

from tinymem.evaluation.multiseed import aggregate_baseline_runs


def make_run(seed: int, recent_accuracy: float) -> dict[str, object]:
    return {
        "task_id": "qa1",
        "manifest_sha256": "manifest",
        "capacity": 12,
        "shared_memory_bytes_per_example": 3324,
        "examples": 400,
        "examples_per_bucket": 0,
        "checkpoint_step": 2000,
        "matched_checkpoint_config": {
            "model": {"d_model": 64, "max_local_tokens": 128},
            "training": {"max_steps": 2000},
        },
        "seed": seed,
        "oracle_dominates": True,
        "baselines": [
            {
                "baseline": "local",
                "memory_bytes": 0,
                "accuracy": 0.25,
                "outside_window_accuracy": 0.2,
            },
            {
                "baseline": "recent",
                "memory_bytes": 3324,
                "accuracy": recent_accuracy,
                "outside_window_accuracy": recent_accuracy,
            },
        ],
    }


def test_aggregate_baseline_runs_reports_all_seeds_and_sample_statistics() -> None:
    result = aggregate_baseline_runs(
        [make_run(1, 0.2), make_run(2, 0.3), make_run(3, 0.4)]
    )

    assert result["status"] == "full_dataset_multi_seed"
    assert result["seeds"] == [1, 2, 3]
    assert result["oracle_dominates_all_seeds"] is True
    recent = result["baselines"][1]
    assert recent["outside_window_accuracy_mean"] == pytest.approx(0.3)
    assert recent["outside_window_accuracy_sample_std"] == pytest.approx(0.1)
    assert [row["seed"] for row in recent["seed_results"]] == [1, 2, 3]


def test_aggregate_baseline_runs_rejects_duplicate_training_seeds() -> None:
    with pytest.raises(ValueError, match="unique"):
        aggregate_baseline_runs([make_run(1, 0.2), make_run(1, 0.3)])


def test_aggregate_baseline_runs_rejects_mismatched_evaluation_inputs() -> None:
    first = make_run(1, 0.2)
    second = make_run(2, 0.3)
    second["manifest_sha256"] = "different"

    with pytest.raises(ValueError, match="manifest_sha256"):
        aggregate_baseline_runs([first, second])
