from copy import deepcopy

import pytest

from tinymem.evaluation.controller_sweep import (
    WRITE_COST_WEIGHTS,
    aggregate_controller_seeds,
    aggregate_controller_sweep,
)


def make_result(weight: float, accuracy: float, writes: float) -> dict[str, object]:
    count = 100
    correct = round(accuracy * count)
    return {
        "seed": 7,
        "base_checkpoint_sha256": "base",
        "training_steps": 20,
        "memory_warmup_steps": 5,
        "training_examples": 50,
        "validation_examples": 20,
        "compressor": "multislot_attention",
        "memory_update": "gated",
        "write_gate": "adaptive",
        "segment_length": 8,
        "capacity": 4,
        "summaries_per_segment": 1,
        "memory_bytes_per_example": 128,
        "max_distributed_distractor_tokens": 64,
        "validation_distributed_distractor_tokens": 64,
        "controller_hidden_width": 8,
        "controller_temperature_start": 2.0,
        "controller_temperature_end": 0.5,
        "controller_anneal_steps": 20,
        "write_cost_weight": weight,
        "controller_comparison": {
            "policies": [
                {
                    "policy": "learned",
                    "correct": correct,
                    "count": count,
                    "writes_per_1000_tokens": writes,
                }
            ]
        },
    }


def make_sweep() -> list[dict[str, object]]:
    metrics = (
        (0.50, 400.0),
        (0.55, 300.0),
        (0.54, 200.0),
        (0.45, 250.0),
        (0.30, 100.0),
    )
    return [
        make_result(weight, accuracy, writes)
        for weight, (accuracy, writes) in zip(
            WRITE_COST_WEIGHTS,
            metrics,
            strict=True,
        )
    ]


def test_controller_sweep_marks_dominated_points() -> None:
    result = aggregate_controller_sweep(make_sweep())

    points = result["points"]
    assert isinstance(points, list)
    assert [point["write_cost_weight"] for point in points] == list(
        WRITE_COST_WEIGHTS
    )
    assert [point["pareto_optimal"] for point in points] == [
        False,
        True,
        True,
        False,
        True,
    ]
    frontier = result["pareto_frontier"]
    assert isinstance(frontier, list)
    assert [point["writes_per_1000_tokens"] for point in frontier] == [
        100.0,
        200.0,
        300.0,
    ]


def test_controller_sweep_rejects_protocol_mismatch() -> None:
    documents = make_sweep()
    documents[1] = deepcopy(documents[1])
    documents[1]["seed"] = 8

    with pytest.raises(ValueError, match="seed"):
        aggregate_controller_sweep(documents)


def test_controller_sweep_requires_exact_weights() -> None:
    documents = make_sweep()
    documents[-1]["write_cost_weight"] = 0.2

    with pytest.raises(ValueError, match="required write costs"):
        aggregate_controller_sweep(documents)


def make_seed_result(seed: int, learned: float, random: float) -> dict[str, object]:
    document = make_result(1e-4, learned, 1.6)
    document.update(
        {
            "seed": seed,
            "base_checkpoint_sha256": f"checkpoint-{seed}",
            "task_id": "qa1",
            "manifest_sha256": "manifest",
        }
    )
    comparison = document["controller_comparison"]
    assert isinstance(comparison, dict)
    comparison.update(
        {
            "policies": [
                {
                    "policy": "random_matched",
                    "accuracy": random,
                    "writes_per_1000_tokens": 1.6,
                },
                {
                    "policy": "learned",
                    "accuracy": learned,
                    "writes_per_1000_tokens": 1.6,
                },
            ],
            "relevant_write_rate": 0.9,
            "background_write_rate": 0.01,
            "learned_vs_random": {
                "accuracy_difference": learned - random,
                "significant": learned > random,
            },
        }
    )
    return document


def test_controller_seed_aggregation_reports_variation() -> None:
    result = aggregate_controller_seeds(
        [
            make_seed_result(1, 0.25, 0.18),
            make_seed_result(2, 0.17, 0.14),
            make_seed_result(3, 0.16, 0.17),
        ]
    )

    assert result["status"] == "development_multi_seed"
    assert result["learned_beats_random_seed_count"] == 2
    assert result["learned_vs_random_significant_seed_count"] == 2
    assert result["learned_vs_random_accuracy_difference_mean"] == pytest.approx(0.03)
    learned = result["policies"][1]
    assert learned["accuracy_mean"] == pytest.approx(0.1933333333)
    assert learned["accuracy_sample_std"] > 0


def test_controller_seed_aggregation_requires_independent_checkpoints() -> None:
    first = make_seed_result(1, 0.2, 0.1)
    second = make_seed_result(2, 0.2, 0.1)
    second["base_checkpoint_sha256"] = first["base_checkpoint_sha256"]

    with pytest.raises(ValueError, match="independently trained"):
        aggregate_controller_seeds([first, second])
