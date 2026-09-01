"""Validation and aggregation for fixed-memory benchmark seeds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import fmean, stdev
from typing import Any


MATCHED_RUN_FIELDS = (
    "task_id",
    "manifest_sha256",
    "capacity",
    "shared_memory_bytes_per_example",
    "examples",
    "examples_per_bucket",
    "checkpoint_step",
    "matched_checkpoint_config",
)


def aggregate_baseline_runs(
    runs: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Aggregate compatible fixed-memory runs from distinct training seeds."""
    if len(runs) < 2:
        raise ValueError("at least two benchmark runs are required")

    reference = runs[0]
    for field in MATCHED_RUN_FIELDS:
        if field not in reference:
            raise ValueError(f"benchmark run is missing {field!r}")

    seeds = []
    baseline_rows: dict[str, list[Mapping[str, Any]]] = {}
    reference_names: tuple[str, ...] | None = None
    for run in runs:
        for field in MATCHED_RUN_FIELDS:
            if run.get(field) != reference[field]:
                raise ValueError(f"benchmark runs disagree on {field!r}")

        seed = run.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("each benchmark run must have a nonnegative integer seed")
        seeds.append(seed)

        rows = run.get("baselines")
        if not isinstance(rows, list) or not rows:
            raise ValueError("each benchmark run must contain baseline results")
        names = tuple(_baseline_name(row) for row in rows)
        if len(set(names)) != len(names):
            raise ValueError("baseline names must be unique within each run")
        if reference_names is None:
            reference_names = names
        elif names != reference_names:
            raise ValueError("benchmark runs must contain the same ordered baselines")
        for name, row in zip(names, rows, strict=True):
            baseline_rows.setdefault(name, []).append(row)

    if len(set(seeds)) != len(seeds):
        raise ValueError("benchmark training seeds must be unique")

    assert reference_names is not None
    aggregates = []
    for name in reference_names:
        rows = baseline_rows[name]
        accuracies = [_metric(row, "accuracy") for row in rows]
        outside_accuracies = [
            _metric(row, "outside_window_accuracy") for row in rows
        ]
        memory_bytes = {_integer_metric(row, "memory_bytes") for row in rows}
        if len(memory_bytes) != 1:
            raise ValueError(f"{name!r} memory bytes differ across seeds")
        aggregates.append(
            {
                "baseline": name,
                "memory_bytes": memory_bytes.pop(),
                "accuracy_mean": fmean(accuracies),
                "accuracy_sample_std": stdev(accuracies),
                "outside_window_accuracy_mean": fmean(outside_accuracies),
                "outside_window_accuracy_sample_std": stdev(outside_accuracies),
                "seed_results": [
                    {
                        "seed": seed,
                        "accuracy": accuracy,
                        "outside_window_accuracy": outside_accuracy,
                    }
                    for seed, accuracy, outside_accuracy in zip(
                        seeds,
                        accuracies,
                        outside_accuracies,
                        strict=True,
                    )
                ],
            }
        )

    return {
        "status": _aggregate_status(
            seed_count=len(seeds),
            examples_per_bucket=reference["examples_per_bucket"],
        ),
        "seeds": seeds,
        "seed_count": len(seeds),
        **{field: reference[field] for field in MATCHED_RUN_FIELDS},
        "baselines": aggregates,
    }


def _baseline_name(row: object) -> str:
    if not isinstance(row, Mapping):
        raise ValueError("baseline results must be mappings")
    name = row.get("baseline")
    if not isinstance(name, str) or not name:
        raise ValueError("each baseline result must have a nonempty name")
    return name


def _metric(row: Mapping[str, Any], name: str) -> float:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"baseline metric {name!r} must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"baseline metric {name!r} must be in [0, 1]")
    return result


def _integer_metric(row: Mapping[str, Any], name: str) -> int:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"baseline metric {name!r} must be a nonnegative integer")
    return value


def _aggregate_status(*, seed_count: int, examples_per_bucket: object) -> str:
    if (
        isinstance(examples_per_bucket, int)
        and not isinstance(examples_per_bucket, bool)
        and examples_per_bucket == 0
        and seed_count >= 3
    ):
        return "full_dataset_multi_seed"
    return "development_multi_seed"
