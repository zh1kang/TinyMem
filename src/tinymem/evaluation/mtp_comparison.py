"""Aggregate matched multi-token prediction delay curves."""

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev


@dataclass(frozen=True)
class MTPDelayRun:
    """Hold one condition, seed, horizon set, and exact delay counts."""

    condition: str
    seed: int
    horizons: tuple[int, ...]
    curve: tuple[dict[str, object], ...]


def load_mtp_delay_run(path: str | Path, *, condition: str) -> MTPDelayRun:
    """Load either a base or continuous-memory delay result."""
    if not isinstance(condition, str) or not condition:
        raise ValueError("condition must be a nonempty string")
    result_path = Path(path)
    document = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("MTP result must contain an object")
    seed = document.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("MTP result must contain a nonnegative seed")
    raw_horizons = document.get("mtp_horizons", [])
    if not isinstance(raw_horizons, list) or any(
        isinstance(horizon, bool) or not isinstance(horizon, int)
        for horizon in raw_horizons
    ):
        raise ValueError("MTP horizons must be a list of integers")
    curve = document.get("curve")
    if curve is None:
        evaluation = document.get("evaluation")
        if not isinstance(evaluation, dict):
            raise ValueError("MTP result must contain an evaluation")
        curve = evaluation.get("curve")
    if not isinstance(curve, list) or not curve:
        raise ValueError("MTP result must contain a nonempty delay curve")
    for bucket in curve:
        if not isinstance(bucket, dict):
            raise ValueError("delay buckets must be objects")
        if not isinstance(bucket.get("label"), str):
            raise ValueError("delay buckets must contain labels")
        for key in ("correct", "count"):
            value = bucket.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"delay bucket {key} must be nonnegative")
        if int(bucket["correct"]) > int(bucket["count"]):
            raise ValueError("delay bucket correct count exceeds total count")
    return MTPDelayRun(
        condition=condition,
        seed=seed,
        horizons=tuple(raw_horizons),
        curve=tuple(curve),
    )


def aggregate_mtp_delay_runs(
    runs: Sequence[MTPDelayRun],
) -> list[dict[str, object]]:
    """Pool exact counts and summarize seed accuracy for each condition."""
    if not runs:
        raise ValueError("runs must be nonempty")
    if not all(isinstance(run, MTPDelayRun) for run in runs):
        raise TypeError("runs must contain MTPDelayRun values")
    grouped: dict[str, list[MTPDelayRun]] = defaultdict(list)
    for run in runs:
        grouped[run.condition].append(run)

    summaries = []
    for condition, condition_runs in sorted(grouped.items()):
        if len({run.seed for run in condition_runs}) != len(condition_runs):
            raise ValueError(f"condition {condition!r} contains duplicate seeds")
        horizons = {run.horizons for run in condition_runs}
        if len(horizons) != 1:
            raise ValueError(f"condition {condition!r} mixes MTP horizons")
        labels = [str(bucket["label"]) for bucket in condition_runs[0].curve]
        if any(
            [str(bucket["label"]) for bucket in run.curve] != labels
            for run in condition_runs[1:]
        ):
            raise ValueError(f"condition {condition!r} mixes delay buckets")

        curve = []
        for index, label in enumerate(labels):
            correct = sum(int(run.curve[index]["correct"]) for run in condition_runs)
            count = sum(int(run.curve[index]["count"]) for run in condition_runs)
            seed_accuracies = [
                (
                    int(run.curve[index]["correct"])
                    / int(run.curve[index]["count"])
                    if int(run.curve[index]["count"])
                    else 0.0
                )
                for run in condition_runs
            ]
            curve.append(
                {
                    "label": label,
                    "correct": correct,
                    "count": count,
                    "accuracy": correct / count if count else 0.0,
                    "seed_accuracy_mean": mean(seed_accuracies),
                    "seed_accuracy_std": (
                        stdev(seed_accuracies) if len(seed_accuracies) > 1 else 0.0
                    ),
                }
            )
        summaries.append(
            {
                "condition": condition,
                "seeds": sorted(run.seed for run in condition_runs),
                "seed_count": len(condition_runs),
                "mtp_horizons": list(next(iter(horizons))),
                "curve": curve,
            }
        )
    return summaries
