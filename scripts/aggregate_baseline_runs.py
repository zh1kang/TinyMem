#!/usr/bin/env python3
"""Aggregate compatible fixed-memory benchmark result files."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from tinymem.evaluation.multiseed import aggregate_baseline_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/predictions/baseline_comparison_multiseed"),
    )
    return parser.parse_args()


def plot_results(result: dict[str, object], destination: Path) -> None:
    """Plot mean outside-window accuracy with sample-standard-deviation bars."""
    baselines = result["baselines"]
    if not isinstance(baselines, list):
        raise ValueError("aggregate result must contain baseline rows")
    names = [str(row["baseline"]).replace("_", " ") for row in baselines]
    means = [float(row["outside_window_accuracy_mean"]) for row in baselines]
    errors = [
        float(row["outside_window_accuracy_sample_std"])
        for row in baselines
    ]
    figure, axis = plt.subplots(figsize=(9, 4.8))
    bars = axis.bar(names, means, yerr=errors, capsize=4)
    axis.axhline(1 / 6, color="gray", linestyle="--", label="chance (1/6)")
    for bar, row in zip(bars, baselines, strict=True):
        axis.annotate(
            f"{int(row['memory_bytes']):,} B",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    axis.set_ylim(0, 1)
    axis.set_ylabel("exact accuracy beyond local window")
    axis.set_title("qa1 fixed-memory comparison across training seeds")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if len(args.results) < 3:
        raise ValueError("final benchmark aggregation requires at least three runs")
    documents = []
    for path in args.results:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid benchmark JSON: {path}") from error
        if not isinstance(document, dict):
            raise ValueError(f"benchmark result must be an object: {path}")
        documents.append(document)

    result = aggregate_baseline_runs(documents)
    result["source_results"] = [str(path.resolve()) for path in args.results]
    created_at = datetime.now(UTC)
    result["created_at"] = created_at.isoformat()
    run_directory = args.artifact_root / (
        f"{created_at.strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid4().hex[:8]}"
    )
    run_directory.mkdir(parents=True, exist_ok=False)
    destination = run_directory / "results.json"
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plot_results(result, run_directory / "baseline_comparison_multiseed.png")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    print(f"artifacts: {run_directory}")


if __name__ == "__main__":
    main()
