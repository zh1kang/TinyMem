#!/usr/bin/env python3
"""Aggregate the required adaptive write-cost sweep and plot its frontier."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from tinymem.evaluation.controller_sweep import aggregate_controller_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def plot_frontier(result: dict[str, object], destination: Path) -> None:
    """Plot delayed accuracy against writes per 1,000 tokens."""
    points = result.get("points")
    frontier = result.get("pareto_frontier")
    if not isinstance(points, list) or not isinstance(frontier, list):
        raise ValueError("controller sweep result must contain point lists")
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.scatter(
        [float(point["writes_per_1000_tokens"]) for point in points],
        [float(point["delayed_recall_accuracy"]) for point in points],
        label="write-cost runs",
    )
    axis.plot(
        [float(point["writes_per_1000_tokens"]) for point in frontier],
        [float(point["delayed_recall_accuracy"]) for point in frontier],
        marker="o",
        label="Pareto frontier",
    )
    for point in points:
        axis.annotate(
            f"lambda={float(point['write_cost_weight']):g}",
            (
                float(point["writes_per_1000_tokens"]),
                float(point["delayed_recall_accuracy"]),
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("writes per 1,000 tokens")
    axis.set_ylabel("delayed-recall accuracy")
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    documents = []
    for path in args.results:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"result document must contain an object: {path}")
        documents.append(value)
    result = aggregate_controller_sweep(documents)
    result["source_results"] = [str(path.resolve()) for path in args.results]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plot_frontier(result, args.output / "controller_pareto_frontier.png")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    print(f"artifacts: {args.output}")


if __name__ == "__main__":
    main()
