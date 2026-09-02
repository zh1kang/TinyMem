#!/usr/bin/env python3
"""Aggregate base and memory MTP runs into matched delay curves."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from tinymem.evaluation.mtp_comparison import (
    aggregate_mtp_delay_runs,
    load_mtp_delay_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        action="append",
        required=True,
        metavar="NAME[@SEED]=RESULTS_JSON",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/predictions/mtp_comparison"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = []
    sources = []
    for value in args.condition:
        condition_spec, separator, raw_path = value.partition("=")
        if not separator or not condition_spec or not raw_path:
            raise ValueError(
                "conditions must have the form NAME[@SEED]=RESULTS_JSON"
            )
        name, seed_separator, raw_seed = condition_spec.rpartition("@")
        if not seed_separator:
            name = condition_spec
            seed_override = None
        else:
            if not name or not raw_seed:
                raise ValueError("condition seed override is invalid")
            seed_override = int(raw_seed)
        path = Path(raw_path).resolve()
        runs.append(
            load_mtp_delay_run(
                path,
                condition=name,
                seed_override=seed_override,
            )
        )
        sources.append({"condition": name, "path": str(path)})

    conditions = aggregate_mtp_delay_runs(runs)
    output_directory = args.output_root / datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%S.%fZ"
    )
    output_directory.mkdir(parents=True, exist_ok=False)
    document = {
        "status": "development_comparison",
        "sources": sources,
        "conditions": conditions,
    }
    (output_directory / "results.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for condition in conditions:
        curve = condition["curve"]
        axis.plot(
            [str(bucket["label"]) for bucket in curve],
            [float(bucket["seed_accuracy_mean"]) for bucket in curve],
            marker="o",
            label=str(condition["condition"]),
        )
    axis.set_ylim(0, 1)
    axis.set_xlabel("tokens after answer-bearing evidence")
    axis.set_ylabel("mean exact answer accuracy")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_directory / "delay_comparison.png", dpi=160)
    plt.close(figure)
    print(json.dumps(document, indent=2, sort_keys=True))
    print(f"artifacts: {output_directory}")


if __name__ == "__main__":
    main()
