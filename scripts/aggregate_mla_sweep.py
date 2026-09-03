#!/usr/bin/env python3
"""Aggregate matched MHA and MLA-lite runs and plot quality versus cache bytes."""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from tinymem.evaluation.mla_sweep import aggregate_mla_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/predictions/mla_sweep"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.results
    ]
    points = aggregate_mla_sweep(records)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = args.artifact_root / f"{timestamp}-{uuid.uuid4().hex[:8]}"
    destination.mkdir(parents=True, exist_ok=False)

    serialized = [point.to_dict() for point in points]
    (destination / "results.json").write_text(
        json.dumps(serialized, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = [
        "MHA" if point.kv_latent_dim is None else f"MLA C={point.kv_latent_dim}"
        for point in points
    ]
    cache_kib = [point.cache_bytes / 1024 for point in points]
    axes[0].plot(
        cache_kib,
        [point.validation_accuracy_mean for point in points],
        marker="o",
    )
    axes[0].set_ylabel("validation answer accuracy")
    axes[1].plot(
        cache_kib,
        [point.delayed_accuracy_mean for point in points],
        marker="o",
    )
    axes[1].set_ylabel("BABILong delayed accuracy")
    for axis, values in zip(
        axes,
        (
            [point.validation_accuracy_mean for point in points],
            [point.delayed_accuracy_mean for point in points],
        ),
        strict=True,
    ):
        axis.set_xlabel("actual cache storage (KiB)")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        for label, x_value, y_value in zip(labels, cache_kib, values, strict=True):
            axis.annotate(
                label,
                (x_value, y_value),
                xytext=(4, 5),
                textcoords="offset points",
            )
    figure.suptitle("MLA-lite quality versus measured KV-cache storage")
    figure.tight_layout()
    figure.savefig(destination / "quality_vs_cache.png", dpi=160)
    plt.close(figure)

    print(json.dumps(serialized, indent=2, sort_keys=True))
    print(f"artifacts: {destination}")


if __name__ == "__main__":
    main()
