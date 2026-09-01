#!/usr/bin/env python3
"""Aggregate compatible fixed-memory benchmark result files."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

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
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    print(f"artifacts: {run_directory}")


if __name__ == "__main__":
    main()
