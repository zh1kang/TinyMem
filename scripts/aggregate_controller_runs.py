#!/usr/bin/env python3
"""Aggregate a selected adaptive write protocol across training seeds."""

import argparse
import json
from pathlib import Path

from tinymem.evaluation.controller_sweep import aggregate_controller_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = []
    for path in args.results:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"result document must contain an object: {path}")
        documents.append(value)
    result = aggregate_controller_seeds(documents)
    result["source_results"] = [str(path.resolve()) for path in args.results]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
