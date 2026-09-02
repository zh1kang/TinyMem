#!/usr/bin/env python3
"""Compare matched continuous and discrete memory result documents."""

import argparse
import json
from pathlib import Path

from tinymem.evaluation.discrete_comparison import (
    compare_discrete_to_continuous,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous-results", type=Path, required=True)
    parser.add_argument("--discrete-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_document(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"result document does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"result document must contain an object: {path}")
    return value


def main() -> None:
    args = parse_args()
    result = compare_discrete_to_continuous(
        _load_document(args.continuous_results),
        _load_document(args.discrete_results),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
