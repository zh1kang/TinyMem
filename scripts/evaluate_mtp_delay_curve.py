#!/usr/bin/env python3
"""Evaluate one continuous-memory checkpoint by BABILong query delay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tinymem.data.babilong import load_babilong_file
from tinymem.evaluation.continuous_checkpoint import load_continuous_checkpoint
from tinymem.evaluation.continuous_memory import evaluate_continuous_qa1
from tinymem.utils.device import select_device
from tinymem.utils.experiment import create_run_directory, current_git_commit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--examples-per-context", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/predictions/mtp_delay_curve"),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 1:
        raise ValueError("batch size must be greater than one")
    if args.examples_per_context < 0:
        raise ValueError("examples per context must be nonnegative")

    repository_root = Path(__file__).resolve().parents[1]
    checkpoint_path = args.checkpoint.resolve()
    device = select_device(args.device)
    loaded = load_continuous_checkpoint(checkpoint_path, device=device)
    examples = []
    context_counts: dict[str, int] = {}
    for context_length in ("1k", "2k", "4k", "8k"):
        context_examples = load_babilong_file(
            repository_root / f"data/raw/babilong/qa1/{context_length}.json",
            task_id="qa1",
            split="test",
        )
        if args.examples_per_context:
            context_examples = context_examples[: args.examples_per_context]
        context_counts[context_length] = len(context_examples)
        examples.extend(context_examples)

    result = evaluate_continuous_qa1(
        loaded.decoder,
        loaded.vocabulary,
        examples,
        batch_size=args.batch_size,
        device=device,
        intervention_name="normal",
    )
    commit = current_git_commit(repository_root)
    run_directory = create_run_directory(
        repository_root / args.artifact_root,
        loaded.config,
        git_commit=commit,
    )
    document = {
        "status": "development_single_seed",
        "device": str(device),
        "git_commit": commit,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_step": loaded.step,
        "architecture": loaded.architecture,
        "seed": loaded.config.seed,
        "mtp_horizons": (
            list(loaded.config.mtp.horizons)
            if loaded.config.mtp.enabled
            else []
        ),
        "mtp_loss_weight": (
            loaded.config.mtp.loss_weight if loaded.config.mtp.enabled else 0.0
        ),
        "examples_per_context": context_counts,
        "evaluation": result.to_dict(),
    }
    (run_directory / "results.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(document, indent=2, sort_keys=True))
    print(f"artifacts: {run_directory}")


if __name__ == "__main__":
    main()
