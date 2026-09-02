#!/usr/bin/env python3
"""Evaluate one frozen continuous-memory checkpoint on one BABILong file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tinymem.data.babilong import load_babilong_file
from tinymem.evaluation.continuous_checkpoint import (
    load_continuous_checkpoint,
)
from tinymem.evaluation.continuous_memory import (
    drop_memory,
    evaluate_continuous_qa1,
    paired_accuracy_test,
    shuffle_memory,
    zero_memory,
)
from tinymem.memory.discrete_compressor import DiscreteMemoryCompressor
from tinymem.utils.device import select_device
from tinymem.utils.experiment import create_run_directory, current_git_commit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--codebook-evaluation-mode",
        choices=("hard", "soft"),
        default="hard",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/predictions/continuous_memory_heldout"),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _training_commit(checkpoint_path: Path) -> str:
    run_path = checkpoint_path.parent / "run.json"
    if not run_path.is_file():
        raise FileNotFoundError(f"checkpoint run metadata does not exist: {run_path}")
    document = json.loads(run_path.read_text(encoding="utf-8"))
    commit = document.get("git_commit")
    if not isinstance(commit, str) or not commit:
        raise ValueError("checkpoint run metadata has no Git commit")
    return commit


def main() -> None:
    args = parse_args()
    if isinstance(args.batch_size, bool) or args.batch_size <= 1:
        raise ValueError("batch size must be an integer greater than one")

    repository_root = Path(__file__).resolve().parents[1]
    checkpoint_path = args.checkpoint.resolve()
    data_path = args.data_file.resolve()
    device = select_device(args.device)
    loaded = load_continuous_checkpoint(checkpoint_path, device=device)
    is_discrete = isinstance(
        loaded.decoder.compressor,
        DiscreteMemoryCompressor,
    )
    if is_discrete:
        loaded.decoder.compressor.set_evaluation_mode(
            args.codebook_evaluation_mode
        )
    elif args.codebook_evaluation_mode != "hard":
        raise ValueError("soft codebook evaluation requires a discrete checkpoint")
    examples = load_babilong_file(
        data_path,
        task_id="qa1",
        split="test",
    )

    interventions = (
        ("normal", None),
        ("drop", drop_memory),
        ("zero", zero_memory),
        ("shuffle", shuffle_memory),
    )
    evaluations = []
    for name, intervention in interventions:
        print(f"evaluating {name} memory...", flush=True)
        evaluations.append(
            evaluate_continuous_qa1(
                loaded.decoder,
                loaded.vocabulary,
                examples,
                batch_size=args.batch_size,
                device=device,
                intervention_name=name,
                memory_intervention=intervention,
            )
        )

    normal = evaluations[0]
    comparison_alpha = 0.05 / (len(interventions) - 1)
    utility = {
        result.intervention: (
            normal.outside_window_accuracy - result.outside_window_accuracy
        )
        for result in evaluations[1:]
    }
    counterfactual_tests = {
        result.intervention: paired_accuracy_test(
            normal.outside_window_correctness,
            result.outside_window_correctness,
            alpha=comparison_alpha,
        )
        for result in evaluations[1:]
    }
    exit_criteria_met = all(
        result.significant for result in counterfactual_tests.values()
    )
    evaluation_commit = current_git_commit(repository_root)
    run_directory = create_run_directory(
        repository_root / args.artifact_root,
        loaded.config,
        git_commit=evaluation_commit,
    )
    manifest = json.loads(
        (repository_root / "data/manifest.json").read_text(encoding="utf-8")
    )
    revision = manifest["datasets"]["babilong"]["revision"]
    result_document = {
        "status": "held_out_single_seed",
        "task_id": "qa1",
        "context_length": data_path.stem,
        "device": str(device),
        "seed": loaded.config.seed,
        "training_step": loaded.step,
        "training_git_commit": _training_commit(checkpoint_path),
        "evaluation_git_commit": evaluation_commit,
        "architecture": loaded.architecture,
        "codebook_evaluation_mode": (
            args.codebook_evaluation_mode if is_discrete else None
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "dataset_file": str(data_path),
        "dataset_revision": revision,
        "dataset_sha256": _sha256(data_path),
        "evaluations": [result.to_dict() for result in evaluations],
        "outside_window_counterfactual_utility": utility,
        "outside_window_counterfactual_tests": {
            name: result.to_dict()
            for name, result in counterfactual_tests.items()
        },
        "exit_criteria_met": exit_criteria_met,
    }
    (run_directory / "results.json").write_text(
        json.dumps(result_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result_document, indent=2, sort_keys=True))
    print(f"artifacts: {run_directory}")


if __name__ == "__main__":
    main()
