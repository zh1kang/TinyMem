#!/usr/bin/env python3
"""Locate retrieval and generation failures with LongMemEval oracle controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from tinymem.data.longmemeval import LongMemEvalExample, load_longmemeval_file
from tinymem.evaluation.longmemeval_diagnostics import (
    DIAGNOSTIC_CONDITIONS,
    LongMemEvalDiagnosticResult,
    evaluate_longmemeval_diagnostics,
)
from tinymem.evaluation.wikitext_checkpoint import load_wikitext_checkpoint
from tinymem.utils.device import select_device
from tinymem.utils.experiment import create_run_directory, current_git_commit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--max-examples", type=int)
    selection.add_argument("--examples-per-type", type=int)
    selection.add_argument("--full", action="store_true")
    parser.add_argument(
        "--conditions",
        choices=DIAGNOSTIC_CONDITIONS,
        nargs="+",
        default=DIAGNOSTIC_CONDITIONS,
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="cpu",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/predictions/longmemeval_diagnostics"),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_examples(
    examples: list[LongMemEvalExample],
    *,
    max_examples: int | None,
    examples_per_type: int | None,
    full: bool,
) -> list[LongMemEvalExample]:
    """Select a stable source-order diagnostic slice."""
    if max_examples is not None:
        if max_examples < 2:
            raise ValueError("max_examples must be at least 2")
        return examples[:max_examples]
    if full:
        return examples
    per_type = 1 if examples_per_type is None else examples_per_type
    if per_type <= 0:
        raise ValueError("examples_per_type must be positive")
    counts: Counter[str] = Counter()
    selected = []
    for example in examples:
        if counts[example.question_type] < per_type:
            selected.append(example)
            counts[example.question_type] += 1
    return selected


def plot_results(
    results: tuple[LongMemEvalDiagnosticResult, ...],
    destination: Path,
) -> None:
    names = [result.condition for result in results]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].bar(names, [result.answer_byte_nll for result in results])
    axes[0, 0].set_ylabel("answer NLL per byte")
    axes[0, 1].bar(names, [result.mean_nll_margin for result in results])
    axes[0, 1].axhline(0.0, color="black", linewidth=1)
    axes[0, 1].set_ylabel("wrong NLL minus answer NLL")
    axes[1, 0].bar(names, [result.answer_preference_rate for result in results])
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_ylabel("answer preference rate")
    axes[1, 1].bar(
        names,
        [result.generated_mean_token_f1 for result in results],
    )
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_ylabel("generated mean token F1")
    for axis in axes.flat:
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    checkpoint_path = args.checkpoint.resolve()
    device = select_device(args.device)
    loaded = load_wikitext_checkpoint(checkpoint_path, device=device)
    if args.chunk_tokens % loaded.selected_window != 0:
        raise ValueError("chunk_tokens must be a multiple of the selected window")

    dataset_path = repository_root / (
        "data/raw/longmemeval/longmemeval_oracle.json"
    )
    all_examples = load_longmemeval_file(dataset_path)
    examples = select_examples(
        all_examples,
        max_examples=args.max_examples,
        examples_per_type=args.examples_per_type,
        full=args.full,
    )
    results = evaluate_longmemeval_diagnostics(
        loaded.decoder,
        examples,
        device=device,
        max_new_tokens=args.max_new_tokens,
        chunk_tokens=args.chunk_tokens,
        conditions=args.conditions,
    )

    lock = json.loads(
        (repository_root / "data/installed.lock.json").read_text(
            encoding="utf-8"
        )
    )
    installed_file = lock["files"][
        "longmemeval:longmemeval/longmemeval_oracle.json"
    ]
    commit = current_git_commit(repository_root)
    run_directory = create_run_directory(
        repository_root / args.artifact_root,
        loaded.config,
        git_commit=commit,
    )
    result_document = {
        "status": "development_oracle_diagnostic",
        "dataset": "LongMemEval oracle",
        "dataset_revision": "98d7416c24c778c2fee6e6f3006e7a073259d48f",
        "dataset_sha256": installed_file["sha256"],
        "dataset_sha256_verified": _sha256(dataset_path)
        == installed_file["sha256"],
        "manifest_sha256": lock["manifest_sha256"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "git_commit": commit,
        "device": str(device),
        "example_count": len(examples),
        "source_example_count": len(all_examples),
        "conditions": list(dict.fromkeys(args.conditions)),
        "max_new_tokens": args.max_new_tokens,
        "chunk_tokens": args.chunk_tokens,
        "training_on_longmemeval": False,
        "answers_used_as_scoring_targets": True,
        "reference_inserted_only_in_copy_condition": True,
        "results": [result.to_dict() for result in results],
    }
    (run_directory / "results.json").write_text(
        json.dumps(result_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_results(results, run_directory / "longmemeval_diagnostics.png")
    print(
        json.dumps(
            {
                "status": result_document["status"],
                "example_count": len(examples),
                "results": {
                    result.condition: {
                        "count": result.count,
                        "source_count": result.source_count,
                        "skipped_missing_answer_message_labels": (
                            result.skipped_missing_answer_message_labels
                        ),
                        "generated_exact_accuracy": (
                            result.generated_exact_accuracy
                        ),
                        "generated_mean_token_f1": (
                            result.generated_mean_token_f1
                        ),
                        "answer_byte_nll": result.answer_byte_nll,
                        "mean_nll_margin": result.mean_nll_margin,
                        "answer_preference_rate": (
                            result.answer_preference_rate
                        ),
                        "first_byte_accuracy": result.first_byte_accuracy,
                    }
                    for result in results
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"artifacts: {run_directory}")


if __name__ == "__main__":
    main()
