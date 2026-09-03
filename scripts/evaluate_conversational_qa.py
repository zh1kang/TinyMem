#!/usr/bin/env python3
"""Evaluate a frozen conversational byte model on untouched controlled tests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tinymem.data.babi import load_babi_file
from tinymem.data.correction_deletion import generate_update_examples
from tinymem.data.sampling import select_reasoning_examples
from tinymem.data.schema import ReasoningExample
from tinymem.evaluation.conversational_qa import evaluate_conversational_qa
from tinymem.evaluation.wikitext_checkpoint import load_wikitext_checkpoint
from tinymem.tokenization.byte_tokenizer import ByteTokenizer
from tinymem.training.conversational_qa import encode_conversational_qa_example
from tinymem.utils.device import select_device
from tinymem.utils.experiment import create_run_directory, current_git_commit
from tinymem.utils.seed import seed_everything


TASKS = ("qa1", "qa2", "qa3", "qa4", "qa5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tasks", choices=TASKS, nargs="+", default=TASKS)
    parser.add_argument("--examples-per-task", type=int)
    parser.add_argument("--update-test-examples", type=int, default=1_000)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="cpu",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/predictions/conversational_qa_holdout"),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_babi_test_examples(
    data_root: Path,
    *,
    tasks: tuple[str, ...],
    examples_per_task: int | None,
    seed: int,
) -> list[ReasoningExample]:
    """Load full official tests or deterministic reporting subsets."""
    selected = []
    for task_index, task in enumerate(tasks):
        examples = load_babi_file(
            data_root / f"{task}_test.txt",
            task_id=task,
            split="test",
        )
        selected.extend(
            select_reasoning_examples(
                examples,
                count=examples_per_task,
                seed=seed + task_index,
            )
        )
    return selected


def main() -> None:
    args = parse_args()
    for name in ("update_test_examples", "max_new_tokens", "chunk_tokens"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.examples_per_task is not None and args.examples_per_task <= 0:
        raise ValueError("examples_per_task must be positive or omitted")
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")

    repository_root = Path(__file__).resolve().parents[1]
    checkpoint_path = args.checkpoint.resolve()
    device = select_device(args.device)
    seed_everything(args.seed)
    loaded = load_wikitext_checkpoint(checkpoint_path, device=device)
    if loaded.architecture != "segmented_continuous_conversational_qa":
        raise ValueError("checkpoint is not a conversational QA model")
    if args.chunk_tokens % loaded.selected_window != 0:
        raise ValueError("chunk_tokens must be a multiple of the selected window")

    tasks = tuple(dict.fromkeys(args.tasks))
    examples = load_babi_test_examples(
        repository_root / "data/raw/tasks_1-20_v1-2/en-valid-10k",
        tasks=tasks,
        examples_per_task=args.examples_per_task,
        seed=args.seed,
    )
    examples.extend(
        generated.example
        for generated in generate_update_examples(
            split="test",
            count=args.update_test_examples,
            base_seed=args.seed,
        )
    )
    tokenizer = ByteTokenizer()
    encoded = [
        encode_conversational_qa_example(example, tokenizer)
        for example in examples
    ]
    evaluation = evaluate_conversational_qa(
        loaded.decoder,
        encoded,
        device=device,
        max_new_tokens=args.max_new_tokens,
        chunk_tokens=args.chunk_tokens,
    )

    commit = current_git_commit(repository_root)
    run_directory = create_run_directory(
        repository_root / args.artifact_root,
        loaded.config,
        git_commit=commit,
    )
    result_document = {
        "status": "heldout_controlled_test_single_seed",
        "git_commit": commit,
        "seed": args.seed,
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "architecture": loaded.architecture,
        "tasks": list(tasks),
        "babi_examples_per_task": args.examples_per_task,
        "update_test_examples": args.update_test_examples,
        "example_count": len(encoded),
        "longmemeval_examples": 0,
        "evaluation": evaluation.to_dict(),
    }
    (run_directory / "results.json").write_text(
        json.dumps(result_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result_document["status"],
                "example_count": len(encoded),
                "overall": evaluation.overall.to_dict(),
                "by_task": {
                    name: result.to_dict()
                    for name, result in evaluation.by_task.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"artifacts: {run_directory}")


if __name__ == "__main__":
    main()
