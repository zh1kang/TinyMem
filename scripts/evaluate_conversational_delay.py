#!/usr/bin/env python3
"""Measure byte-level delayed recall with WikiText filler before the question."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tinymem.data.babi import load_babi_file
from tinymem.data.sampling import select_reasoning_examples
from tinymem.data.schema import ReasoningExample
from tinymem.data.wikitext import load_wikitext_parquet
from tinymem.evaluation.conversational_delay import build_delayed_examples
from tinymem.evaluation.conversational_qa import (
    MEMORY_CONDITIONS,
    evaluate_conversational_qa,
    resolve_memory_condition,
)
from tinymem.evaluation.wikitext_checkpoint import load_wikitext_checkpoint
from tinymem.tokenization.byte_tokenizer import ByteTokenizer
from tinymem.utils.device import select_device
from tinymem.utils.experiment import create_run_directory, current_git_source_state
from tinymem.utils.seed import seed_everything


TASKS = ("qa1", "qa2", "qa3", "qa4", "qa5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tasks", choices=TASKS, nargs="+", default=("qa1",))
    parser.add_argument("--examples-per-task", type=int, default=500)
    parser.add_argument(
        "--delays",
        type=int,
        nargs="+",
        default=(0, 64, 128, 256, 512, 1024, 2048),
        help="filler bytes inserted between the facts and the question",
    )
    parser.add_argument(
        "--conditions",
        choices=MEMORY_CONDITIONS,
        nargs="+",
        default=("normal", "drop_at_query"),
    )
    parser.add_argument(
        "--filler-split",
        choices=("validation", "test"),
        default="validation",
    )
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
        default=Path("artifacts/predictions/conversational_qa_delay"),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_test_examples(
    data_root: Path,
    *,
    tasks: tuple[str, ...],
    examples_per_task: int,
    seed: int,
) -> list[ReasoningExample]:
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
    for name in ("examples_per_task", "max_new_tokens", "chunk_tokens"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if any(delay < 0 for delay in args.delays):
        raise ValueError("delays must be nonnegative")
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")

    repository_root = Path(__file__).resolve().parents[1]
    checkpoint_path = args.checkpoint.resolve()
    device = select_device(args.device)
    seed_everything(args.seed)
    loaded = load_wikitext_checkpoint(checkpoint_path, device=device)
    if args.chunk_tokens % loaded.selected_window != 0:
        raise ValueError("chunk_tokens must be a multiple of the selected window")

    tasks = tuple(dict.fromkeys(args.tasks))
    delays = tuple(dict.fromkeys(int(delay) for delay in args.delays))
    conditions = tuple(dict.fromkeys(args.conditions))
    examples = load_test_examples(
        repository_root / "data/raw/tasks_1-20_v1-2/en-valid-10k",
        tasks=tasks,
        examples_per_task=args.examples_per_task,
        seed=args.seed,
    )
    filler_path = repository_root / f"data/raw/wikitext2/{args.filler_split}.parquet"
    filler_text = load_wikitext_parquet(filler_path, split=args.filler_split).text
    tokenizer = ByteTokenizer()

    curve: dict[str, dict[str, object]] = {}
    for delay in delays:
        delayed = build_delayed_examples(
            examples,
            tokenizer,
            filler_text=filler_text,
            filler_bytes=delay,
            seed=args.seed + delay,
        )
        for condition in conditions:
            intervention, update_memory = resolve_memory_condition(condition)
            evaluation = evaluate_conversational_qa(
                loaded.decoder,
                delayed,
                device=device,
                max_new_tokens=args.max_new_tokens,
                chunk_tokens=args.chunk_tokens,
                query_memory_intervention=intervention,
                update_memory=update_memory,
            )
            curve.setdefault(condition, {})[str(delay)] = evaluation.to_dict()
            print(
                json.dumps(
                    {
                        "condition": condition,
                        "delay_bytes": delay,
                        "exact_accuracy": evaluation.overall.exact_accuracy,
                        "first_byte_accuracy": (
                            evaluation.overall.first_byte_accuracy
                        ),
                    }
                ),
                flush=True,
            )

    source_state = current_git_source_state(repository_root)
    commit = source_state.commit
    run_directory = create_run_directory(
        repository_root / args.artifact_root,
        loaded.config,
        git_commit=commit,
        source_state=source_state,
    )
    result_document = {
        "status": "heldout_controlled_delay_sweep_single_seed",
        "git_commit": commit,
        "source_state": source_state.to_dict(),
        "seed": args.seed,
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "architecture": loaded.architecture,
        "selected_window": loaded.selected_window,
        "memory": loaded.memory_spec.to_metadata(),
        "tasks": list(tasks),
        "examples_per_task": args.examples_per_task,
        "example_count": len(examples),
        "delays": list(delays),
        "conditions": list(conditions),
        "filler_split": args.filler_split,
        "filler_sha256": _sha256(filler_path),
        "curve": curve,
    }
    (run_directory / "results.json").write_text(
        json.dumps(result_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"artifacts: {run_directory}")


if __name__ == "__main__":
    main()
