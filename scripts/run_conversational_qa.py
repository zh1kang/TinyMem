#!/usr/bin/env python3
"""Fine-tune a WikiText byte model on controlled conversational answers."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from tinymem.data.babi import load_babi_file
from tinymem.data.correction_deletion import generate_update_examples
from tinymem.data.schema import ReasoningExample
from tinymem.data.wikitext import load_wikitext_parquet
from tinymem.evaluation.conversational_qa import (
    ConversationalQAEvaluation,
    evaluate_conversational_qa,
)
from tinymem.evaluation.wikitext_checkpoint import load_wikitext_checkpoint
from tinymem.tokenization.byte_tokenizer import ByteTokenizer
from tinymem.training.checkpointing import save_checkpoint
from tinymem.training.conversational_qa import (
    ByteQAExample,
    ConversationalQATrainingHistory,
    encode_conversational_qa_example,
    train_conversational_qa,
)
from tinymem.training.language_model import (
    LanguageModelEvaluation,
    evaluate_segmented_language_model,
)
from tinymem.utils.device import select_device
from tinymem.utils.experiment import create_run_directory, current_git_commit
from tinymem.utils.seed import seed_everything


TASKS = ("qa1", "qa2", "qa3", "qa4", "qa5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tasks", choices=TASKS, nargs="+", default=TASKS)
    parser.add_argument("--train-examples-per-task", type=int, default=1_000)
    parser.add_argument("--validation-examples-per-task", type=int, default=100)
    parser.add_argument("--update-train-examples", type=int, default=1_000)
    parser.add_argument("--update-validation-examples", type=int, default=200)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--language-model-loss-weight", type=float, default=0.25)
    parser.add_argument("--language-model-sequence-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--max-wikitext-validation-tokens", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="cpu",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/predictions/conversational_qa"),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_examples(
    examples: list[ReasoningExample],
    *,
    count: int,
    seed: int,
) -> list[ReasoningExample]:
    """Select a deterministic sample without changing source examples."""
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be an integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if count <= 0 or count > len(examples):
        raise ValueError("count must select available examples")
    return random.Random(seed).sample(examples, count)


def load_controlled_examples(
    data_root: Path,
    *,
    tasks: tuple[str, ...],
    split: str,
    examples_per_task: int,
    seed: int,
) -> list[ReasoningExample]:
    """Load and deterministically sample official bAbI qa1 through qa5."""
    suffix = "valid" if split == "validation" else split
    selected = []
    for task_index, task in enumerate(tasks):
        examples = load_babi_file(
            data_root / f"{task}_{suffix}.txt",
            task_id=task,
            split=split,
        )
        selected.extend(
            select_examples(
                examples,
                count=examples_per_task,
                seed=seed + task_index,
            )
        )
    return selected


def encode_examples(
    examples: list[ReasoningExample],
    tokenizer: ByteTokenizer,
) -> list[ByteQAExample]:
    return [
        encode_conversational_qa_example(example, tokenizer)
        for example in examples
    ]


def plot_results(
    history: ConversationalQATrainingHistory,
    before: ConversationalQAEvaluation,
    after: ConversationalQAEvaluation,
    wikitext_before: LanguageModelEvaluation,
    wikitext_after: LanguageModelEvaluation,
    destination: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(history.answer_losses, label="answer")
    if history.language_model_losses is not None:
        axes[0].plot(history.language_model_losses, label="WikiText")
    axes[0].set_title("training losses")
    axes[0].legend()
    axes[1].bar(
        ("before", "after"),
        (before.overall.exact_accuracy, after.overall.exact_accuracy),
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_title("controlled exact accuracy")
    axes[2].bar(
        ("before", "after"),
        (wikitext_before.perplexity, wikitext_after.perplexity),
    )
    axes[2].set_title("WikiText validation byte PPL")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    for name in (
        "train_examples_per_task",
        "validation_examples_per_task",
        "update_train_examples",
        "update_validation_examples",
        "steps",
        "batch_size",
        "max_new_tokens",
        "chunk_tokens",
        "max_wikitext_validation_tokens",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    repository_root = Path(__file__).resolve().parents[1]
    checkpoint_path = args.checkpoint.resolve()
    device = select_device(args.device)
    seed_everything(args.seed)
    loaded = load_wikitext_checkpoint(checkpoint_path, device=device)
    if args.chunk_tokens % loaded.selected_window != 0:
        raise ValueError("chunk_tokens must be a multiple of the selected window")
    decoder = loaded.decoder
    tokenizer = ByteTokenizer()

    babi_root = repository_root / "data/raw/tasks_1-20_v1-2/en-valid-10k"
    tasks = tuple(dict.fromkeys(args.tasks))
    train_examples = load_controlled_examples(
        babi_root,
        tasks=tasks,
        split="train",
        examples_per_task=args.train_examples_per_task,
        seed=args.seed,
    )
    validation_examples = load_controlled_examples(
        babi_root,
        tasks=tasks,
        split="validation",
        examples_per_task=args.validation_examples_per_task,
        seed=args.seed + 10_000,
    )
    train_examples.extend(
        generated.example
        for generated in generate_update_examples(
            split="train",
            count=args.update_train_examples,
            base_seed=args.seed,
        )
    )
    validation_examples.extend(
        generated.example
        for generated in generate_update_examples(
            split="validation",
            count=args.update_validation_examples,
            base_seed=args.seed,
        )
    )
    encoded_train = encode_examples(train_examples, tokenizer)
    encoded_validation = encode_examples(validation_examples, tokenizer)

    wikitext_train = load_wikitext_parquet(
        repository_root / "data/raw/wikitext2/train.parquet",
        split="train",
    )
    wikitext_validation = load_wikitext_parquet(
        repository_root / "data/raw/wikitext2/validation.parquet",
        split="validation",
    )
    wikitext_train_ids = torch.tensor(
        tokenizer.encode(wikitext_train.text),
        dtype=torch.long,
    )
    wikitext_validation_ids = torch.tensor(
        tokenizer.encode(wikitext_validation.text),
        dtype=torch.long,
    )

    before = evaluate_conversational_qa(
        decoder,
        encoded_validation,
        device=device,
        max_new_tokens=args.max_new_tokens,
        chunk_tokens=args.chunk_tokens,
    )
    wikitext_before = evaluate_segmented_language_model(
        decoder,
        wikitext_validation_ids,
        sequence_length=512,
        batch_size=args.batch_size,
        pad_id=tokenizer.special_tokens["<pad>"],
        device=device,
        max_tokens=args.max_wikitext_validation_tokens,
    )

    fine_tune_config = replace(
        loaded.config,
        seed=args.seed,
        training=replace(
            loaded.config.training,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            gradient_clip_norm=args.gradient_clip_norm,
            warmup_steps=0,
            max_steps=args.steps,
        ),
    )
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    training_started = time.perf_counter()
    history = train_conversational_qa(
        decoder,
        optimizer,
        encoded_train,
        steps=args.steps,
        batch_size=args.batch_size,
        gradient_clip_norm=args.gradient_clip_norm,
        pad_id=tokenizer.special_tokens["<pad>"],
        device=device,
        seed=args.seed,
        language_model_token_ids=wikitext_train_ids,
        language_model_loss_weight=args.language_model_loss_weight,
        language_model_sequence_length=args.language_model_sequence_length,
    )
    training_seconds = time.perf_counter() - training_started

    after = evaluate_conversational_qa(
        decoder,
        encoded_validation,
        device=device,
        max_new_tokens=args.max_new_tokens,
        chunk_tokens=args.chunk_tokens,
    )
    wikitext_after = evaluate_segmented_language_model(
        decoder,
        wikitext_validation_ids,
        sequence_length=512,
        batch_size=args.batch_size,
        pad_id=tokenizer.special_tokens["<pad>"],
        device=device,
        max_tokens=args.max_wikitext_validation_tokens,
    )

    commit = current_git_commit(repository_root)
    run_directory = create_run_directory(
        repository_root / args.artifact_root,
        fine_tune_config,
        git_commit=commit,
    )
    output_checkpoint = run_directory / "checkpoint.pt"
    save_checkpoint(
        output_checkpoint,
        model=decoder,
        optimizer=optimizer,
        step=args.steps,
        config=fine_tune_config,
        extra={
            "architecture": "segmented_continuous_conversational_qa",
            "selected_window": loaded.selected_window,
            "training_segment_length": loaded.selected_window,
            "tokenizer": "utf8_bytes_v1",
            "parent_checkpoint": str(checkpoint_path),
            "parent_checkpoint_sha256": _sha256(checkpoint_path),
            "training_datasets": [
                f"bAbI {','.join(tasks)}",
                "TinyMem updates",
                "WikiText-2",
            ],
            "longmemeval_training_examples": 0,
        },
    )
    result_document = {
        "status": "development_single_seed_conversational_qa",
        "git_commit": commit,
        "seed": args.seed,
        "device": str(device),
        "parent_checkpoint": str(checkpoint_path),
        "parent_checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint": str(output_checkpoint),
        "checkpoint_sha256": _sha256(output_checkpoint),
        "tasks": list(tasks),
        "train_example_count": len(encoded_train),
        "validation_example_count": len(encoded_validation),
        "update_train_examples": args.update_train_examples,
        "update_validation_examples": args.update_validation_examples,
        "longmemeval_training_examples": 0,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "language_model_loss_weight": args.language_model_loss_weight,
        "training_seconds": training_seconds,
        "final_total_loss": history.total_losses[-1],
        "final_answer_loss": history.answer_losses[-1],
        "final_language_model_loss": (
            history.language_model_losses[-1]
            if history.language_model_losses is not None
            else None
        ),
        "controlled_before": before.to_dict(),
        "controlled_after": after.to_dict(),
        "wikitext_validation_before": wikitext_before.to_dict(),
        "wikitext_validation_after": wikitext_after.to_dict(),
        "training_history": history.to_dict(),
    }
    (run_directory / "results.json").write_text(
        json.dumps(result_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_results(
        history,
        before,
        after,
        wikitext_before,
        wikitext_after,
        run_directory / "conversational_qa.png",
    )
    print(
        json.dumps(
            {
                "status": result_document["status"],
                "train_example_count": len(encoded_train),
                "validation_example_count": len(encoded_validation),
                "controlled_before": before.overall.to_dict(),
                "controlled_after": after.overall.to_dict(),
                "wikitext_validation_before": wikitext_before.to_dict(),
                "wikitext_validation_after": wikitext_after.to_dict(),
                "training_seconds": training_seconds,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"artifacts: {run_directory}")


if __name__ == "__main__":
    main()
