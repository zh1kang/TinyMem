#!/usr/bin/env python3
"""Run the controlled content-aware memory replacement experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
import torch
from torch import nn

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from tinymem.data.replacement_qa import generate_replacement_qa_examples
from tinymem.evaluation.replacement_qa import (
    ReplacementQAEvaluation,
    evaluate_replacement_qa,
)
from tinymem.evaluation.wikitext_checkpoint import load_wikitext_checkpoint
from tinymem.memory.continuous import MeanPoolMemoryCompressor
from tinymem.memory.recurrent_memory import RecurrentMemoryBank
from tinymem.memory.replacement import SlotReplacementController
from tinymem.tokenization.byte_tokenizer import ByteTokenizer
from tinymem.training.checkpointing import save_checkpoint
from tinymem.training.replacement_qa import (
    ReplacementQATrainingHistory,
    train_replacement_qa,
)
from tinymem.utils.device import select_device
from tinymem.utils.experiment import create_run_directory, current_git_source_state
from tinymem.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--segment-length", type=int, default=64)
    parser.add_argument("--memory-capacity", type=int, default=3)
    parser.add_argument("--train-examples", type=int, default=999)
    parser.add_argument("--validation-examples", type=int, default=99)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--slot-pretrain-steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--replacement-loss-weight", type=float, default=1.0)
    parser.add_argument("--semantic-prefix-bytes", type=int, default=2)
    parser.add_argument("--semantic-prefix-weight", type=float, default=8.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--validation-seed", type=int, default=10_000)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/predictions/replacement_qa"),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _passes_gate(result: ReplacementQAEvaluation) -> bool:
    normal = result.conditions["normal"]
    return (
        result.replacement_slot_accuracy >= 0.9
        and normal.exact_accuracy >= 0.8
        and normal.corrected_query_accuracy >= 0.8
        and normal.unchanged_query_accuracy >= 0.8
        and normal.exact_accuracy
        - result.conditions["drop"].exact_accuracy
        >= 0.3
        and normal.exact_accuracy
        - result.conditions["zero"].exact_accuracy
        >= 0.3
        and normal.exact_accuracy
        - result.conditions["shuffle"].exact_accuracy
        >= 0.3
        and normal.corrected_query_accuracy
        - result.conditions["frozen"].corrected_query_accuracy
        >= 0.3
    )


def _plot(
    history: ReplacementQATrainingHistory,
    result: ReplacementQAEvaluation,
    destination: Path,
) -> None:
    losses = history.losses
    answer_losses = history.answer_losses
    replacement_losses = history.replacement_losses
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(losses, label="total")
    axes[0].plot(answer_losses, label="answer", alpha=0.75)
    axes[0].plot(replacement_losses, label="replacement", alpha=0.75)
    axes[0].set_title("training loss")
    axes[0].set_xlabel("step")
    axes[0].legend()
    conditions = tuple(result.conditions)
    axes[1].bar(
        conditions,
        tuple(result.conditions[name].exact_accuracy for name in conditions),
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_title("generated exact accuracy")
    axes[1].tick_params(axis="x", rotation=30)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    for name in (
        "segment_length",
        "memory_capacity",
        "train_examples",
        "validation_examples",
        "steps",
        "batch_size",
        "semantic_prefix_bytes",
        "max_new_tokens",
        "progress_every",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0 <= args.slot_pretrain_steps < args.steps:
        raise ValueError("slot_pretrain_steps must be in [0, steps)")
    if args.learning_rate <= 0 or args.gradient_clip_norm <= 0:
        raise ValueError("learning rate and gradient clip norm must be positive")
    if args.weight_decay < 0 or args.replacement_loss_weight < 0:
        raise ValueError("weight decay and replacement loss weight must be nonnegative")
    if args.semantic_prefix_weight < 1:
        raise ValueError("semantic_prefix_weight must be at least one")
    balance_period = args.memory_capacity**2
    if (
        args.train_examples % balance_period != 0
        or args.validation_examples % balance_period != 0
    ):
        raise ValueError("example counts must be multiples of memory_capacity squared")
    if args.seed < 0 or args.validation_seed < 0:
        raise ValueError("seeds must be nonnegative")

    repository_root = Path(__file__).resolve().parents[1]
    checkpoint_path = args.checkpoint.resolve()
    device = select_device(args.device)
    seed_everything(args.seed)
    loaded = load_wikitext_checkpoint(checkpoint_path, device=device)
    if args.segment_length > loaded.config.model.max_local_tokens:
        raise ValueError("segment_length exceeds the model local window")
    if (args.memory_capacity + 2) * args.segment_length > (
        loaded.config.model.max_local_tokens
    ):
        raise ValueError("the full correction episode exceeds positional capacity")
    if not isinstance(loaded.decoder.compressor, MeanPoolMemoryCompressor):
        raise ValueError("replacement QA requires the mean-pool compressor")

    decoder = loaded.decoder
    decoder.segment_length = args.segment_length
    decoder.bank = RecurrentMemoryBank(
        capacity=args.memory_capacity,
        model_width=decoder.model.config.d_model,
    ).to(device)
    decoder.memory_position_mode = "virtual"
    controller = SlotReplacementController(
        decoder.model.config.d_model,
        temperature=args.temperature,
    ).to(device)
    tokenizer = ByteTokenizer()
    train_examples = generate_replacement_qa_examples(
        tokenizer,
        split="train",
        count=args.train_examples,
        memory_capacity=args.memory_capacity,
        segment_length=args.segment_length,
        base_seed=args.seed,
    )
    validation_examples = generate_replacement_qa_examples(
        tokenizer,
        split="validation",
        count=args.validation_examples,
        memory_capacity=args.memory_capacity,
        segment_length=args.segment_length,
        base_seed=args.validation_seed,
    )
    optimizer = torch.optim.AdamW(
        (*decoder.parameters(), *controller.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    started = time.perf_counter()

    def report_progress(step: int, loss: float) -> None:
        if step % args.progress_every == 0 or step == args.steps:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": round(loss, 5),
                        "elapsed_seconds": round(
                            time.perf_counter() - started,
                            1,
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    history = train_replacement_qa(
        decoder,
        controller,
        optimizer,
        train_examples,
        steps=args.steps,
        slot_pretrain_steps=args.slot_pretrain_steps,
        batch_size=args.batch_size,
        replacement_loss_weight=args.replacement_loss_weight,
        semantic_prefix_bytes=args.semantic_prefix_bytes,
        semantic_prefix_weight=args.semantic_prefix_weight,
        gradient_clip_norm=args.gradient_clip_norm,
        pad_id=tokenizer.special_tokens["<pad>"],
        device=device,
        seed=args.seed,
        on_step=report_progress,
    )
    training_seconds = time.perf_counter() - started
    result = evaluate_replacement_qa(
        decoder,
        controller,
        validation_examples,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )

    experiment_config = replace(
        loaded.config,
        seed=args.seed,
        stream=replace(
            loaded.config.stream,
            segment_length=args.segment_length,
        ),
        memory=replace(loaded.config.memory, n_slots=args.memory_capacity),
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
    source_state = current_git_source_state(repository_root)
    run_directory = create_run_directory(
        repository_root
        / args.artifact_root
        / f"capacity_{args.memory_capacity}"
        / f"seed_{args.seed}",
        experiment_config,
        git_commit=source_state.commit,
        source_state=source_state,
    )
    system = nn.ModuleDict({"decoder": decoder, "controller": controller})
    checkpoint_output = run_directory / "checkpoint.pt"
    save_checkpoint(
        checkpoint_output,
        model=system,
        optimizer=optimizer,
        step=args.steps,
        config=experiment_config,
        extra={
            "architecture": "content_aware_replacement_qa",
            "parent_checkpoint": str(checkpoint_path),
            "parent_checkpoint_sha256": _sha256(checkpoint_path),
            "segment_length": args.segment_length,
            "memory_capacity": args.memory_capacity,
            "memory_position_mode": "virtual",
            "compressor": "mean",
            "replacement_controller": "pairwise_mlp",
            "temperature": args.temperature,
            "slot_pretrain_steps": args.slot_pretrain_steps,
            "replacement_loss_weight": args.replacement_loss_weight,
            "semantic_prefix_bytes": args.semantic_prefix_bytes,
            "semantic_prefix_weight": args.semantic_prefix_weight,
            "answer_only_loss": True,
        },
    )
    result_document = {
        "status": "development_content_aware_replacement",
        "gate_passed": _passes_gate(result),
        "seed": args.seed,
        "validation_seed": args.validation_seed,
        "device": str(device),
        "source_state": source_state.to_dict(),
        "parent_checkpoint": str(checkpoint_path),
        "parent_checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint": str(checkpoint_output),
        "checkpoint_sha256": _sha256(checkpoint_output),
        "segment_length": args.segment_length,
        "memory_capacity": args.memory_capacity,
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "steps": args.steps,
        "slot_pretrain_steps": args.slot_pretrain_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "replacement_loss_weight": args.replacement_loss_weight,
        "semantic_prefix_bytes": args.semantic_prefix_bytes,
        "semantic_prefix_weight": args.semantic_prefix_weight,
        "gradient_clip_norm": args.gradient_clip_norm,
        "temperature": args.temperature,
        "training_seconds": training_seconds,
        "training_history": history.to_dict(),
        "evaluation": result.to_dict(),
    }
    (run_directory / "results.json").write_text(
        json.dumps(result_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot(history, result, run_directory / "replacement_qa.png")
    print(
        json.dumps(
            {
                "gate_passed": result_document["gate_passed"],
                "seed": args.seed,
                "training_seconds": training_seconds,
                "replacement_slot_accuracy": result.replacement_slot_accuracy,
                "conditions": {
                    name: {
                        "exact_accuracy": condition.exact_accuracy,
                        "corrected_query_accuracy": (
                            condition.corrected_query_accuracy
                        ),
                        "unchanged_query_accuracy": (
                            condition.unchanged_query_accuracy
                        ),
                        "first_byte_logit_linf": (
                            condition.mean_first_byte_logit_linf_from_normal
                        ),
                    }
                    for name, condition in result.conditions.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"artifacts: {run_directory}")


if __name__ == "__main__":
    main()
