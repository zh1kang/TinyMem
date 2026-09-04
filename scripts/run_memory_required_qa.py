#!/usr/bin/env python3
"""Run the gated oracle-reader or learned-writer byte memory experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from tinymem.data.babi import load_babi_file
from tinymem.data.memory_required_qa import (
    DISTRACTOR_TEXT_BANKS,
    build_memory_required_qa_examples,
)
from tinymem.data.sampling import select_reasoning_examples
from tinymem.evaluation.memory_required_qa import (
    MemoryRequiredQAConditionResult,
    evaluate_memory_required_qa,
    evaluate_memory_required_write_gate,
)
from tinymem.evaluation.wikitext_checkpoint import load_wikitext_checkpoint
from tinymem.memory.continuous import MeanPoolMemoryCompressor
from tinymem.memory.recurrent_memory import (
    GatedRecurrentMemoryBank,
    RecurrentMemoryBank,
)
from tinymem.memory.write_gate import TokenSegmentWriteGate
from tinymem.tokenization.byte_tokenizer import ByteTokenizer
from tinymem.training.checkpointing import save_checkpoint
from tinymem.training.memory_required_qa import (
    audit_cross_segment_gradients,
    precompute_oracle_memories,
    train_memory_required_qa,
)
from tinymem.utils.device import select_device
from tinymem.utils.experiment import create_run_directory, current_git_source_state
from tinymem.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("oracle", "learned", "gated"),
        required=True,
    )
    parser.add_argument("--segment-length", type=int, default=512)
    parser.add_argument("--train-examples", type=int, default=1_000)
    parser.add_argument("--validation-examples", type=int, default=100)
    parser.add_argument("--distractor-segments", type=int, default=0)
    parser.add_argument(
        "--train-distractor-variant",
        choices=tuple(DISTRACTOR_TEXT_BANKS),
        default="trained",
    )
    parser.add_argument(
        "--validation-distractor-variant",
        choices=tuple(DISTRACTOR_TEXT_BANKS),
        default="trained",
    )
    parser.add_argument("--memory-capacity", type=int)
    parser.add_argument(
        "--distractor-write-policy",
        choices=("all", "none", "learned"),
        default="all",
    )
    parser.add_argument("--write-loss-weight", type=float, default=1.0)
    parser.add_argument("--write-gate-kernel-size", type=int, default=3)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
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
        default=Path("artifacts/predictions/memory_required_qa"),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _condition_documents(
    results: dict[str, MemoryRequiredQAConditionResult],
) -> dict[str, object]:
    return {name: result.to_dict() for name, result in results.items()}


def _passes_gate(
    mode: str,
    results: dict[str, MemoryRequiredQAConditionResult],
) -> bool:
    normal = results["normal"].exact_accuracy
    minimum_accuracy = 0.9 if mode == "oracle" else 0.8
    required = ("drop", "zero", "shuffle")
    if mode != "oracle":
        required = (*required, "no_writes")
    return normal >= minimum_accuracy and all(
        normal - results[condition].exact_accuracy >= 0.3
        for condition in required
    )


def _plot(
    losses: tuple[float, ...],
    results: dict[str, MemoryRequiredQAConditionResult],
    destination: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(losses)
    axes[0].set_title("training loss")
    axes[0].set_xlabel("step")
    axes[1].bar(
        tuple(results),
        tuple(result.exact_accuracy for result in results.values()),
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_title("generated exact accuracy")
    axes[1].tick_params(axis="x", rotation=25)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    for name in (
        "segment_length",
        "train_examples",
        "validation_examples",
        "steps",
        "batch_size",
        "max_new_tokens",
        "progress_every",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.learning_rate <= 0 or args.gradient_clip_norm <= 0:
        raise ValueError("learning rate and gradient clip norm must be positive")
    if args.weight_decay < 0:
        raise ValueError("weight decay must be nonnegative")
    if args.distractor_segments < 0:
        raise ValueError("distractor_segments must be nonnegative")
    if args.memory_capacity is not None and args.memory_capacity <= 0:
        raise ValueError("memory_capacity must be positive")
    if args.write_loss_weight < 0:
        raise ValueError("write_loss_weight must be nonnegative")
    if args.write_gate_kernel_size <= 0 or args.write_gate_kernel_size % 2 == 0:
        raise ValueError("write_gate_kernel_size must be a positive odd integer")
    if args.mode == "gated" and args.distractor_write_policy != "learned":
        raise ValueError("gated mode requires distractor_write_policy='learned'")
    if args.mode == "gated" and args.write_loss_weight == 0:
        raise ValueError("gated mode requires positive write_loss_weight")
    if args.mode != "gated" and args.distractor_write_policy == "learned":
        raise ValueError("learned write policy requires gated mode")
    if args.seed < 0 or args.validation_seed < 0:
        raise ValueError("seeds must be nonnegative")

    repository_root = Path(__file__).resolve().parents[1]
    checkpoint_path = args.checkpoint.resolve()
    device = select_device(args.device)
    seed_everything(args.seed)
    loaded = load_wikitext_checkpoint(checkpoint_path, device=device)
    if args.segment_length > loaded.config.model.max_local_tokens:
        raise ValueError("segment_length exceeds the model local window")
    if not isinstance(loaded.decoder.compressor, MeanPoolMemoryCompressor):
        raise ValueError("the minimal causal test requires a mean-pool compressor")

    decoder = loaded.decoder
    decoder.segment_length = args.segment_length
    memory_capacity = (
        args.memory_capacity
        if args.memory_capacity is not None
        else 1 + args.distractor_segments
    )
    if args.mode == "gated":
        decoder.bank = GatedRecurrentMemoryBank(
            capacity=memory_capacity,
            model_width=decoder.model.config.d_model,
        ).to(device)
        decoder.write_gate = TokenSegmentWriteGate(
            decoder.model.config.d_model,
            kernel_size=args.write_gate_kernel_size,
        ).to(device)
    else:
        decoder.bank = RecurrentMemoryBank(
            capacity=memory_capacity,
            model_width=decoder.model.config.d_model,
        ).to(device)
    decoder.memory_position_mode = "virtual"
    write_distractors = args.distractor_write_policy != "none"
    tokenizer = ByteTokenizer()
    babi_root = repository_root / "data/raw/tasks_1-20_v1-2/en-valid-10k"
    train_source = load_babi_file(
        babi_root / "qa1_train.txt",
        task_id="qa1",
        split="train",
    )
    validation_source = load_babi_file(
        babi_root / "qa1_valid.txt",
        task_id="qa1",
        split="validation",
    )
    train_examples = build_memory_required_qa_examples(
        select_reasoning_examples(
            train_source,
            count=args.train_examples,
            seed=args.seed,
        ),
        tokenizer,
        segment_length=args.segment_length,
        distractor_segments=args.distractor_segments,
        distractor_variant=args.train_distractor_variant,
    )
    validation_examples = build_memory_required_qa_examples(
        select_reasoning_examples(
            validation_source,
            count=args.validation_examples,
            seed=args.validation_seed,
        ),
        tokenizer,
        segment_length=args.segment_length,
        distractor_segments=args.distractor_segments,
        distractor_variant=args.validation_distractor_variant,
    )

    oracle_train = None
    oracle_validation = None
    if args.mode == "oracle":
        oracle_train = precompute_oracle_memories(
            decoder,
            train_examples,
            device=device,
        )
        oracle_validation = precompute_oracle_memories(
            decoder,
            validation_examples,
            device=device,
        )

    before = evaluate_memory_required_qa(
        decoder,
        validation_examples,
        mode=args.mode,
        device=device,
        max_new_tokens=args.max_new_tokens,
        oracle_values=oracle_validation,
        write_distractors=write_distractors,
    )
    write_gate_before = (
        evaluate_memory_required_write_gate(
            decoder,
            validation_examples,
            device=device,
        )
        if args.mode == "gated"
        else None
    )
    gradient_before = None
    if args.mode == "learned":
        gradient_before = {
            condition: audit_cross_segment_gradients(
                decoder,
                validation_examples[0],
                condition=condition,
                pad_id=tokenizer.special_tokens["<pad>"],
                device=device,
                write_distractors=write_distractors,
            ).to_dict()
            for condition in ("normal", "drop")
        }

    parameters = (
        decoder.model.parameters()
        if args.mode == "oracle"
        else decoder.parameters()
    )
    optimizer = torch.optim.AdamW(
        parameters,
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
                        "elapsed_seconds": round(time.perf_counter() - started, 1),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    history = train_memory_required_qa(
        decoder,
        optimizer,
        train_examples,
        mode=args.mode,
        steps=args.steps,
        batch_size=args.batch_size,
        gradient_clip_norm=args.gradient_clip_norm,
        pad_id=tokenizer.special_tokens["<pad>"],
        device=device,
        seed=args.seed,
        oracle_values=oracle_train,
        write_distractors=write_distractors,
        write_loss_weight=(args.write_loss_weight if args.mode == "gated" else 0.0),
        on_step=report_progress,
    )
    training_seconds = time.perf_counter() - started
    after = evaluate_memory_required_qa(
        decoder,
        validation_examples,
        mode=args.mode,
        device=device,
        max_new_tokens=args.max_new_tokens,
        oracle_values=oracle_validation,
        write_distractors=write_distractors,
    )
    write_gate_after = (
        evaluate_memory_required_write_gate(
            decoder,
            validation_examples,
            device=device,
        )
        if args.mode == "gated"
        else None
    )
    gradient_after = None
    if args.mode == "learned":
        gradient_after = {
            condition: audit_cross_segment_gradients(
                decoder,
                validation_examples[0],
                condition=condition,
                pad_id=tokenizer.special_tokens["<pad>"],
                device=device,
                write_distractors=write_distractors,
            ).to_dict()
            for condition in ("normal", "drop")
        }

    experiment_config = replace(
        loaded.config,
        seed=args.seed,
        stream=replace(
            loaded.config.stream,
            segment_length=args.segment_length,
        ),
        memory=replace(loaded.config.memory, n_slots=memory_capacity),
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
        / args.mode
        / f"distractors_{args.distractor_segments}"
        / f"capacity_{memory_capacity}"
        / f"writes_{args.distractor_write_policy}"
        / f"validation_{args.validation_distractor_variant}"
        / f"seed_{args.seed}",
        experiment_config,
        git_commit=source_state.commit,
        source_state=source_state,
    )
    checkpoint_output = run_directory / "checkpoint.pt"
    save_checkpoint(
        checkpoint_output,
        model=decoder,
        optimizer=optimizer,
        step=args.steps,
        config=experiment_config,
        extra={
            "architecture": "segmented_memory_required_byte_qa",
            "mode": args.mode,
            "parent_checkpoint": str(checkpoint_path),
            "parent_checkpoint_sha256": _sha256(checkpoint_path),
            "segment_length": args.segment_length,
            "memory_capacity": memory_capacity,
            "memory_position_mode": "virtual",
            "compressor": "mean",
            "memory_update": "gated" if args.mode == "gated" else "fifo",
            "write_gate": "token_conv" if args.mode == "gated" else "none",
            "write_gate_kernel_size": (
                args.write_gate_kernel_size if args.mode == "gated" else None
            ),
            "answer_only_loss": args.mode != "gated",
            "write_supervision": (
                "support_positive_distractor_negative"
                if args.mode == "gated"
                else "none"
            ),
            "distractor_segments": args.distractor_segments,
            "train_distractor_variant": args.train_distractor_variant,
            "validation_distractor_variant": args.validation_distractor_variant,
            "distractor_write_policy": args.distractor_write_policy,
            "write_loss_weight": (
                args.write_loss_weight if args.mode == "gated" else 0.0
            ),
        },
    )
    result_document = {
        "status": "development_memory_required_qa",
        "gate_passed": _passes_gate(args.mode, after),
        "mode": args.mode,
        "seed": args.seed,
        "validation_seed": args.validation_seed,
        "device": str(device),
        "source_state": source_state.to_dict(),
        "parent_checkpoint": str(checkpoint_path),
        "parent_checkpoint_sha256": _sha256(checkpoint_path),
        "train_data_sha256": _sha256(babi_root / "qa1_train.txt"),
        "validation_data_sha256": _sha256(babi_root / "qa1_valid.txt"),
        "checkpoint": str(checkpoint_output),
        "checkpoint_sha256": _sha256(checkpoint_output),
        "segment_length": args.segment_length,
        "memory_capacity": memory_capacity,
        "distractor_segments": args.distractor_segments,
        "train_distractor_variant": args.train_distractor_variant,
        "validation_distractor_variant": args.validation_distractor_variant,
        "distractor_write_policy": args.distractor_write_policy,
        "memory_position_mode": "virtual",
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip_norm": args.gradient_clip_norm,
        "training_seconds": training_seconds,
        "training_history": history.to_dict(),
        "write_gate_before": (
            write_gate_before.to_dict() if write_gate_before is not None else None
        ),
        "write_gate_after": (
            write_gate_after.to_dict() if write_gate_after is not None else None
        ),
        "gradient_before": gradient_before,
        "gradient_after": gradient_after,
        "before": _condition_documents(before),
        "after": _condition_documents(after),
    }
    (run_directory / "results.json").write_text(
        json.dumps(result_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot(history.losses, after, run_directory / "memory_required_qa.png")
    print(
        json.dumps(
            {
                "gate_passed": result_document["gate_passed"],
                "mode": args.mode,
                "seed": args.seed,
                "training_seconds": training_seconds,
                "after": {
                    name: {
                        "exact_accuracy": result.exact_accuracy,
                        "first_byte_accuracy": result.first_byte_accuracy,
                        "answer_byte_nll": result.answer_byte_nll,
                        "first_byte_logit_linf": (
                            result.mean_first_byte_logit_linf_from_normal
                        ),
                    }
                    for name, result in after.items()
                },
                "write_gate_after": (
                    write_gate_after.to_dict()
                    if write_gate_after is not None
                    else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"artifacts: {run_directory}")


if __name__ == "__main__":
    main()
