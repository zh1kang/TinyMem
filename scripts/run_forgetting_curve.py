#!/usr/bin/env python3
"""Train a qa1 baseline and measure accuracy by BABILong evidence delay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from tinymem.data.babi import load_babi_file
from tinymem.data.babilong import load_babilong_file
from tinymem.evaluation.forgetting_curve import evaluate_local_forgetting_curve
from tinymem.evaluation.multi_token_prediction import evaluate_base_mtp_loss
from tinymem.model.config import (
    ExperimentConfig,
    MemoryConfig,
    ModelConfig,
    MTPConfig,
    StreamConfig,
    TrainingConfig,
)
from tinymem.model.multi_token_prediction import MultiTokenPredictionHeads
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.training.checkpointing import save_checkpoint
from tinymem.training.controlled_qa import (
    answer_accuracy,
    build_qa_vocabulary,
    encode_qa_examples,
    train_answer_supervision,
)
from tinymem.utils.device import select_device
from tinymem.utils.experiment import create_run_directory, current_git_commit
from tinymem.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--local-window", type=int, default=128)
    parser.add_argument("--mtp-horizons", type=int, nargs="*", default=())
    parser.add_argument("--mtp-loss-weight", type=float, default=0.2)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/predictions/forgetting_curve"),
    )
    return parser.parse_args()


def plot_results(results: list[dict[str, object]], destination: Path) -> None:
    labels = [str(result["label"]) for result in results]
    accuracies = [float(result["accuracy"]) for result in results]
    counts = [int(result["count"]) for result in results]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(labels, accuracies, marker="o", linewidth=2)
    axis.axhline(1 / 6, color="gray", linestyle="--", label="chance (1/6)")
    axis.axvline(2.5, color="tab:red", linestyle=":", label="128-token window")
    for index, (accuracy, count) in enumerate(zip(accuracies, counts, strict=True)):
        axis.annotate(
            f"n={count}",
            (index, accuracy),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )
    axis.set_ylim(0, 1)
    axis.set_xlabel("tokens after answer-bearing evidence")
    axis.set_ylabel("exact answer accuracy")
    axis.set_title("qa1 local-window forgetting curve")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    mtp_config = MTPConfig(
        enabled=bool(args.mtp_horizons),
        horizons=tuple(args.mtp_horizons) or (2, 3, 4),
        loss_weight=args.mtp_loss_weight,
    )
    repository_root = Path(__file__).resolve().parents[1]
    device = select_device(args.device)
    seed_everything(args.seed)

    babi_root = repository_root / "data/raw/tasks_1-20_v1-2/en-valid-10k"
    train_examples = load_babi_file(
        babi_root / "qa1_train.txt",
        task_id="qa1",
        split="train",
    )
    validation_examples = load_babi_file(
        babi_root / "qa1_valid.txt",
        task_id="qa1",
        split="validation",
    )
    vocabulary = build_qa_vocabulary(train_examples)
    encoded_train, skipped_train = encode_qa_examples(
        train_examples,
        vocabulary,
        max_tokens=args.local_window,
    )
    encoded_validation, skipped_validation = encode_qa_examples(
        validation_examples,
        vocabulary,
        max_tokens=args.local_window,
    )

    model_config = ModelConfig(
        vocab_size=len(vocabulary),
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_local_tokens=args.local_window,
    )
    config = ExperimentConfig(
        seed=args.seed,
        model=model_config,
        stream=StreamConfig(
            segment_length=min(64, args.local_window),
            local_window=args.local_window,
        ),
        memory=MemoryConfig(code_dim=args.d_model),
        training=TrainingConfig(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            gradient_clip_norm=args.gradient_clip_norm,
            warmup_steps=0,
            max_steps=args.steps,
        ),
        mtp=mtp_config,
    )
    model = DecoderOnlyTransformer(model_config).to(device)
    mtp_heads = (
        MultiTokenPredictionHeads(
            model_config.d_model,
            model_config.vocab_size,
            mtp_config.horizons,
        ).to(device)
        if mtp_config.enabled
        else None
    )
    optimizer = torch.optim.AdamW(
        (
            (*model.parameters(), *mtp_heads.parameters())
            if mtp_heads is not None
            else model.parameters()
        ),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    auxiliary_loss_before = (
        evaluate_base_mtp_loss(
            model,
            mtp_heads,
            encoded_validation,
            batch_size=args.batch_size,
            pad_id=vocabulary.token_to_id["<pad>"],
            device=device,
        )
        if mtp_heads is not None
        else None
    )
    losses = train_answer_supervision(
        model,
        optimizer,
        encoded_train,
        steps=args.steps,
        batch_size=args.batch_size,
        gradient_clip_norm=args.gradient_clip_norm,
        pad_id=vocabulary.token_to_id["<pad>"],
        device=device,
        seed=args.seed,
        mtp_heads=mtp_heads,
        mtp_loss_weight=(mtp_config.loss_weight if mtp_heads is not None else 0.0),
    )
    auxiliary_loss_after = (
        evaluate_base_mtp_loss(
            model,
            mtp_heads,
            encoded_validation,
            batch_size=args.batch_size,
            pad_id=vocabulary.token_to_id["<pad>"],
            device=device,
        )
        if mtp_heads is not None
        else None
    )
    validation_accuracy = answer_accuracy(
        model,
        encoded_validation,
        batch_size=args.batch_size,
        pad_id=vocabulary.token_to_id["<pad>"],
        device=device,
    )

    babilong_examples = []
    for context_length in ("1k", "2k", "4k", "8k"):
        babilong_examples.extend(
            load_babilong_file(
                repository_root / f"data/raw/babilong/qa1/{context_length}.json",
                task_id="qa1",
                split="test",
            )
        )
    curve = evaluate_local_forgetting_curve(
        model,
        vocabulary,
        babilong_examples,
        batch_size=args.batch_size,
        device=device,
    )

    commit = current_git_commit(repository_root)
    run_directory = create_run_directory(
        repository_root / args.artifact_root,
        config,
        git_commit=commit,
    )
    curve_records = [bucket.to_dict() for bucket in curve]
    result = {
        "status": "development_single_seed",
        "task_id": "qa1",
        "device": str(device),
        "git_commit": commit,
        "manifest_sha256": json.loads(
            (repository_root / "data/installed.lock.json").read_text()
        )["manifest_sha256"],
        "train_examples": len(encoded_train),
        "skipped_train_examples": skipped_train,
        "validation_examples": len(encoded_validation),
        "skipped_validation_examples": skipped_validation,
        "final_training_loss": losses[-1],
        "validation_accuracy": validation_accuracy,
        "mtp_horizons": list(mtp_config.horizons) if mtp_heads is not None else [],
        "mtp_loss_weight": mtp_config.loss_weight if mtp_heads is not None else 0.0,
        "auxiliary_loss_before": auxiliary_loss_before,
        "auxiliary_loss_after": auxiliary_loss_after,
        "curve": curve_records,
    }
    (run_directory / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    save_checkpoint(
        run_directory / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        step=args.steps,
        config=config,
        extra={
            "vocabulary": list(vocabulary.id_to_token),
            "validation_accuracy": validation_accuracy,
            "mtp_horizons": (
                list(mtp_config.horizons) if mtp_heads is not None else []
            ),
            "mtp_head_state": (
                mtp_heads.state_dict() if mtp_heads is not None else None
            ),
        },
    )
    plot_results(curve_records, run_directory / "forgetting_curve.png")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"artifacts: {run_directory}")


if __name__ == "__main__":
    main()
