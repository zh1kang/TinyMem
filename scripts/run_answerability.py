#!/usr/bin/env python3
"""Train and evaluate TinyMem reliability on correction/deletion episodes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from tinymem.data.correction_deletion import UpdateExample, generate_update_examples
from tinymem.evaluation.answerability import (
    DecoderAnswerabilityResult,
    evaluate_decoder_answerability,
    freeze_before_correction_write_mask,
)
from tinymem.evaluation.continuous_memory import (
    drop_newest_memory,
    keep_oldest_memory,
    replace_newest_memory,
    shuffle_memory,
)
from tinymem.memory.continuous import MultiSlotAttentionMemoryCompressor
from tinymem.memory.controller import AdaptiveWriteController
from tinymem.memory.recurrent_memory import GatedRecurrentMemoryBank
from tinymem.model.answerability import AnswerabilityHead
from tinymem.model.config import (
    ExperimentConfig,
    MemoryConfig,
    ModelConfig,
    StreamConfig,
    TrainingConfig,
)
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.training.checkpointing import save_checkpoint
from tinymem.training.continuous import (
    encode_update_with_write_targets,
    train_continuous_answer_supervision,
)
from tinymem.training.controlled_qa import EncodedQAExample, build_qa_vocabulary
from tinymem.utils.device import select_device
from tinymem.utils.experiment import create_run_directory, current_git_commit
from tinymem.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1_500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--answerability-loss-weight", type=float, default=1.0)
    parser.add_argument("--write-loss-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="cpu",
        help="CPU is the deterministic default for this experiment",
    )
    parser.add_argument("--train-examples", type=int, default=2_000)
    parser.add_argument("--validation-examples", type=int, default=400)
    parser.add_argument("--test-examples", type=int, default=400)
    parser.add_argument("--deletion-rate", type=float, default=0.35)
    parser.add_argument("--query-delay", type=int, default=4)
    parser.add_argument("--distractor-count", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--segment-length", type=int, default=16)
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/predictions/answerability"),
    )
    return parser.parse_args()


def plot_results(
    results: list[DecoderAnswerabilityResult],
    destination: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    normal = results[0]
    coverage = [point.coverage for point in normal.evaluation.points]
    accuracy = [point.selective_accuracy for point in normal.evaluation.points]
    axes[0].plot(coverage, accuracy, marker="o")
    axes[0].set_xlabel("coverage")
    axes[0].set_ylabel("accuracy among answered examples")
    axes[0].set_title("normal selective prediction")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.25)

    names = [result.intervention for result in results]
    false_confidence = [
        result.evaluation.points[1].false_confidence_rate
        for result in results
    ]
    axes[1].bar(names, false_confidence)
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("false confidence rate at p >= 0.5")
    axes[1].set_title("memory intervention sensitivity")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    device = select_device(args.device)
    seed_everything(args.seed)

    train_updates = generate_update_examples(
        split="train",
        count=args.train_examples,
        base_seed=args.seed,
        deletion_rate=args.deletion_rate,
        query_delay=args.query_delay,
        distractor_count=args.distractor_count,
    )
    validation_updates = generate_update_examples(
        split="validation",
        count=args.validation_examples,
        base_seed=args.seed,
        deletion_rate=args.deletion_rate,
        query_delay=args.query_delay,
        distractor_count=args.distractor_count,
    )
    test_updates = generate_update_examples(
        split="test",
        count=args.test_examples,
        base_seed=args.seed,
        deletion_rate=args.deletion_rate,
        query_delay=args.query_delay,
        distractor_count=args.distractor_count,
    )
    vocabulary = build_qa_vocabulary([item.example for item in train_updates])

    def encode(updates: list[UpdateExample]) -> list[EncodedQAExample]:
        return [
            encode_update_with_write_targets(
                item.example,
                vocabulary,
                segment_length=args.segment_length,
            )
            for item in updates
        ]

    encoded_train = encode(train_updates)
    encoded_validation = encode(validation_updates)
    encoded_test = encode(test_updates)
    model_config = ModelConfig(
        vocab_size=len(vocabulary),
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_local_tokens=args.segment_length,
    )
    config = ExperimentConfig(
        seed=args.seed,
        model=model_config,
        stream=StreamConfig(
            segment_length=args.segment_length,
            local_window=args.segment_length,
        ),
        memory=MemoryConfig(
            n_slots=args.capacity,
            code_dim=args.d_model,
            codes_per_write=1,
        ),
        training=TrainingConfig(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            gradient_clip_norm=args.gradient_clip_norm,
            warmup_steps=0,
            max_steps=args.steps,
        ),
    )
    decoder = SegmentedContinuousDecoder(
        DecoderOnlyTransformer(model_config),
        MultiSlotAttentionMemoryCompressor(args.d_model, summary_slots=1),
        GatedRecurrentMemoryBank(
            capacity=args.capacity,
            model_width=args.d_model,
        ),
        segment_length=args.segment_length,
        write_controller=AdaptiveWriteController(args.d_model),
        answerability_head=AnswerabilityHead(args.d_model),
    ).to(device)
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    training_started = time.perf_counter()
    losses = train_continuous_answer_supervision(
        decoder,
        optimizer,
        encoded_train,
        steps=args.steps,
        batch_size=args.batch_size,
        gradient_clip_norm=args.gradient_clip_norm,
        pad_id=vocabulary.token_to_id["<pad>"],
        device=device,
        seed=args.seed,
        write_loss_weight=args.write_loss_weight,
        answerability_loss_weight=args.answerability_loss_weight,
    )
    training_seconds = time.perf_counter() - training_started

    def evaluate(
        examples: list[EncodedQAExample],
        *,
        name: str,
        intervention=None,
        forced_writes: torch.Tensor | None = None,
        corrupted: bool = False,
    ) -> DecoderAnswerabilityResult:
        return evaluate_decoder_answerability(
            decoder,
            examples,
            batch_size=args.batch_size,
            pad_id=vocabulary.token_to_id["<pad>"],
            device=device,
            intervention_name=name,
            memory_intervention=intervention,
            forced_writes=forced_writes,
            corrupted_memory=corrupted,
            thresholds=(0.25, 0.5, 0.75),
        )

    validation = evaluate(encoded_validation, name="normal")
    results = [
        evaluate(encoded_test, name="normal"),
        evaluate(
            encoded_test,
            name="drop_newest",
            intervention=drop_newest_memory,
            corrupted=True,
        ),
        evaluate(
            encoded_test,
            name="shuffle",
            intervention=shuffle_memory,
            corrupted=True,
        ),
        evaluate(
            encoded_test,
            name="stale",
            intervention=keep_oldest_memory,
            corrupted=True,
        ),
        evaluate(
            encoded_test,
            name="replace_newest",
            intervention=replace_newest_memory,
            corrupted=True,
        ),
        evaluate(
            encoded_test,
            name="freeze_before_correction",
            forced_writes=freeze_before_correction_write_mask(encoded_test),
            corrupted=True,
        ),
    ]
    commit = current_git_commit(repository_root)
    run_directory = create_run_directory(
        repository_root / args.artifact_root,
        config,
        git_commit=commit,
    )
    result_document = {
        "status": "development_single_seed",
        "seed": args.seed,
        "device": str(device),
        "git_commit": commit,
        "training_examples": len(encoded_train),
        "validation_examples": len(encoded_validation),
        "test_examples": len(encoded_test),
        "training_seconds": training_seconds,
        "final_training_loss": losses[-1],
        "validation": validation.to_dict(),
        "test_interventions": [result.to_dict() for result in results],
    }
    (run_directory / "results.json").write_text(
        json.dumps(result_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    save_checkpoint(
        run_directory / "checkpoint.pt",
        model=decoder,
        optimizer=optimizer,
        step=args.steps,
        config=config,
        extra={
            "architecture": "segmented_continuous_answerability",
            "vocabulary": list(vocabulary.id_to_token),
        },
    )
    plot_results(results, run_directory / "answerability.png")
    print(
        json.dumps(
            {
                "seed": args.seed,
                "training_seconds": training_seconds,
                "final_training_loss": losses[-1],
                "validation": validation.evaluation.to_dict(),
                "test": {
                    result.intervention: result.evaluation.to_dict()
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
