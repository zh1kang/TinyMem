#!/usr/bin/env python3
"""Train TinyMem on WikiText-2 and report held-out perplexity."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from tinymem.data.wikitext import load_wikitext_parquet
from tinymem.evaluation.continuous_memory import drop_memory, zero_memory
from tinymem.memory.continuous import MeanPoolMemoryCompressor
from tinymem.memory.recurrent_memory import RecurrentMemoryBank
from tinymem.model.config import (
    ExperimentConfig,
    MemoryConfig,
    ModelConfig,
    StreamConfig,
    TrainingConfig,
)
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.tokenization.byte_tokenizer import ByteTokenizer
from tinymem.training.checkpointing import save_checkpoint
from tinymem.training.language_model import (
    LanguageModelEvaluation,
    evaluate_segmented_language_model,
    evaluate_streaming_language_model,
    train_segmented_language_model,
)
from tinymem.utils.device import select_device
from tinymem.utils.experiment import create_run_directory, current_git_commit
from tinymem.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1_500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="cpu",
        help="CPU is the deterministic default for this experiment",
    )
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--segment-length", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument(
        "--evaluation-windows",
        type=int,
        nargs="+",
        default=(16, 32, 64),
    )
    parser.add_argument("--evaluation-batch-size", type=int, default=16)
    parser.add_argument("--max-validation-tokens", type=int, default=0)
    parser.add_argument("--max-test-tokens", type=int, default=0)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/predictions/wikitext_language_model"),
    )
    return parser.parse_args()


def plot_results(
    window_results: dict[int, LanguageModelEvaluation],
    ablations: dict[str, LanguageModelEvaluation],
    destination: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    windows = sorted(window_results)
    axes[0].plot(
        windows,
        [window_results[window].perplexity for window in windows],
        marker="o",
    )
    axes[0].set_xlabel("local segment tokens")
    axes[0].set_ylabel("validation byte perplexity")
    axes[0].set_title("local-window sweep")
    axes[0].grid(alpha=0.25)

    names = list(ablations)
    axes[1].bar(
        names,
        [ablations[name].perplexity for name in names],
    )
    axes[1].set_ylabel("validation byte perplexity")
    axes[1].set_title("semantic-memory ablation")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.sequence_length <= args.segment_length:
        raise ValueError("sequence_length must exceed segment_length")
    if any(
        window <= 0 or window > args.segment_length
        for window in args.evaluation_windows
    ):
        raise ValueError("evaluation windows must be in (0, segment_length]")

    repository_root = Path(__file__).resolve().parents[1]
    data_root = repository_root / "data/raw/wikitext2"
    device = select_device(args.device)
    seed_everything(args.seed)
    tokenizer = ByteTokenizer()
    splits = {
        split: load_wikitext_parquet(
            data_root / f"{split}.parquet",
            split=split,
        )
        for split in ("train", "validation", "test")
    }
    token_streams = {
        split: torch.tensor(tokenizer.encode(data.text), dtype=torch.long)
        for split, data in splits.items()
    }
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
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
        MeanPoolMemoryCompressor(args.d_model),
        RecurrentMemoryBank(
            capacity=args.capacity,
            model_width=args.d_model,
        ),
        segment_length=args.segment_length,
    ).to(device)
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    training_started = time.perf_counter()
    losses = train_segmented_language_model(
        decoder,
        optimizer,
        token_streams["train"],
        steps=args.steps,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        gradient_clip_norm=args.gradient_clip_norm,
        device=device,
        seed=args.seed,
    )
    training_seconds = time.perf_counter() - training_started
    max_validation_tokens = args.max_validation_tokens or None
    max_test_tokens = args.max_test_tokens or None
    window_results: dict[int, LanguageModelEvaluation] = {}
    for window in sorted(set(args.evaluation_windows)):
        decoder.segment_length = window
        window_results[window] = evaluate_segmented_language_model(
            decoder,
            token_streams["validation"],
            sequence_length=args.sequence_length,
            batch_size=args.evaluation_batch_size,
            pad_id=tokenizer.special_tokens["<pad>"],
            device=device,
            max_tokens=max_validation_tokens,
        )
    selected_window = min(
        window_results,
        key=lambda window: window_results[window].loss,
    )
    decoder.segment_length = selected_window
    validation_ablations = {
        "normal": evaluate_segmented_language_model(
            decoder,
            token_streams["validation"],
            sequence_length=args.sequence_length,
            batch_size=args.evaluation_batch_size,
            pad_id=tokenizer.special_tokens["<pad>"],
            device=device,
            max_tokens=max_validation_tokens,
        ),
        "drop_memory": evaluate_segmented_language_model(
            decoder,
            token_streams["validation"],
            sequence_length=args.sequence_length,
            batch_size=args.evaluation_batch_size,
            pad_id=tokenizer.special_tokens["<pad>"],
            device=device,
            max_tokens=max_validation_tokens,
            memory_intervention=drop_memory,
        ),
        "zero_memory": evaluate_segmented_language_model(
            decoder,
            token_streams["validation"],
            sequence_length=args.sequence_length,
            batch_size=args.evaluation_batch_size,
            pad_id=tokenizer.special_tokens["<pad>"],
            device=device,
            max_tokens=max_validation_tokens,
            memory_intervention=zero_memory,
        ),
    }
    within_sequence_final_segment_validation = evaluate_segmented_language_model(
        decoder,
        token_streams["validation"],
        sequence_length=args.sequence_length,
        batch_size=args.evaluation_batch_size,
        pad_id=tokenizer.special_tokens["<pad>"],
        device=device,
        max_tokens=max_validation_tokens,
        final_segment_only=True,
    )
    streaming_validation = evaluate_streaming_language_model(
        decoder,
        token_streams["validation"],
        chunk_tokens=args.sequence_length,
        device=device,
        max_tokens=max_validation_tokens,
    )
    windowed_test = evaluate_segmented_language_model(
        decoder,
        token_streams["test"],
        sequence_length=args.sequence_length,
        batch_size=args.evaluation_batch_size,
        pad_id=tokenizer.special_tokens["<pad>"],
        device=device,
        max_tokens=max_test_tokens,
    )
    streaming_test = evaluate_streaming_language_model(
        decoder,
        token_streams["test"],
        chunk_tokens=args.sequence_length,
        device=device,
        max_tokens=max_test_tokens,
    )

    commit = current_git_commit(repository_root)
    run_directory = create_run_directory(
        repository_root / args.artifact_root,
        config,
        git_commit=commit,
    )
    memory_bytes = (
        args.capacity
        * args.d_model
        * decoder.model.token_embedding.weight.element_size()
    )
    result_document = {
        "status": (
            "development_single_seed_capped_test"
            if max_test_tokens is not None
            else "development_single_seed_full_test"
        ),
        "dataset": "wikitext-2-raw-v1",
        "seed": args.seed,
        "device": str(device),
        "git_commit": commit,
        "training_tokens": token_streams["train"].numel(),
        "validation_tokens": token_streams["validation"].numel(),
        "test_tokens": token_streams["test"].numel(),
        "evaluated_validation_tokens": max_validation_tokens,
        "evaluated_test_tokens": max_test_tokens,
        "training_seconds": training_seconds,
        "final_training_loss": losses[-1],
        "selected_window": selected_window,
        "memory_bytes_per_example": memory_bytes,
        "validation_windows": {
            str(window): result.to_dict()
            for window, result in window_results.items()
        },
        "validation_ablations": {
            name: result.to_dict()
            for name, result in validation_ablations.items()
        },
        "within_sequence_final_segment_validation": (
            within_sequence_final_segment_validation.to_dict()
        ),
        "streaming_validation": streaming_validation.to_dict(),
        "windowed_test": windowed_test.to_dict(),
        "streaming_test": streaming_test.to_dict(),
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
        config=replace(
            config,
            stream=replace(
                config.stream,
                segment_length=selected_window,
            ),
        ),
        extra={
            "architecture": "segmented_continuous_wikitext_byte_lm",
            "selected_window": selected_window,
            "training_segment_length": args.segment_length,
            "tokenizer": "utf8_bytes_v1",
        },
    )
    plot_results(
        window_results,
        validation_ablations,
        run_directory / "wikitext_perplexity.png",
    )
    print(json.dumps(result_document, indent=2, sort_keys=True))
    print(f"artifacts: {run_directory}")


if __name__ == "__main__":
    main()
