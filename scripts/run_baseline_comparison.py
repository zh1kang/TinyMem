#!/usr/bin/env python3
"""Compare fixed qa1 memory baselines under one measured byte budget."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from tinymem.data.babilong import load_babilong_file
from tinymem.data.schema import ReasoningExample
from tinymem.data.vocabulary import SPECIAL_TOKENS, ControlledVocabulary
from tinymem.evaluation.baseline_comparison import (
    BASELINE_NAMES,
    BaselineResult,
    evaluate_qa1_baseline,
    require_equal_memory_budget,
)
from tinymem.evaluation.forgetting_curve import delay_label, qa1_evidence_delay_tokens
from tinymem.model.config import (
    ExperimentConfig,
    MTPConfig,
    MemoryConfig,
    ModelConfig,
    StreamConfig,
    TrainingConfig,
)
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.training.checkpointing import load_checkpoint
from tinymem.utils.device import select_device
from tinymem.utils.experiment import create_run_directory, current_git_commit
from tinymem.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--examples-per-bucket",
        type=int,
        default=0,
        help="zero uses every example; a positive value makes a development slice",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/predictions/baseline_comparison"),
    )
    return parser.parse_args()


def _experiment_config(data: object) -> ExperimentConfig:
    if not isinstance(data, dict):
        raise ValueError("checkpoint must contain an experiment configuration")
    try:
        return ExperimentConfig(
            seed=data["seed"],
            model=ModelConfig(**data["model"]),
            stream=StreamConfig(**data["stream"]),
            memory=MemoryConfig(**data["memory"]),
            training=TrainingConfig(**data["training"]),
            mtp=MTPConfig(**data["mtp"]),
        )
    except (KeyError, TypeError) as error:
        raise ValueError("checkpoint experiment configuration is invalid") from error


def _checkpoint_inputs(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[ExperimentConfig, DecoderOnlyTransformer, ControlledVocabulary, int]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a dictionary")
    config = _experiment_config(payload.get("config"))
    extra = payload.get("extra")
    if not isinstance(extra, dict):
        raise ValueError("checkpoint extra metadata must be a dictionary")
    vocabulary_tokens = extra.get("vocabulary")
    if not isinstance(vocabulary_tokens, list) or not all(
        isinstance(token, str) for token in vocabulary_tokens
    ):
        raise ValueError("checkpoint must contain its controlled vocabulary")
    if tuple(vocabulary_tokens[: len(SPECIAL_TOKENS)]) != SPECIAL_TOKENS:
        raise ValueError("checkpoint vocabulary has invalid special tokens")
    vocabulary = ControlledVocabulary(vocabulary_tokens[len(SPECIAL_TOKENS) :])
    if vocabulary.id_to_token != tuple(vocabulary_tokens):
        raise ValueError("checkpoint vocabulary is not in standard order")
    if len(vocabulary) != config.model.vocab_size:
        raise ValueError("checkpoint vocabulary size does not match model config")

    model = DecoderOnlyTransformer(config.model).to(device)
    metadata = load_checkpoint(checkpoint_path, model=model, map_location=device)
    step = metadata["step"]
    if isinstance(step, bool) or not isinstance(step, int):
        raise ValueError("checkpoint step must be an integer")
    return config, model, vocabulary, step


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _select_development_slice(
    examples: list[ReasoningExample],
    vocabulary: ControlledVocabulary,
    *,
    examples_per_bucket: int,
) -> list[ReasoningExample]:
    if examples_per_bucket < 0:
        raise ValueError("examples_per_bucket must be nonnegative")
    if examples_per_bucket == 0:
        return examples
    selected = []
    bucket_counts: dict[str, int] = {}
    for example in examples:
        label = delay_label(qa1_evidence_delay_tokens(example, vocabulary))
        count = bucket_counts.get(label, 0)
        if count >= examples_per_bucket:
            continue
        selected.append(example)
        bucket_counts[label] = count + 1
    return selected


def plot_results(results: list[BaselineResult], destination: Path) -> None:
    names = [result.baseline.replace("_", " ") for result in results]
    accuracies = [result.outside_window_accuracy for result in results]
    figure, axis = plt.subplots(figsize=(9, 4.8))
    bars = axis.bar(names, accuracies)
    axis.axhline(1 / 6, color="gray", linestyle="--", label="chance (1/6)")
    for bar, result in zip(bars, results, strict=True):
        axis.annotate(
            f"{result.memory_bytes} B\n{result.outside_window_correct}/"
            f"{result.outside_window_count}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    axis.set_ylim(0, 1)
    axis.set_ylabel("exact accuracy beyond local window")
    axis.set_title("qa1 fixed-memory comparison at matched allocation")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.capacity <= 0 or args.batch_size <= 0 or args.seed < 0:
        raise ValueError("capacity and batch_size must be positive and seed nonnegative")
    repository_root = Path(__file__).resolve().parents[1]
    checkpoint_path = args.checkpoint.resolve()
    device = select_device(args.device)
    seed_everything(args.seed)
    checkpoint_config, model, vocabulary, checkpoint_step = _checkpoint_inputs(
        checkpoint_path,
        device=device,
    )

    all_examples = []
    for context_length in ("1k", "2k", "4k", "8k"):
        all_examples.extend(
            load_babilong_file(
                repository_root / f"data/raw/babilong/qa1/{context_length}.json",
                task_id="qa1",
                split="test",
            )
        )
    examples = _select_development_slice(
        all_examples,
        vocabulary,
        examples_per_bucket=args.examples_per_bucket,
    )
    if not examples:
        raise ValueError("the selected evaluation slice is empty")

    results = []
    for baseline in BASELINE_NAMES:
        print(f"evaluating {baseline}...", flush=True)
        result = evaluate_qa1_baseline(
            model,
            vocabulary,
            examples,
            baseline=baseline,
            capacity=args.capacity,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
        )
        results.append(result)
        print(
            f"{baseline}: outside-window accuracy "
            f"{result.outside_window_accuracy:.4f}",
            flush=True,
        )

    shared_memory_bytes = require_equal_memory_budget(results)
    oracle_accuracy = next(
        result.outside_window_accuracy
        for result in results
        if result.baseline == "oracle"
    )
    oracle_dominates = oracle_accuracy > max(
        result.outside_window_accuracy
        for result in results
        if result.baseline != "oracle"
    )

    memory_config = replace(
        checkpoint_config.memory,
        n_slots=args.capacity,
        codes_per_write=min(
            checkpoint_config.memory.codes_per_write,
            args.capacity,
        ),
    )
    evaluation_config = replace(
        checkpoint_config,
        seed=args.seed,
        memory=memory_config,
    )
    commit = current_git_commit(repository_root)
    run_directory = create_run_directory(
        repository_root / args.artifact_root,
        evaluation_config,
        git_commit=commit,
    )
    result_document = {
        "status": (
            "development_single_seed"
            if args.examples_per_bucket == 0
            else "development_slice_single_seed"
        ),
        "task_id": "qa1",
        "device": str(device),
        "seed": args.seed,
        "git_commit": commit,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "manifest_sha256": json.loads(
            (repository_root / "data/installed.lock.json").read_text()
        )["manifest_sha256"],
        "capacity": args.capacity,
        "shared_memory_bytes_per_example": shared_memory_bytes,
        "examples": len(examples),
        "examples_per_bucket": args.examples_per_bucket,
        "oracle_dominates": oracle_dominates,
        "baselines": [result.to_dict() for result in results],
    }
    (run_directory / "results.json").write_text(
        json.dumps(result_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_results(results, run_directory / "baseline_comparison.png")
    print(json.dumps(result_document, indent=2, sort_keys=True))
    print(f"artifacts: {run_directory}")


if __name__ == "__main__":
    main()
