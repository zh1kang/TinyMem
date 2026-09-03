#!/usr/bin/env python3
"""Evaluate a frozen WikiText TinyMem checkpoint on LongMemEval."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from tinymem.data.longmemeval import load_longmemeval_file
from tinymem.evaluation.continuous_memory import (
    drop_newest_memory,
    keep_oldest_memory,
    replace_newest_memory,
)
from tinymem.evaluation.longmemeval import (
    LongMemEvalResult,
    evaluate_longmemeval,
)
from tinymem.memory.continuous import MeanPoolMemoryCompressor
from tinymem.memory.recurrent_memory import RecurrentMemoryBank
from tinymem.model.config import ExperimentConfig
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.training.checkpointing import load_checkpoint
from tinymem.utils.device import select_device
from tinymem.utils.experiment import create_run_directory, current_git_commit


CONDITIONS = (
    "normal",
    "drop_newest_at_query",
    "stale_at_query",
    "replace_newest_at_query",
    "freeze_all_writes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset-variant",
        choices=("oracle", "small"),
        default="small",
    )
    parser.add_argument(
        "--conditions",
        choices=CONDITIONS,
        nargs="+",
        default=("normal",),
    )
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--examples-per-type", type=int, default=0)
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
        default=Path("artifacts/predictions/longmemeval"),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_decoder(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[SegmentedContinuousDecoder, ExperimentConfig]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError("checkpoint must contain an experiment configuration")
    extra = payload.get("extra")
    if not isinstance(extra, dict):
        raise ValueError("checkpoint must contain architecture metadata")
    if extra.get("architecture") != "segmented_continuous_wikitext_byte_lm":
        raise ValueError("checkpoint is not a WikiText segmented language model")
    selected_window = extra.get("selected_window")
    if isinstance(selected_window, bool) or not isinstance(selected_window, int):
        raise ValueError("checkpoint is missing its selected local window")

    config = ExperimentConfig.from_dict(payload["config"])
    decoder = SegmentedContinuousDecoder(
        DecoderOnlyTransformer(config.model),
        MeanPoolMemoryCompressor(config.model.d_model),
        RecurrentMemoryBank(
            capacity=config.memory.n_slots,
            model_width=config.model.d_model,
        ),
        segment_length=selected_window,
    ).to(device)
    load_checkpoint(
        checkpoint_path,
        model=decoder,
        map_location=device,
    )
    return decoder, config


def plot_results(
    results: list[LongMemEvalResult],
    destination: Path,
) -> None:
    names = [result.condition for result in results]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(names, [result.overall.exact_accuracy for result in results])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("normalized exact accuracy")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(names, [result.overall.mean_token_f1 for result in results])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("mean token F1")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    device = select_device(args.device)
    decoder, config = _load_decoder(checkpoint_path, device=device)
    if args.chunk_tokens % decoder.segment_length != 0:
        raise ValueError("chunk_tokens must be a multiple of the selected window")

    dataset_name = (
        "longmemeval_oracle.json"
        if args.dataset_variant == "oracle"
        else "longmemeval_s_cleaned.json"
    )
    dataset_path = repository_root / "data/raw/longmemeval" / dataset_name
    examples = load_longmemeval_file(dataset_path)
    if args.max_examples and args.examples_per_type:
        raise ValueError(
            "max_examples and examples_per_type are mutually exclusive"
        )
    if args.max_examples:
        if args.max_examples < 0:
            raise ValueError("max_examples must be nonnegative")
        examples = examples[: args.max_examples]
    if args.examples_per_type:
        if args.examples_per_type < 0:
            raise ValueError("examples_per_type must be nonnegative")
        selected = []
        counts: Counter[str] = Counter()
        for example in examples:
            if counts[example.question_type] < args.examples_per_type:
                selected.append(example)
                counts[example.question_type] += 1
        examples = selected
    if not examples:
        raise ValueError("no LongMemEval examples were selected")

    intervention_by_name = {
        "normal": None,
        "drop_newest_at_query": drop_newest_memory,
        "stale_at_query": keep_oldest_memory,
        "replace_newest_at_query": replace_newest_memory,
        "freeze_all_writes": None,
    }
    results = [
        evaluate_longmemeval(
            decoder,
            examples,
            device=device,
            max_new_tokens=args.max_new_tokens,
            chunk_tokens=args.chunk_tokens,
            condition=condition,
            query_memory_intervention=intervention_by_name[condition],
            update_memory=condition != "freeze_all_writes",
        )
        for condition in dict.fromkeys(args.conditions)
    ]
    lock = json.loads(
        (repository_root / "data/installed.lock.json").read_text(
            encoding="utf-8"
        )
    )
    file_key = f"longmemeval:longmemeval/{dataset_name}"
    installed_file = lock["files"][file_key]
    commit = current_git_commit(repository_root)
    run_directory = create_run_directory(
        repository_root / args.artifact_root,
        config,
        git_commit=commit,
    )
    result_document = {
        "status": (
            "frozen_full_external_test"
            if args.max_examples == 0
            and args.examples_per_type == 0
            and args.dataset_variant == "small"
            else "development_external_evaluation"
        ),
        "dataset": "LongMemEval cleaned",
        "dataset_variant": args.dataset_variant,
        "dataset_revision": (
            "98d7416c24c778c2fee6e6f3006e7a073259d48f"
        ),
        "dataset_sha256": installed_file["sha256"],
        "dataset_sha256_verified": _sha256(dataset_path)
        == installed_file["sha256"],
        "manifest_sha256": lock["manifest_sha256"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "git_commit": commit,
        "device": str(device),
        "example_count": len(examples),
        "max_new_tokens": args.max_new_tokens,
        "chunk_tokens": args.chunk_tokens,
        "answers_used_in_prompts": False,
        "training_on_longmemeval": False,
        "gold_unanswerable_labels": False,
        "abstention_precision_recall": None,
        "results": [result.to_dict() for result in results],
    }
    (run_directory / "results.json").write_text(
        json.dumps(result_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_results(results, run_directory / "longmemeval.png")
    print(
        json.dumps(
            {
                "status": result_document["status"],
                "variant": args.dataset_variant,
                "example_count": len(examples),
                "results": {
                    result.condition: result.overall.to_dict()
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
