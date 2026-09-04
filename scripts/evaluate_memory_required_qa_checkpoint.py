#!/usr/bin/env python3
"""Evaluate a frozen causal byte-memory checkpoint under new conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch

from tinymem.data.babi import load_babi_file
from tinymem.data.memory_required_qa import (
    DISTRACTOR_TEXT_BANKS,
    build_memory_required_qa_examples,
)
from tinymem.data.sampling import select_reasoning_examples
from tinymem.evaluation.memory_required_qa import (
    MemoryRequiredQAConditionResult,
    evaluate_memory_required_qa,
)
from tinymem.memory.continuous import MeanPoolMemoryCompressor
from tinymem.memory.recurrent_memory import RecurrentMemoryBank
from tinymem.model.config import ExperimentConfig
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.tokenization.byte_tokenizer import ByteTokenizer
from tinymem.training.checkpointing import load_checkpoint
from tinymem.utils.device import select_device
from tinymem.utils.experiment import create_run_directory, current_git_source_state
from tinymem.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-examples", type=int, default=100)
    parser.add_argument("--validation-seed", type=int, default=10_000)
    parser.add_argument(
        "--distractor-variant",
        choices=tuple(DISTRACTOR_TEXT_BANKS),
        default="heldout",
    )
    parser.add_argument("--memory-capacity", type=int)
    parser.add_argument(
        "--distractor-write-policy",
        choices=("all", "none"),
        default="all",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/predictions/memory_required_qa_frozen"),
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
    memory_capacity: int | None,
    device: torch.device | str,
) -> tuple[
    SegmentedContinuousDecoder,
    ExperimentConfig,
    dict[str, object],
    int,
    int,
]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError("checkpoint must contain an experiment configuration")
    extra = payload.get("extra")
    if not isinstance(extra, dict):
        raise ValueError("checkpoint must contain architecture metadata")
    if extra.get("architecture") != "segmented_memory_required_byte_qa":
        raise ValueError("checkpoint is not a causal memory-required QA model")
    if extra.get("mode") != "learned":
        raise ValueError("frozen evaluation requires a learned-memory checkpoint")
    if extra.get("compressor") != "mean" or extra.get("memory_update") != "fifo":
        raise ValueError("checkpoint must use the mean-pool FIFO memory path")
    if extra.get("memory_position_mode") != "virtual":
        raise ValueError("checkpoint must use virtual memory positions")

    config = ExperimentConfig.from_dict(payload["config"])
    if config.model.vocab_size != ByteTokenizer.vocab_size:
        raise ValueError("checkpoint vocabulary does not match the byte tokenizer")
    segment_length = extra.get("segment_length")
    if isinstance(segment_length, bool) or not isinstance(segment_length, int):
        raise ValueError("checkpoint is missing its segment length")
    checkpoint_capacity = extra.get("memory_capacity")
    if isinstance(checkpoint_capacity, bool) or not isinstance(
        checkpoint_capacity,
        int,
    ):
        raise ValueError("checkpoint is missing its memory capacity")
    evaluation_capacity = (
        checkpoint_capacity if memory_capacity is None else int(memory_capacity)
    )
    if evaluation_capacity <= 0:
        raise ValueError("memory capacity must be positive")
    checkpoint_step = payload.get("step", 0)
    if (
        isinstance(checkpoint_step, bool)
        or not isinstance(checkpoint_step, int)
        or checkpoint_step < 0
    ):
        raise ValueError("checkpoint step must be a nonnegative integer")

    decoder = SegmentedContinuousDecoder(
        DecoderOnlyTransformer(config.model),
        MeanPoolMemoryCompressor(config.model.d_model),
        RecurrentMemoryBank(
            capacity=evaluation_capacity,
            model_width=config.model.d_model,
        ),
        segment_length=segment_length,
        memory_position_mode="virtual",
    ).to(device)
    load_checkpoint(checkpoint_path, model=decoder, map_location=device)
    return decoder, config, extra, evaluation_capacity, checkpoint_step


def _passes_gate(results: dict[str, MemoryRequiredQAConditionResult]) -> bool:
    normal = results["normal"].exact_accuracy
    return normal >= 0.8 and all(
        normal - results[condition].exact_accuracy >= 0.3
        for condition in ("drop", "zero", "shuffle", "no_writes")
    )


def main() -> None:
    args = parse_args()
    for name in ("validation_examples", "max_new_tokens"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.validation_seed < 0:
        raise ValueError("validation_seed must be nonnegative")
    if args.memory_capacity is not None and args.memory_capacity <= 0:
        raise ValueError("memory_capacity must be positive")

    repository_root = Path(__file__).resolve().parents[1]
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    device = select_device(args.device)
    decoder, config, extra, evaluation_capacity, checkpoint_step = _load_decoder(
        checkpoint_path,
        memory_capacity=args.memory_capacity,
        device=device,
    )
    seed_everything(config.seed)
    distractor_segments = extra.get("distractor_segments")
    if isinstance(distractor_segments, bool) or not isinstance(
        distractor_segments,
        int,
    ):
        raise ValueError("checkpoint is missing its distractor count")

    babi_root = repository_root / "data/raw/tasks_1-20_v1-2/en-valid-10k"
    validation_source = load_babi_file(
        babi_root / "qa1_valid.txt",
        task_id="qa1",
        split="validation",
    )
    examples = build_memory_required_qa_examples(
        select_reasoning_examples(
            validation_source,
            count=args.validation_examples,
            seed=args.validation_seed,
        ),
        ByteTokenizer(),
        segment_length=decoder.segment_length,
        distractor_segments=distractor_segments,
        distractor_variant=args.distractor_variant,
    )
    results = evaluate_memory_required_qa(
        decoder,
        examples,
        mode="learned",
        device=device,
        max_new_tokens=args.max_new_tokens,
        write_distractors=args.distractor_write_policy == "all",
    )

    source_state = current_git_source_state(repository_root)
    evaluation_config = replace(
        config,
        memory=replace(config.memory, n_slots=evaluation_capacity),
    )
    run_directory = create_run_directory(
        repository_root
        / args.artifact_root
        / f"distractors_{distractor_segments}"
        / f"capacity_{evaluation_capacity}"
        / f"writes_{args.distractor_write_policy}"
        / args.distractor_variant
        / f"seed_{config.seed}",
        evaluation_config,
        git_commit=source_state.commit,
        source_state=source_state,
    )
    document = {
        "status": "development_frozen_memory_required_qa",
        "gate_passed": _passes_gate(results),
        "seed": config.seed,
        "validation_seed": args.validation_seed,
        "device": str(device),
        "source_state": source_state.to_dict(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "checkpoint_memory_capacity": extra["memory_capacity"],
        "evaluation_memory_capacity": evaluation_capacity,
        "distractor_segments": distractor_segments,
        "distractor_write_policy": args.distractor_write_policy,
        "training_distractor_variant": extra.get(
            "train_distractor_variant",
            "trained",
        ),
        "evaluation_distractor_variant": args.distractor_variant,
        "validation_examples": len(examples),
        "validation_data_sha256": _sha256(babi_root / "qa1_valid.txt"),
        "after": {name: result.to_dict() for name, result in results.items()},
    }
    (run_directory / "results.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "gate_passed": document["gate_passed"],
                "seed": config.seed,
                "distractor_variant": args.distractor_variant,
                "memory_capacity": evaluation_capacity,
                "distractor_write_policy": args.distractor_write_policy,
                "after": {
                    name: {
                        "exact_accuracy": result.exact_accuracy,
                        "first_byte_accuracy": result.first_byte_accuracy,
                        "first_byte_logit_linf": (
                            result.mean_first_byte_logit_linf_from_normal
                        ),
                    }
                    for name, result in results.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"artifacts: {run_directory}")


if __name__ == "__main__":
    main()
