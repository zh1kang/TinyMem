#!/usr/bin/env python3
"""Fine-tune and evaluate the first continuous-memory qa1 model."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch

from tinymem.data.babi import load_babi_file
from tinymem.data.babilong import load_babilong_file
from tinymem.data.vocabulary import SPECIAL_TOKENS, ControlledVocabulary
from tinymem.evaluation.continuous_memory import (
    drop_memory,
    evaluate_continuous_qa1,
    shuffle_memory,
    zero_memory,
)
from tinymem.memory.continuous import MeanPoolMemoryCompressor
from tinymem.memory.recurrent_memory import RecurrentMemoryBank
from tinymem.model.config import ExperimentConfig
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.training.checkpointing import load_checkpoint, save_checkpoint
from tinymem.training.continuous import train_continuous_answer_supervision
from tinymem.training.controlled_qa import encode_qa_example
from tinymem.utils.device import select_device
from tinymem.utils.experiment import create_run_directory, current_git_commit
from tinymem.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--segment-length", type=int, default=128)
    parser.add_argument("--capacity", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--max-eval-examples",
        type=int,
        default=0,
        help="zero uses all examples; a positive value makes a smoke-test slice",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/predictions/continuous_memory"),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_base_checkpoint(
    path: Path,
    *,
    device: torch.device,
) -> tuple[ExperimentConfig, DecoderOnlyTransformer, ControlledVocabulary]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("base checkpoint must contain a dictionary")
    config_values = payload.get("config")
    if not isinstance(config_values, dict):
        raise ValueError("base checkpoint must contain an experiment config")
    config = ExperimentConfig.from_dict(config_values)
    extra = payload.get("extra")
    if not isinstance(extra, dict):
        raise ValueError("base checkpoint must contain extra metadata")
    tokens = extra.get("vocabulary")
    if not isinstance(tokens, list) or not all(
        isinstance(token, str) for token in tokens
    ):
        raise ValueError("base checkpoint must contain its vocabulary")
    if tuple(tokens[: len(SPECIAL_TOKENS)]) != SPECIAL_TOKENS:
        raise ValueError("base checkpoint vocabulary has invalid special tokens")
    vocabulary = ControlledVocabulary(tokens[len(SPECIAL_TOKENS) :])
    if vocabulary.id_to_token != tuple(tokens):
        raise ValueError("base checkpoint vocabulary is not in standard order")

    model = DecoderOnlyTransformer(config.model).to(device)
    load_checkpoint(path, model=model, map_location=device)
    return config, model, vocabulary


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("steps and training batch size must be positive")
    if args.eval_batch_size <= 1:
        raise ValueError("evaluation batch size must be greater than one")
    if args.segment_length <= 0 or args.capacity <= 0 or args.seed < 0:
        raise ValueError("segment length and capacity must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning rate must be positive and weight decay nonnegative")
    if args.gradient_clip_norm <= 0:
        raise ValueError("gradient clip norm must be positive")
    if args.max_eval_examples < 0:
        raise ValueError("max evaluation examples must be nonnegative")

    repository_root = Path(__file__).resolve().parents[1]
    checkpoint_path = args.base_checkpoint.resolve()
    device = select_device(args.device)
    seed_everything(args.seed)
    base_config, model, vocabulary = _load_base_checkpoint(
        checkpoint_path,
        device=device,
    )
    if args.segment_length > model.config.max_local_tokens:
        raise ValueError("segment length must not exceed the base local window")

    decoder = SegmentedContinuousDecoder(
        model,
        MeanPoolMemoryCompressor(model.config.d_model),
        RecurrentMemoryBank(
            capacity=args.capacity,
            model_width=model.config.d_model,
        ),
        segment_length=args.segment_length,
    ).to(device)
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    babi_root = repository_root / "data/raw/tasks_1-20_v1-2/en-valid-10k"
    train_examples = load_babi_file(
        babi_root / "qa1_train.txt",
        task_id="qa1",
        split="train",
    )
    encoded_train = [
        encode_qa_example(example, vocabulary)
        for example in train_examples
    ]
    memory_curriculum = [
        example
        for example in encoded_train
        if len(example.input_ids) > args.segment_length
    ]
    if not memory_curriculum:
        raise ValueError("no training examples cross a segment boundary")

    losses = train_continuous_answer_supervision(
        decoder,
        optimizer,
        memory_curriculum,
        steps=args.steps,
        batch_size=args.batch_size,
        gradient_clip_norm=args.gradient_clip_norm,
        pad_id=vocabulary.token_to_id["<pad>"],
        device=device,
        seed=args.seed,
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
    if args.max_eval_examples:
        babilong_examples = babilong_examples[: args.max_eval_examples]
    evaluations = []
    for name, intervention in (
        ("normal", None),
        ("drop", drop_memory),
        ("zero", zero_memory),
        ("shuffle", shuffle_memory),
    ):
        print(f"evaluating {name} memory...", flush=True)
        evaluations.append(
            evaluate_continuous_qa1(
                decoder,
                vocabulary,
                babilong_examples,
                batch_size=args.eval_batch_size,
                device=device,
                intervention_name=name,
                memory_intervention=intervention,
            )
        )

    normal = evaluations[0]
    counterfactual_utility = {
        result.intervention: (
            normal.outside_window_accuracy - result.outside_window_accuracy
        )
        for result in evaluations[1:]
    }
    config = replace(
        base_config,
        seed=args.seed,
        stream=replace(
            base_config.stream,
            segment_length=args.segment_length,
        ),
        memory=replace(
            base_config.memory,
            n_slots=args.capacity,
            codes_per_write=min(
                base_config.memory.codes_per_write,
                args.capacity,
            ),
        ),
        training=replace(
            base_config.training,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            gradient_clip_norm=args.gradient_clip_norm,
            warmup_steps=0,
            max_steps=args.steps,
        ),
    )
    commit = current_git_commit(repository_root)
    run_directory = create_run_directory(
        repository_root / args.artifact_root,
        config,
        git_commit=commit,
    )
    result_document = {
        "status": "development_single_seed",
        "task_id": "qa1",
        "device": str(device),
        "seed": args.seed,
        "git_commit": commit,
        "base_checkpoint": str(checkpoint_path),
        "base_checkpoint_sha256": _sha256(checkpoint_path),
        "manifest_sha256": json.loads(
            (repository_root / "data/installed.lock.json").read_text()
        )["manifest_sha256"],
        "training_examples": len(memory_curriculum),
        "training_steps": args.steps,
        "segment_length": args.segment_length,
        "capacity": args.capacity,
        "memory_bytes_per_example": args.capacity
        * (
            model.config.d_model
            * model.token_embedding.weight.element_size()
            + 1
            + 8
        ),
        "final_training_loss": losses[-1],
        "evaluations": [result.to_dict() for result in evaluations],
        "outside_window_counterfactual_utility": counterfactual_utility,
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
            "vocabulary": list(vocabulary.id_to_token),
            "architecture": "segmented_continuous_mean_pool",
        },
    )
    print(json.dumps(result_document, indent=2, sort_keys=True))
    print(f"artifacts: {run_directory}")


if __name__ == "__main__":
    main()
