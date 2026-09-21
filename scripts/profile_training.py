"""Measure phase-two training execution without scoring validation or test data."""

import argparse
from dataclasses import asdict
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import shutil
import time

import torch
from safetensors.torch import save_file

from tinymem.reader.adapter import configure_read_adapter
from tinymem.studies.delta.data import build_dataset
from tinymem.studies.delta.encoding import build_feature_cache, encode_episode
from tinymem.studies.artifacts import CELLS, file_hash, frozen_base_hash, profile_cell, write_json
from tinymem.reader.pretrained import load_qwen_reader, verify_qwen_snapshot
from tinymem.reader.lora import attach_reader_lora
from tinymem.studies.runtime import REPOSITORY, prepare_device, synchronize


def source_paths() -> list[Path]:
    paths = [p for directory in ("src", "scripts", "tests")
             for p in (REPOSITORY / directory).rglob("*.py")]
    return sorted([*paths, REPOSITORY / "pyproject.toml", REPOSITORY / "uv.lock"])


def training_schedule(episodes, *, steps: int, batch_size: int, seed: int):
    if type(steps) is not int or type(batch_size) is not int or steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch size must be positive integers")
    if not episodes or any(e.split != "train" for e in episodes):
        raise ValueError("profile schedule accepts training episodes only")
    rng = random.Random(seed)
    conditions = ("no_write", "repeat", "correction", "balanced")
    pools = {condition: [e for e in episodes if e.condition == condition] for condition in conditions}
    if any(not pool for pool in pools.values()):
        raise ValueError("training schedule requires all four conditions")
    for pool in pools.values():
        rng.shuffle(pool)
    return tuple(tuple(pools[conditions[step % 4]][(step // 4 * batch_size + j) % len(pools[conditions[step % 4]])]
                       for j in range(batch_size)) for step in range(steps))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.steps < 4 or not 0 <= args.warmup_steps < args.steps or args.batch_size <= 0:
        parser.error("use at least four steps, positive batch size, and fewer warmup steps than total steps")
    if args.output.exists():
        raise FileExistsError(args.output)
    settings = {
        "purpose": "training_execution_and_compute_only", "evaluation_scored": False,
        "checkpoint_reuse": False, "cells": CELLS, "steps": args.steps,
        "warmup_steps": args.warmup_steps, "batch_size": args.batch_size,
        "data_seed": 2026091301, "schedule_seed": 631, "writer_seed": 937,
        "bridge_seed": 938, "adapter_seed": 419, "lora_rank": 8,
        "learning_rate": 0.001, "weight_decay": 0.01, "clip_norm": 1.0,
        "loss": "equal episode weight; equal endpoint-query weight within episode",
        "write_features": "frozen base; independent statement; position starts at zero",
        "global_decay": False, "base_trainable": False, "gradient_checkpointing": False,
    }
    device = prepare_device(args.device)
    torch.set_num_threads(4)
    snapshot = verify_qwen_snapshot(args.snapshot)
    sources = {str(p.relative_to(REPOSITORY)): file_hash(p) for p in source_paths()}
    dataset = build_dataset(seed=settings["data_seed"], train_prefixes=16, validation_prefixes=16, test_prefixes=16)
    schedule = training_schedule(dataset.train, steps=args.steps, batch_size=args.batch_size,
                                 seed=settings["schedule_seed"])
    selected = {e.id: e for batch in schedule for e in batch}
    args.output.mkdir(parents=True)
    write_json(args.output / "training_inputs.json", [asdict(selected[key]) for key in sorted(selected)])
    for name in sources:
        target = args.output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPOSITORY / name, target)
        if file_hash(target) != sources[name]:
            raise ValueError("source changed while creating the snapshot: " + name)
    runtime = {
        "device": device.type, "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "peft", "numpy", "safetensors", "tokenizers")},
        "reader_dtype": "bfloat16", "writer_dtype": "float32",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "attention_implementation": "sdpa", "cuda_build": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine(),
    }
    protocol = {"kind": "delta_fact_training_profile_v1", "settings": settings,
                "snapshot": snapshot, "source_sha256": sources, "runtime": runtime,
                "training_input_sha256": file_hash(args.output / "training_inputs.json"),
                "schedule": [[e.id for e in batch] for batch in schedule]}
    write_json(args.output / "protocol.json", protocol)
    print(json.dumps({"status": "loading_verified_reader", "device": device.type}), flush=True)
    reader = load_qwen_reader(args.snapshot, device=device, dtype=torch.bfloat16)
    start = time.perf_counter()
    features = build_feature_cache(reader, tuple(selected.values()))
    synchronize(device)
    feature_seconds = time.perf_counter() - start
    texts = sorted(features)
    save_file({str(i): features[text].contiguous() for i, text in enumerate(texts)},
              str(args.output / "training_features.safetensors"))
    write_json(args.output / "feature_texts.json", texts)
    encoded = {key: encode_episode(reader, episode, features) for key, episode in selected.items()}
    batches = tuple(tuple(encoded[e.id] for e in batch) for batch in schedule)
    torch.manual_seed(settings["adapter_seed"])
    attach_reader_lora(reader, rank=settings["lora_rank"], checkpointing=False)
    configure_read_adapter(reader, trainable=True)
    initial_adapter = {name: p.detach().clone() for name, p in reader.model.named_parameters() if p.requires_grad}
    base_hash = frozen_base_hash(reader)
    summaries = {}
    for kind, width in CELLS:
        with torch.no_grad():
            for name, parameter in reader.model.named_parameters():
                if name in initial_adapter:
                    parameter.copy_(initial_adapter[name])
        cell = f"{kind}_{2 * width * 4 + 2}"
        summaries[cell] = profile_cell(reader, batches, args.output / cell, kind=kind, width=width,
                                       seed=settings["writer_seed"], warmup_steps=args.warmup_steps,
                                       learning_rate=settings["learning_rate"])
    if frozen_base_hash(reader) != base_hash:
        raise ValueError("profile changed the frozen base model")
    if {str(p.relative_to(REPOSITORY)): file_hash(p) for p in source_paths()} != sources:
        raise ValueError("sources changed during the profile")
    if any(not math.isfinite(row["mean_measured_step_seconds"]) for row in summaries.values()):
        raise ValueError("invalid profile timing")
    write_json(args.output / "report.json", {
        "kind": protocol["kind"], "status": "complete", "accuracy_scored": False,
        "checkpoint_reuse": False, "base_sha256_unchanged": base_hash,
        "feature_extraction_seconds": feature_seconds, "unique_training_statements": len(features),
        "cells": summaries,
    })
    write_json(args.output / "complete.json", {
        "files": {str(p.relative_to(args.output)): file_hash(p) for p in sorted(args.output.rglob("*"))
                  if p.is_file()},
    })
    print(json.dumps({"status": "complete", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
