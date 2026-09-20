"""Frozen inputs, full-epoch schedules, and completion seals for phase two."""

from dataclasses import asdict
import json
from pathlib import Path
import random
import shutil

from tinymem.research.delta_fact_data import Episode, FactDataset, Statement, build_dataset, validate_dataset
from tinymem.research.delta_fact_profile import file_hash, write_json


def settings(*, device: str, smoke: bool = False) -> dict:
    if device not in ("cpu", "mps", "cuda") or type(smoke) is not bool:
        raise ValueError("invalid execution mode")
    return {
        "purpose": "implementation_smoke" if smoke else "fixed_phase_two_comparison",
        "device": device, "data_seed": 771731 if smoke else 2026091407,
        "train_prefixes": 16 if smoke else 256, "validation_prefixes": 16 if smoke else 32,
        "test_prefixes": 16 if smoke else 64, "epochs": 1 if smoke else 4,
        "batch_size": 1 if smoke else 4, "schedule_seed": 86291,
        "smoke_steps": 4 if smoke else None, "evaluation_prefix_limit": 1 if smoke else None,
        "seeds": [991] if smoke else [3101, 3102, 3103],
        "widths": [8] if smoke else [8, 32], "writers": ["delta"] if smoke else ["gated", "delta"],
        "lora_rank": 8, "learning_rate": 0.001, "weight_decay": 0.01, "clip_norm": 1.0,
        "training_endpoints": [8, 16], "evaluation_endpoints": [8, 9, 16],
        "primary_endpoint": 16, "primary_condition": "repeat", "primary_scope": "unspoken",
        "probe_fit": "training_prefix_states_only; four prefix-group folds",
        "probe_alphas": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        "checkpoint_selection": "fixed_final", "accuracy_exclusion": False,
        "failure_policy": "retain failed attempt; no automatic retry, seed replacement, or recipe change; no final evaluation while training is incomplete",
        "max_new_tokens": 8, "bootstrap_samples": 10000, "bootstrap_seed": 8237,
        "interval": "99% pointwise paired-prefix bootstrap conditional on these seeds",
        "controls": ["zero_values_valid_slots", "other_prefix_memory", "one_byte_exact", "base_full_text"],
    }


def cells(spec: dict) -> list[dict]:
    return [{"index": i, "writer": writer, "width": width, "writer_seed": seed,
             "bridge_seed": seed + 1, "adapter_seed": seed + 10000,
             "persistent_bytes": 2 * width * 4 + 2}
            for i, (seed, width, writer) in enumerate(
                (s, w, k) for s in spec["seeds"] for w in spec["widths"] for k in spec["writers"])]


def epoch_schedule(episodes, spec: dict) -> list[list[list[str]]]:
    ids = [e.id for e in episodes]
    if not ids or len(set(ids)) != len(ids) or any(e.split != "train" for e in episodes):
        raise ValueError("schedule requires unique training episodes")
    if len(ids) % spec["batch_size"]:
        raise ValueError("every epoch must contain complete batches")
    result = []
    for epoch in range(spec["epochs"]):
        order = ids.copy()
        random.Random(spec["schedule_seed"] + epoch).shuffle(order)
        batches = [order[i:i + spec["batch_size"]] for i in range(0, len(order), spec["batch_size"])]
        if spec["smoke_steps"] is not None:
            batches = batches[:spec["smoke_steps"]]
        elif sorted(item for batch in batches for item in batch) != sorted(ids):
            raise ValueError("epoch must include every training episode exactly once")
        result.append(batches)
    return result


def read_dataset(path: Path) -> FactDataset:
    raw = json.loads(path.read_text())
    if set(raw) != {"train", "validation", "test"}:
        raise ValueError("dataset split schema differs")
    splits = {}
    for split, records in raw.items():
        splits[split] = tuple(Episode(**{**row,
            "prefix": tuple(Statement(**s) for s in row["prefix"]),
            "tail": tuple(Statement(**s) for s in row["tail"])}) for row in records)
    dataset = FactDataset(**splits)
    validate_dataset(dataset)
    return dataset


def source_files(root: Path) -> list[Path]:
    return sorted([p for directory in ("src", "scripts", "tests")
                   for p in (root / directory).rglob("*") if p.suffix in (".py", ".slurm")]
                  + [root / "pyproject.toml", root / "uv.lock"])


def prepare_study(root: Path, output: Path, snapshot: dict, spec: dict) -> dict:
    expected = settings(device=spec["device"], smoke=spec["purpose"] == "implementation_smoke")
    if spec != expected:
        raise ValueError("settings differ from the declared procedure")
    dataset = build_dataset(seed=spec["data_seed"], train_prefixes=spec["train_prefixes"],
                            validation_prefixes=spec["validation_prefixes"], test_prefixes=spec["test_prefixes"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "dataset.json", asdict(dataset))
    sources = {}
    for path in source_files(root):
        name = str(path.relative_to(root))
        sources[name] = file_hash(path)
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if file_hash(target) != sources[name]:
            raise ValueError("source changed during copy: " + name)
    protocol = {"kind": "delta_fact_study_v1", "settings": spec, "cells": cells(spec),
                "schedule": epoch_schedule(dataset.train, spec), "snapshot": snapshot,
                "dataset_sha256": file_hash(output / "dataset.json"), "source_sha256": sources}
    write_json(output / "protocol.json", protocol)
    return protocol


def verify_study(study: Path, root: Path) -> tuple[dict, FactDataset]:
    protocol = json.loads((study / "protocol.json").read_text())
    spec = protocol["settings"]
    if (protocol["kind"] != "delta_fact_study_v1"
            or spec != settings(device=spec["device"], smoke=spec["purpose"] == "implementation_smoke")
            or protocol["cells"] != cells(spec)):
        raise ValueError("study procedure differs")
    if file_hash(study / "dataset.json") != protocol["dataset_sha256"]:
        raise ValueError("study data changed")
    dataset = read_dataset(study / "dataset.json")
    expected_data = build_dataset(seed=spec["data_seed"], train_prefixes=spec["train_prefixes"],
                                 validation_prefixes=spec["validation_prefixes"], test_prefixes=spec["test_prefixes"])
    if dataset != expected_data:
        raise ValueError("dataset differs from the fixed seed and counts")
    if protocol["schedule"] != epoch_schedule(dataset.train, spec):
        raise ValueError("training schedule changed")
    actual = {str(p.relative_to(root)): file_hash(p) for p in source_files(root)}
    if actual != protocol["source_sha256"]:
        raise ValueError("execution sources differ from the frozen study")
    for name, digest in actual.items():
        if file_hash(study / "source" / name) != digest:
            raise ValueError("copied execution source changed: " + name)
    return protocol, dataset


def seal_directory(directory: Path, identity: dict) -> None:
    marker = directory / "complete.json"
    if marker.exists():
        raise FileExistsError(marker)
    files = {str(p.relative_to(directory)): file_hash(p) for p in sorted(directory.rglob("*")) if p.is_file()}
    write_json(marker, {"identity": identity, "files": files})


def verify_completion(directory: Path, identity: dict) -> dict:
    marker = json.loads((directory / "complete.json").read_text())
    if marker["identity"] != identity:
        raise ValueError("completion belongs to a different study or cell")
    actual = {str(p.relative_to(directory)): file_hash(p) for p in sorted(directory.rglob("*"))
              if p.is_file() and p != directory / "complete.json"}
    if marker["files"] != actual:
        raise ValueError("completed outputs changed")
    return marker


def cell_identity(study: Path, protocol: dict, index: int, stage: str) -> dict:
    if type(index) is not int or not 0 <= index < len(protocol["cells"]):
        raise ValueError("cell index is outside the declaration")
    return {"protocol_sha256": file_hash(study / "protocol.json"), "cell": protocol["cells"][index], "stage": stage}


def seal_training(study: Path, protocol: dict) -> None:
    marker = study / "training_sealed.json"
    if marker.exists():
        raise FileExistsError(marker)
    digests = {}
    runtime = None
    base = None
    for cell in protocol["cells"]:
        directory = study / "training" / str(cell["index"])
        verify_training_result(study, protocol, cell["index"])
        report = json.loads((directory / "report.json").read_text())
        if runtime is not None and (report["runtime"] != runtime or report["base_before_sha256"] != base):
            raise ValueError("training cells used different base models or execution environments")
        runtime, base = report["runtime"], report["base_before_sha256"]
        digests[str(cell["index"])] = file_hash(directory / "complete.json")
    write_json(marker, {"protocol_sha256": file_hash(study / "protocol.json"), "training_completions": digests})


def require_training_seal(study: Path, protocol: dict) -> None:
    seal = json.loads((study / "training_sealed.json").read_text())
    if seal["protocol_sha256"] != file_hash(study / "protocol.json"):
        raise ValueError("training seal belongs to a different protocol")
    if set(seal["training_completions"]) != {str(c["index"]) for c in protocol["cells"]}:
        raise ValueError("training seal omits a cell")
    for cell in protocol["cells"]:
        directory = study / "training" / str(cell["index"])
        verify_training_result(study, protocol, cell["index"])
        if file_hash(directory / "complete.json") != seal["training_completions"][str(cell["index"])]:
            raise ValueError("training completion changed after sealing")


def verify_training_result(study: Path, protocol: dict, index: int) -> None:
    directory = study / "training" / str(index)
    marker = verify_completion(directory, cell_identity(study, protocol, index, "training"))
    required = {"checkpoint.safetensors", "probe.json", "report.json", "metrics.jsonl", "curves.json",
                "features.safetensors", "feature_texts.json", "training_states.safetensors", "training_states.json"}
    if not required.issubset(marker["files"]):
        raise ValueError("training completion omits a required output")
    report = json.loads((directory / "report.json").read_text())
    expected_steps = sum(len(epoch) for epoch in protocol["schedule"])
    if (report["optimizer_steps"] != expected_steps or report["epochs"] != len(protocol["schedule"])
            or report["base_before_sha256"] != report["base_after_sha256"] or report["test_scored"] is not False
            or report["checkpoint_selection"] != "fixed_final"
            or report["persistent_bytes"] != protocol["cells"][index]["persistent_bytes"]):
        raise ValueError("training result differs from its fixed endpoint")
    runtime = report["runtime"]
    if protocol["settings"]["purpose"] != "implementation_smoke" and (
            runtime["device"] != protocol["settings"]["device"] or runtime["reader_dtype"] != "torch.bfloat16"
            or runtime["deterministic"] is not True or runtime["tf32_matmul"] is not False):
        raise ValueError("training runtime differs from the numerical contract")
    metrics = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    if [row["step"] for row in metrics] != list(range(1, expected_steps + 1)):
        raise ValueError("training metrics omit or repeat optimizer steps")
