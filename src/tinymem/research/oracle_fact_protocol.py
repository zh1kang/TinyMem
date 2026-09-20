"""Fixed declaration and seals for the privileged correct-state diagnostic."""

from dataclasses import asdict
import json
from pathlib import Path
import shutil

import torch

from tinymem.research.delta_fact_data import build_dataset
from tinymem.research.delta_fact_fit import load_state_records
from tinymem.research.delta_fact_profile import file_hash, write_json
from tinymem.research.delta_fact_protocol import (
    cell_identity, epoch_schedule, read_dataset, settings as comparison_settings,
    source_files, verify_completion,
)
from tinymem.research.oracle_fact_state import oracle_records


def settings(*, device: str, smoke: bool = False) -> dict:
    previous = comparison_settings(device=device, smoke=smoke)
    shared = ("device", "data_seed", "train_prefixes", "validation_prefixes", "test_prefixes",
              "epochs", "batch_size", "schedule_seed", "smoke_steps", "evaluation_prefix_limit",
              "seeds", "lora_rank", "learning_rate", "weight_decay", "clip_norm",
              "training_endpoints", "evaluation_endpoints", "checkpoint_selection",
              "accuracy_exclusion", "failure_policy", "max_new_tokens", "bootstrap_samples",
              "bootstrap_seed", "interval")
    return {**{key: previous[key] for key in shared},
            "batch_size": 4,
            "purpose": "implementation_smoke" if smoke else "privileged_correct_state_diagnostic",
            "panel_status": "previously inspected diagnostic panel; not fresh confirmation",
            "memory_width": 32, "persistent_bytes": 258,
            "state_code": {"matrix_shape": [8, 8], "entity_rows": [0, 1, 2, 3],
                           "value_column": 0, "value_targets": [-0.5, 0.5], "beta": 0.75,
                           "layout": "row-major two slots; facts in first slot; both valid"},
            "write_inputs": "current statement text through exact parser; no query; no learned writer",
            "trainable": ["affine_bridge", "rank8_q_v_lora"],
            "controls": ["zero_values_valid_slots", "other_prefix_memory"],
            "primary_readouts": [{"condition": "repeat", "scope": "unspoken", "after_write": 16},
                                 {"condition": "correction", "scope": "target", "after_write": 16}],
            "wording_interpretation": "parser removes wording; paired states and predictions must agree"}


def cells(spec: dict) -> list[dict]:
    return [{"index": index, "seed": seed, "bridge_seed": seed + 1,
             "adapter_seed": seed + 10000, "persistent_bytes": 258}
            for index, seed in enumerate(spec["seeds"])]


def prepare_study(root: Path, output: Path, snapshot: dict, spec: dict) -> dict:
    if spec != settings(device=spec["device"], smoke=spec["purpose"] == "implementation_smoke"):
        raise ValueError("settings differ from the declared oracle diagnostic")
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
    protocol = {"kind": "oracle_fact_readout_v1", "settings": spec, "cells": cells(spec),
                "schedule": epoch_schedule(dataset.train, spec), "snapshot": snapshot,
                "dataset_sha256": file_hash(output / "dataset.json"), "source_sha256": sources}
    write_json(output / "protocol.json", protocol)
    return protocol


def verify_study(study: Path, root: Path):
    protocol = json.loads((study / "protocol.json").read_text())
    spec = protocol["settings"]
    if (protocol["kind"] != "oracle_fact_readout_v1"
            or spec != settings(device=spec["device"], smoke=spec["purpose"] == "implementation_smoke")
            or protocol["cells"] != cells(spec)):
        raise ValueError("oracle diagnostic procedure differs")
    if file_hash(study / "dataset.json") != protocol["dataset_sha256"]:
        raise ValueError("study data changed")
    dataset = read_dataset(study / "dataset.json")
    expected = build_dataset(seed=spec["data_seed"], train_prefixes=spec["train_prefixes"],
                            validation_prefixes=spec["validation_prefixes"], test_prefixes=spec["test_prefixes"])
    if dataset != expected or protocol["schedule"] != epoch_schedule(dataset.train, spec):
        raise ValueError("data or schedule differs from the declaration")
    actual = {str(p.relative_to(root)): file_hash(p) for p in source_files(root)}
    if actual != protocol["source_sha256"]:
        raise ValueError("execution sources differ from the frozen diagnostic")
    for name, digest in actual.items():
        if file_hash(study / "source" / name) != digest:
            raise ValueError("copied execution source changed: " + name)
    return protocol, dataset


def verify_training_result(study: Path, protocol: dict, index: int) -> dict:
    directory = study / "training" / str(index)
    marker = verify_completion(directory, cell_identity(study, protocol, index, "training"))
    required = {"checkpoint.safetensors", "report.json", "metrics.jsonl", "curves.json",
                "training_states.safetensors", "training_states.json"}
    if set(marker["files"]) != required:
        raise ValueError("training output schema differs")
    report = json.loads((directory / "report.json").read_text())
    steps = sum(len(epoch) for epoch in protocol["schedule"])
    if (report["optimizer_steps"] != steps or report["epochs"] != len(protocol["schedule"])
            or report["base_before_sha256"] != report["base_after_sha256"]
            or report["test_scored"] is not False or report["checkpoint_selection"] != "fixed_final"
            or report["persistent_bytes"] != 258 or report["parameters"]["writer"] != 0):
        raise ValueError("training result differs from the fixed endpoint")
    runtime = report["runtime"]
    if runtime["device"] != protocol["settings"]["device"]:
        raise ValueError("training device differs")
    if protocol["settings"]["purpose"] != "implementation_smoke" and (
            runtime["reader_dtype"] != "torch.bfloat16" or runtime["deterministic"] is not True
            or runtime["tf32_matmul"] is not False):
        raise ValueError("training runtime differs from the numerical contract")
    metrics = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    if [row["step"] for row in metrics] != list(range(1, steps + 1)):
        raise ValueError("training metrics omit or repeat steps")
    curves = json.loads((directory / "curves.json").read_text())
    if [row["epoch"] for row in curves] != list(range(1, report["epochs"] + 1)):
        raise ValueError("validation curves omit or repeat epochs")
    dataset = read_dataset(study / "dataset.json")
    expected_states = oracle_records(tuple(e for e in dataset.train if e.condition == "no_write"))
    saved_states = load_state_records(directory / "training_states.safetensors")
    if len(saved_states) != len(expected_states):
        raise ValueError("saved training states omit or add prefixes")
    for saved, expected in zip(saved_states, expected_states, strict=True):
        for field in expected.__dataclass_fields__:
            actual, wanted = getattr(saved, field), getattr(expected, field)
            equal = torch.equal(actual, wanted) if isinstance(wanted, torch.Tensor) else actual == wanted
            if not equal:
                raise ValueError("saved training state differs from the declared oracle: " + field)
    return report


def seal_training(study: Path, protocol: dict) -> None:
    marker = study / "training_sealed.json"
    if marker.exists():
        raise FileExistsError(marker)
    reports = [verify_training_result(study, protocol, cell["index"]) for cell in protocol["cells"]]
    for report in reports[1:]:
        for field in ("runtime", "base_before_sha256", "unadapted_base_sha256"):
            if report[field] != reports[0][field]:
                raise ValueError("training cells used different base models or execution environments")
    write_json(marker, {"protocol_sha256": file_hash(study / "protocol.json"),
                       "training_completions": {str(c["index"]): file_hash(
                           study / "training" / str(c["index"]) / "complete.json") for c in protocol["cells"]}})


def require_training_seal(study: Path, protocol: dict) -> None:
    seal = json.loads((study / "training_sealed.json").read_text())
    if (seal["protocol_sha256"] != file_hash(study / "protocol.json")
            or set(seal["training_completions"]) != {str(c["index"]) for c in protocol["cells"]}):
        raise ValueError("training seal differs from the declared cells")
    for cell in protocol["cells"]:
        verify_training_result(study, protocol, cell["index"])
        digest = file_hash(study / "training" / str(cell["index"]) / "complete.json")
        if digest != seal["training_completions"][str(cell["index"])]:
            raise ValueError("training completion changed after sealing")
