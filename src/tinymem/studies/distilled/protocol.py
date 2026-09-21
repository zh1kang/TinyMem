"""Fixed inputs and completion seals for supervised writing into frozen readers."""

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from tinymem.studies.distilled import replication
from tinymem.studies.oracle import protocol as parent_protocol
from tinymem.studies.delta.data import build_dataset
from tinymem.studies.delta.encoding import _cached_feature, build_feature_cache
from tinymem.studies.delta.fit import execution_record, load_state_records
from tinymem.studies.artifacts import file_hash, frozen_base_hash, write_json
from tinymem.studies.delta.protocol import (
    cell_identity,
    epoch_schedule,
    read_dataset,
    seal_directory,
    source_files,
    verify_completion,
)


def settings(*, device: str, smoke: bool = False, fixed_beta: float | None = None,
             normalize_hidden: bool = False, replicate: bool = False) -> dict:
    if type(normalize_hidden) is not bool:
        raise TypeError("normalize_hidden must be a bool")
    spec = parent_protocol.settings(device=device, smoke=smoke)
    result = {**spec,
            "purpose": "implementation_smoke" if smoke else "learned_writer_state_supervision",
            "writer_device": "cpu", "writer_hidden_width": 64, "key_width": 8,
            "loss": "mean_histories_mean_writes_sum_64_squared_error",
            "training_endpoints": "every write, including prefix initialization",
            "write_inputs": "current frozen base statement features and own preceding predicted state",
            "trainable": ["delta_writer"],
            "frozen": ["base_reader", "parent_affine_bridge", "parent_rank8_q_v_lora"],
            "supervision": "train-only full oracle trajectory; all 64 coordinates; no head targets",
            "rollout": "own preceding state with gradients through every write",
            "evaluation_state": "raw CPU FP32 serialized writer output; no rounding, repair, or projection",
            "direct_bits": "negative=0, positive=1, exact zero undefined and incorrect for known facts",
            "controls": ["canonical_oracle", "zero_values_valid_slots", "other_prefix_learned_memory"],
            "wording_interpretation": "learned familiar and heldout wordings may differ; paired by prefix",
            "seed_selection": "all paired parent readers; smoke uses the first parent reader",
            "claim_limit": "privileged supervised writer feasibility; no storage advantage claim"}
    if fixed_beta is not None:
        if type(fixed_beta) not in (float, int) or float(fixed_beta) != 0.75:
            raise ValueError("distilled fixed-beta studies require fixed_beta=0.75")
        result["fixed_beta"] = 0.75
    if normalize_hidden:
        if fixed_beta != 0.75:
            raise ValueError("hidden normalization studies require fixed_beta=0.75")
        result["normalize_hidden"] = True
        result["hidden_normalization"] = {
            "kind": "layer_norm",
            "width": 64,
            "eps": 1e-5,
            "elementwise_affine": False,
            "placement": "preGELU",
        }
    if type(replicate) is not bool:
        raise TypeError("replicate must be a bool")
    if replicate:
        if fixed_beta != 0.75:
            raise ValueError("replication requires fixed_beta=0.75")
        result["replication"] = replication.declaration(smoke=smoke)
        result["seeds"] = result["replication"]["writer_seeds"]
        result["seed_selection"] = "fresh writer seeds; every writer evaluated through every parent reader"
        result["panel_status"] = "fresh logical test histories and reserved heldout templates"
        result["interval"] = "auxiliary within-arm 99% paired-prefix intervals; primary paired-arm 95% writer-seed intervals"
    return result


def cells(spec: dict, parent: dict) -> list[dict]:
    selected = parent["cells"][:1] if spec["purpose"] == "implementation_smoke" else parent["cells"]
    if "replication" in spec:
        if spec["purpose"] != "implementation_smoke" and [cell["seed"] for cell in selected] != [3101, 3102, 3103]:
            raise ValueError("replication requires the declared parent readers")
        return [{"index": index, "seed": seed, "persistent_bytes": 258}
                for index, seed in enumerate(spec["seeds"])]
    if spec["purpose"] != "implementation_smoke" and (
            parent["settings"]["purpose"] != "privileged_correct_state_diagnostic"
            or [cell["seed"] for cell in selected] != spec["seeds"]):
        raise ValueError("full study requires all three declared parent readers")
    return [{"index": index, "seed": cell["seed"], "parent_index": cell["index"],
             "persistent_bytes": 258} for index, cell in enumerate(selected)]


def evaluation_identity(study: Path, protocol: dict, index: int) -> dict:
    return cell_identity(study, {**protocol, "cells": replication.evaluation_cells(protocol)},
                         index, "evaluation")


def _dataset(spec: dict):
    original = build_dataset(seed=spec["data_seed"], train_prefixes=spec["train_prefixes"],
                             validation_prefixes=spec["validation_prefixes"], test_prefixes=spec["test_prefixes"])
    if "replication" in spec:
        return replication.build_replication_dataset(original, smoke=spec["purpose"] == "implementation_smoke")
    return original


def _inventory(directory: Path) -> dict[str, str]:
    return {str(path.relative_to(directory)): file_hash(path)
            for path in sorted(directory.rglob("*")) if path.is_file()}


def _parent(study: Path) -> dict:
    parent, _ = parent_protocol.verify_study(study, study / "source")
    parent_protocol.require_training_seal(study, parent)
    return parent


def prepare_study(root: Path, output: Path, snapshot: dict, spec: dict, parent: Path) -> dict:
    if spec != settings(device=spec["device"], smoke=spec["purpose"] == "implementation_smoke",
                        fixed_beta=spec.get("fixed_beta"),
                        normalize_hidden=spec.get("normalize_hidden", False), replicate="replication" in spec):
        raise ValueError("settings differ from the declared supervised writer study")
    old = _parent(parent)
    selected = cells(spec, old)
    if snapshot != old["snapshot"]:
        raise ValueError("parent and writer study must use the same base snapshot")
    if spec["device"] != old["settings"]["device"]:
        raise ValueError("parent and writer study must use the same reader device")
    dataset = _dataset(spec)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "dataset.json", asdict(dataset))
    parent_files = _inventory(parent)
    shutil.copytree(parent, output / "parent")
    if _inventory(output / "parent") != parent_files:
        raise ValueError("parent changed while copying")
    if _parent(output / "parent") != old:
        raise ValueError("copied parent differs from the validated reader study")
    sources = {}
    for path in source_files(root):
        name = str(path.relative_to(root))
        sources[name] = file_hash(path)
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if file_hash(target) != sources[name]:
            raise ValueError("source changed during copy: " + name)
    protocol = {"kind": "distilled_fact_writer_v1", "settings": spec, "cells": selected,
                "schedule": epoch_schedule(dataset.train, spec), "snapshot": snapshot,
                "dataset_sha256": file_hash(output / "dataset.json"), "source_sha256": sources,
                "parent_protocol_sha256": file_hash(parent / "protocol.json"),
                "parent_files_sha256": parent_files}
    if "replication" in spec:
        protocol["kind"] = "distilled_fact_replication_v1"
        protocol["readers"] = [{"index": cell["index"], "seed": cell["seed"]}
                               for cell in old["cells"]]
    write_json(output / "protocol.json", protocol)
    return protocol


def verify_study(study: Path, root: Path):
    protocol = json.loads((study / "protocol.json").read_text())
    spec = protocol["settings"]
    expected_kind = "distilled_fact_replication_v1" if "replication" in spec else "distilled_fact_writer_v1"
    if (protocol["kind"] != expected_kind
            or spec != settings(device=spec["device"], smoke=spec["purpose"] == "implementation_smoke",
                                fixed_beta=spec.get("fixed_beta"),
                                normalize_hidden=spec.get("normalize_hidden", False),
                                replicate="replication" in spec)):
        raise ValueError("supervised writer procedure differs")
    if (_inventory(study / "parent") != protocol["parent_files_sha256"]
            or file_hash(study / "parent/protocol.json") != protocol["parent_protocol_sha256"]):
        raise ValueError("frozen parent study changed")
    parent = _parent(study / "parent")
    if protocol["snapshot"] != parent["snapshot"] or protocol["cells"] != cells(spec, parent):
        raise ValueError("parent reader selection or snapshot differs")
    if "replication" in spec and protocol.get("readers") != [
            {"index": cell["index"], "seed": cell["seed"]} for cell in parent["cells"]]:
        raise ValueError("replication reader selection differs")
    if file_hash(study / "dataset.json") != protocol["dataset_sha256"]:
        raise ValueError("study data changed")
    dataset = read_dataset(study / "dataset.json")
    expected = _dataset(spec)
    if dataset != expected or protocol["schedule"] != epoch_schedule(dataset.train, spec):
        raise ValueError("data or schedule differs from the declaration")
    actual = {str(path.relative_to(root)): file_hash(path) for path in source_files(root)}
    if actual != protocol["source_sha256"]:
        raise ValueError("execution sources differ from the frozen study")
    copied = {str(path.relative_to(study / "source")): file_hash(path)
              for path in source_files(study / "source")}
    if copied != protocol["source_sha256"]:
        raise ValueError("copied execution source inventory differs")
    return protocol, dataset


def _feature_identity(study: Path) -> dict:
    return {"protocol_sha256": file_hash(study / "protocol.json"), "stage": "features"}


def _feature_texts(dataset) -> list[str]:
    return sorted({statement.text for episode in (*dataset.train, *dataset.validation)
                   for statement in (*episode.prefix, *episode.tail)})


def prepare_features(reader, study: Path, protocol: dict, dataset) -> dict:
    directory = study / "features"
    directory.mkdir(parents=True, exist_ok=False)
    if reader.model.device.type != protocol["settings"]["device"]:
        raise ValueError("feature reader device differs")
    before = frozen_base_hash(reader)
    parent_report = json.loads((study / "parent/training/0/report.json").read_text())
    if before != parent_report["unadapted_base_sha256"]:
        raise ValueError("feature reader base differs from the parent")
    features = build_feature_cache(reader, (*dataset.train, *dataset.validation))
    texts = _feature_texts(dataset)
    if set(features) != set(texts) or frozen_base_hash(reader) != before:
        raise ValueError("feature extraction changed the base or statement inventory")
    write_json(directory / "feature_texts.json", texts)
    save_file({str(index): features[text] for index, text in enumerate(texts)},
              str(directory / "features.safetensors"))
    report = {"reader_width": reader.model.config.hidden_size, "texts": len(texts),
              "splits": ["train", "validation"], "unadapted_base_sha256": before,
              "runtime": execution_record(reader), "test_scored": False}
    write_json(directory / "report.json", report)
    seal_directory(directory, _feature_identity(study))
    require_features(study, protocol)
    return report


def require_features(study: Path, protocol: dict) -> dict:
    directory = study / "features"
    marker = verify_completion(directory, _feature_identity(study))
    if set(marker["files"]) != {"features.safetensors", "feature_texts.json", "report.json"}:
        raise ValueError("feature output schema differs")
    report = json.loads((directory / "report.json").read_text())
    texts = json.loads((directory / "feature_texts.json").read_text())
    if texts != _feature_texts(read_dataset(study / "dataset.json")):
        raise ValueError("feature texts differ from train and validation statements")
    parent = json.loads((study / "parent/training/0/report.json").read_text())
    if (report["splits"] != ["train", "validation"] or report["test_scored"] is not False
            or report["texts"] != len(texts) or type(report["reader_width"]) is not int
            or report["reader_width"] <= 0
            or report["unadapted_base_sha256"] != parent["unadapted_base_sha256"]
            or report["runtime"]["device"] != protocol["settings"]["device"]):
        raise ValueError("feature provenance differs")
    if protocol["settings"]["purpose"] != "implementation_smoke" and report["runtime"] != parent["runtime"]:
        raise ValueError("feature runtime differs from the parent reader")
    tensors = load_file(str(directory / "features.safetensors"))
    if set(tensors) != {str(index) for index in range(len(texts))}:
        raise ValueError("feature tensor inventory differs")
    for index, text in enumerate(texts):
        _cached_feature({text: tensors[str(index)]}, text, report["reader_width"])
    return report


def load_features(study: Path, protocol: dict) -> dict[str, torch.Tensor]:
    report = require_features(study, protocol)
    directory = study / "features"
    texts = json.loads((directory / "feature_texts.json").read_text())
    tensors = load_file(str(directory / "features.safetensors"))
    return {text: _cached_feature({text: tensors[str(index)]}, text, report["reader_width"])
            for index, text in enumerate(texts)}


def verify_training_result(study: Path, protocol: dict, index: int) -> dict:
    require_features(study, protocol)
    directory = study / "training" / str(index)
    marker = verify_completion(directory, cell_identity(study, protocol, index, "training"))
    if set(marker["files"]) != {"checkpoint.safetensors", "report.json", "metrics.jsonl", "curves.json",
                                "training_states.safetensors", "training_states.json"}:
        raise ValueError("training output schema differs")
    report = json.loads((directory / "report.json").read_text())
    steps = sum(len(epoch) for epoch in protocol["schedule"])
    if (report["optimizer_steps"] != steps or report["epochs"] != len(protocol["schedule"])
            or report["test_scored"] is not False or report["checkpoint_selection"] != "fixed_final"
            or report["persistent_bytes"] != 258 or report["reader_trained"] is not False
            or report["features_completion_sha256"] != file_hash(study / "features/complete.json")):
        raise ValueError("training result differs from the fixed endpoint")
    runtime = report["runtime"]
    if runtime["device"] != "cpu" or runtime["writer_dtype"] != "torch.float32":
        raise ValueError("writer training runtime differs")
    if protocol["settings"]["purpose"] != "implementation_smoke" and runtime["deterministic"] is not True:
        raise ValueError("writer training must use deterministic operations")
    from tinymem.studies.distilled.fit import load_writer

    writer = load_writer(study, protocol, index)
    if report["parameters"] != {"writer": sum(parameter.numel() for parameter in writer.parameters())}:
        raise ValueError("training parameter count differs from the declared writer")
    metrics = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    if [row["step"] for row in metrics] != list(range(1, steps + 1)):
        raise ValueError("training metrics omit or repeat steps")
    curves = json.loads((directory / "curves.json").read_text())
    if [row["epoch"] for row in curves] != list(range(1, report["epochs"] + 1)):
        raise ValueError("validation curves omit or repeat epochs")
    dataset = read_dataset(study / "dataset.json")
    expected = {episode.id: episode for episode in dataset.train if episode.condition == "no_write"}
    records = load_state_records(directory / "training_states.safetensors")
    if len(records) != len(expected) or {record.episode_id for record in records} != set(expected):
        raise ValueError("training states omit or add prefixes")
    for record in records:
        episode = expected[record.episode_id]
        if (record.after_write != 8 or record.split != "train" or record.prefix_id != episode.prefix_id
                or record.wording != episode.wording or record.condition != "no_write"
                or record.values.shape != (1, 2, 32) or record.values.dtype != torch.float32
                or record.valid.shape != (1, 2) or record.valid.dtype != torch.bool
                or not bool(record.valid.all()) or not bool(torch.isfinite(record.values).all())):
            raise ValueError("saved training state is invalid")
    return report


def seal_training(study: Path, protocol: dict) -> None:
    marker = study / "training_sealed.json"
    if marker.exists():
        raise FileExistsError(marker)
    reports = [verify_training_result(study, protocol, cell["index"]) for cell in protocol["cells"]]
    if any(report["runtime"] != reports[0]["runtime"] for report in reports[1:]):
        raise ValueError("training cells used different execution environments")
    write_json(marker, {"protocol_sha256": file_hash(study / "protocol.json"),
                       "training_completions": {str(cell["index"]): file_hash(
                           study / "training" / str(cell["index"]) / "complete.json")
                           for cell in protocol["cells"]}})


def require_training_seal(study: Path, protocol: dict) -> None:
    seal = json.loads((study / "training_sealed.json").read_text())
    if (seal["protocol_sha256"] != file_hash(study / "protocol.json")
            or set(seal["training_completions"]) != {str(cell["index"]) for cell in protocol["cells"]}):
        raise ValueError("training seal differs from the declared cells")
    for cell in protocol["cells"]:
        verify_training_result(study, protocol, cell["index"])
        digest = file_hash(study / "training" / str(cell["index"]) / "complete.json")
        if digest != seal["training_completions"][str(cell["index"])]:
            raise ValueError("training completion changed after sealing")
