"""CPU fitting and persistence for state-supervised distilled writers."""

from __future__ import annotations

import importlib.metadata
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from tinymem.memory.delta_slots import DeltaSlotWriter
from tinymem.research.delta_fact_evaluation import collect_states
from tinymem.research.delta_fact_fit import save_state_records
from tinymem.research.delta_fact_profile import file_hash, write_json
from tinymem.research.distilled_fact_protocol import (
    cell_identity,
    load_features,
    require_features,
    seal_directory,
)
from tinymem.research.distilled_fact_training import (
    DistilledExample,
    aggregate_validation,
    build_example,
    train_batch,
    trajectory_diagnostics,
)


def _settings(protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    settings = protocol.get("settings")
    if not isinstance(settings, Mapping):
        raise TypeError("protocol settings are required")
    required = {
        "writer_device", "writer_hidden_width", "key_width", "loss", "learning_rate",
        "weight_decay", "clip_norm", "epochs", "batch_size",
    }
    missing = sorted(required.difference(settings))
    if missing:
        raise ValueError("protocol settings omit: " + ", ".join(missing))
    if settings["writer_device"] != "cpu":
        raise ValueError("distilled writer training requires writer_device='cpu'")
    if settings["loss"] != "mean_histories_mean_writes_sum_64_squared_error":
        raise ValueError("protocol loss does not match full-state supervision")
    fixed_beta = settings.get("fixed_beta")
    if fixed_beta is not None and fixed_beta != 0.75:
        raise ValueError("distilled fixed-beta studies require fixed_beta=0.75")
    normalize_hidden = settings.get("normalize_hidden", False)
    if type(normalize_hidden) is not bool:
        raise TypeError("normalize_hidden must be a bool")
    if normalize_hidden:
        if fixed_beta != 0.75 or settings.get("hidden_normalization") != {
            "kind": "layer_norm",
            "width": 64,
            "eps": 1e-5,
            "elementwise_affine": False,
            "placement": "preGELU",
        }:
            raise ValueError("hidden normalization identity differs from the declared procedure")
    elif "hidden_normalization" in settings:
        raise ValueError("hidden normalization identity requires normalize_hidden=true")
    return settings


def _cell(protocol: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    cells = protocol.get("cells")
    if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes)):
        raise TypeError("protocol cells are required")
    if type(index) is not int or not 0 <= index < len(cells):
        raise ValueError("cell index is outside the protocol")
    cell = cells[index]
    if not isinstance(cell, Mapping) or cell.get("index") != index:
        raise ValueError("cell index does not match its declaration")
    if cell.get("persistent_bytes") != 258:
        raise ValueError("distilled writer cells must declare 258 persistent bytes")
    if type(cell.get("seed")) is not int:
        raise ValueError("distilled writer cells require an integer seed")
    return cell


def _feature_inputs(study: Path, protocol: Mapping[str, Any]) -> tuple[dict[str, torch.Tensor], Mapping[str, Any]]:
    report = require_features(study, protocol)
    features = load_features(study, protocol)
    if not isinstance(features, Mapping) or not isinstance(report, Mapping):
        raise TypeError("feature protocol helpers must return mappings")
    reader_width = report.get("reader_width")
    if type(reader_width) is not int or reader_width <= 0:
        raise ValueError("feature report must declare a positive reader_width")
    for text, feature in features.items():
        if (not isinstance(text, str) or not isinstance(feature, torch.Tensor)
                or feature.device.type != "cpu" or feature.dtype != torch.float32
                or feature.ndim != 2 or feature.shape[0] == 0 or feature.shape[1] != reader_width
                or feature.requires_grad or feature.grad_fn is not None
                or not bool(torch.isfinite(feature).all())):
            raise ValueError("feature cache contains an invalid detached CPU FP32 feature")
    return {text: feature.detach().clone().contiguous() for text, feature in features.items()}, report


def _writer(reader_width: int, settings: Mapping[str, Any], seed: int) -> DeltaSlotWriter:
    if type(reader_width) is not int or reader_width <= 0:
        raise ValueError("reader_width must be a positive integer")
    if settings["writer_hidden_width"] != 64 or settings["key_width"] != 8:
        raise ValueError("distilled writer dimensions differ from the declared architecture")
    torch.manual_seed(seed)
    writer = DeltaSlotWriter(
        reader_width,
        32,
        key_width=8,
        hidden_width=64,
        fixed_beta=settings.get("fixed_beta"),
        normalize_hidden=settings.get("normalize_hidden", False),
    ).to(device="cpu", dtype=torch.float32)
    return writer


def _episode_examples(
    episodes: Sequence[Any], features: Mapping[str, torch.Tensor], writer: DeltaSlotWriter, expected_split: str,
) -> dict[str, DistilledExample]:
    result = {}
    for episode in episodes:
        if episode.split != expected_split:
            raise ValueError(f"expected {expected_split} episodes")
        if episode.id in result:
            raise ValueError("episode IDs must be unique")
        result[episode.id] = build_example(episode, features, writer)
    return result


def _runtime() -> dict[str, Any]:
    return {
        "device": "cpu",
        "writer_device": "cpu",
        "writer_dtype": "torch.float32",
        "torch_version": torch.__version__,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "threads": torch.get_num_threads(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "safetensors", "numpy")
        },
    }


def _feature_completion(study: Path) -> str:
    marker = study / "features" / "complete.json"
    if not marker.is_file():
        raise ValueError("feature completion marker is missing")
    return file_hash(marker)


def _validate_schedule(protocol: Mapping[str, Any], episode_ids: set[str], settings: Mapping[str, Any]) -> list[list[list[str]]]:
    schedule = protocol.get("schedule")
    if not isinstance(schedule, list) or len(schedule) != settings["epochs"]:
        raise ValueError("protocol schedule does not match its epoch count")
    normalized: list[list[list[str]]] = []
    for epoch in schedule:
        if not isinstance(epoch, list):
            raise TypeError("each schedule epoch must be a list")
        batches: list[list[str]] = []
        seen: list[str] = []
        for batch in epoch:
            if not isinstance(batch, list) or len(batch) != settings["batch_size"]:
                raise ValueError("schedule contains an incomplete batch")
            if any(not isinstance(identifier, str) or identifier not in episode_ids for identifier in batch):
                raise ValueError("schedule references a non-training episode")
            batches.append(list(batch))
            seen.extend(batch)
        smoke_steps = settings.get("smoke_steps")
        if smoke_steps is None:
            if sorted(seen) != sorted(episode_ids):
                raise ValueError("each full epoch must contain every training episode once")
        elif len(batches) != smoke_steps or len(set(seen)) != len(seen):
            raise ValueError("smoke schedule does not match its fixed step limit")
        normalized.append(batches)
    return normalized


def _selected_validation(episodes: Sequence[Any], settings: Mapping[str, Any]) -> tuple[Any, ...]:
    if any(episode.split != "validation" for episode in episodes):
        raise ValueError("validation diagnostics require validation episodes")
    limit = settings.get("evaluation_prefix_limit")
    if limit is None:
        return tuple(episodes)
    if type(limit) is not int or limit <= 0:
        raise ValueError("evaluation_prefix_limit must be positive or None")
    prefix_ids = sorted({episode.prefix_id for episode in episodes})[:limit]
    return tuple(episode for episode in episodes if episode.prefix_id in prefix_ids)


def train_cell(study: Path, protocol: dict, dataset: Any, index: int) -> dict:
    """Train one writer cell on frozen features and save its fixed final state."""

    study = Path(study)
    settings = _settings(protocol)
    cell = _cell(protocol, index)
    if not hasattr(dataset, "train") or not hasattr(dataset, "validation"):
        raise TypeError("dataset must expose train and validation episode groups")
    directory = study / "training" / str(index)
    directory.mkdir(parents=True, exist_ok=False)
    features, feature_report = _feature_inputs(study, protocol)
    reader_width = int(feature_report["reader_width"])
    writer = _writer(reader_width, settings, int(cell["seed"]))
    train_examples = _episode_examples(dataset.train, features, writer, "train")
    validation_episodes = _selected_validation(dataset.validation, settings)
    schedule = _validate_schedule(protocol, set(train_examples), settings)
    optimizer = torch.optim.AdamW(
        writer.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"])
    )
    curves: list[dict[str, Any]] = []
    started = time.perf_counter()
    step = 0
    with (directory / "metrics.jsonl").open("x") as metrics_file:
        for epoch, batches in enumerate(schedule, start=1):
            epoch_metrics: list[dict[str, Any]] = []
            writer.train()
            for batch_ids in batches:
                batch_started = time.perf_counter()
                metric = train_batch(
                    writer,
                    tuple(train_examples[identifier] for identifier in batch_ids),
                    optimizer,
                    clip_norm=float(settings["clip_norm"]),
                )
                step += 1
                row = {"step": step, "epoch": epoch, "seconds": time.perf_counter() - batch_started, **metric}
                metrics_file.write(json.dumps(row, allow_nan=False) + "\n")
                metrics_file.flush()
                epoch_metrics.append(metric)
                if step == 1 or step % 25 == 0:
                    print(json.dumps({"cell": index, **row}, allow_nan=False), flush=True)
            writer.zero_grad(set_to_none=True)
            writer.eval()
            writer.requires_grad_(False)
            validation_rows = trajectory_diagnostics(
                writer, validation_episodes, features,
            )
            validation = aggregate_validation(validation_rows)
            curves.append({
                "epoch": epoch,
                "training_state_loss": sum(float(row["state_loss"]) for row in epoch_metrics) / len(epoch_metrics),
                "training_off_fact_loss": sum(float(row["off_fact_loss"]) for row in epoch_metrics) / len(epoch_metrics),
                "validation": validation,
            })
            write_json(directory / "curves.json", curves)
            if epoch != len(schedule):
                writer.requires_grad_(True)
                writer.train()
    writer.eval()
    writer.requires_grad_(False)
    writer.zero_grad(set_to_none=True)
    elapsed = time.perf_counter() - started
    state_records = collect_states(
        writer,
        tuple(episode for episode in dataset.train if episode.condition == "no_write"),
        features,
    )
    save_state_records(state_records, directory / "training_states.safetensors")
    checkpoint = {name: tensor.detach().cpu().contiguous() for name, tensor in writer.state_dict().items()}
    save_file(checkpoint, str(directory / "checkpoint.safetensors"))
    report = {
        "optimizer_steps": step,
        "epochs": len(schedule),
        "runtime": _runtime(),
        "training_seconds": elapsed,
        "test_scored": False,
        "checkpoint_selection": "fixed_final",
        "persistent_bytes": 258,
        "reader_trained": False,
        "parameters": {"writer": sum(parameter.numel() for parameter in writer.parameters())},
        "features_completion_sha256": _feature_completion(study),
        "validation": curves,
    }
    write_json(directory / "report.json", report)
    seal_directory(directory, cell_identity(study, protocol, index, "training"))
    return report


def load_writer(study: Path, protocol: dict, index: int) -> DeltaSlotWriter:
    """Load a fixed-final writer as a frozen CPU FP32 module."""

    settings = _settings(protocol)
    cell = _cell(protocol, index)
    feature_report = require_features(Path(study), protocol)
    writer = _writer(int(feature_report["reader_width"]), settings, int(cell["seed"]))
    path = Path(study) / "training" / str(index) / "checkpoint.safetensors"
    if not path.is_file():
        raise ValueError("writer checkpoint is missing")
    tensors = load_file(str(path), device="cpu")
    expected = writer.state_dict()
    if set(tensors) != set(expected):
        raise ValueError("writer checkpoint parameter schema differs")
    for name, expected_tensor in expected.items():
        tensor = tensors[name]
        if (tensor.shape != expected_tensor.shape or tensor.dtype != expected_tensor.dtype
                or tensor.device.type != "cpu" or not bool(torch.isfinite(tensor).all())):
            raise ValueError("writer checkpoint shape, dtype, device, or finite-value contract differs")
    writer.load_state_dict({name: tensor.contiguous() for name, tensor in tensors.items()}, strict=True)
    writer.zero_grad(set_to_none=True)
    writer.requires_grad_(False).eval()
    return writer
