"""Atomic model and optimizer checkpoint persistence."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer

from tinymem.model.config import ExperimentConfig


CHECKPOINT_FORMAT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    step: int = 0,
    config: ExperimentConfig | None = None,
    extra: Mapping[str, object] | None = None,
) -> None:
    """Atomically save the state required to resume an experiment."""
    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError("step must be an integer")
    if step < 0:
        raise ValueError("step must be nonnegative")
    if config is not None and not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig or None")
    if extra is not None and not isinstance(extra, Mapping):
        raise TypeError("extra must be a mapping or None")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "config": config.to_dict() if config is not None else None,
        "extra": dict(extra) if extra is not None else {},
    }

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        torch.save(payload, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    """Load checkpoint state into a model and optional optimizer."""
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a dictionary")
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format version")
    if "model_state" not in payload:
        raise ValueError("checkpoint is missing model state")

    model.load_state_dict(payload["model_state"])
    optimizer_state = payload.get("optimizer_state")
    if optimizer is not None:
        if optimizer_state is None:
            raise ValueError("checkpoint does not contain optimizer state")
        optimizer.load_state_dict(optimizer_state)

    return {
        "step": payload.get("step", 0),
        "config": payload.get("config"),
        "extra": payload.get("extra", {}),
    }
