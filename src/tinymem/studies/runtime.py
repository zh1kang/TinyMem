"""Deterministic device setup shared by every study runner."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import torch

from tinymem.utils.seed import seed_everything


REPOSITORY = Path(__file__).resolve().parents[3]


def check_repository() -> None:
    if sys.flags.optimize:
        raise RuntimeError("do not use python -O: these study runners require assertions")
    if Path.cwd().resolve() != REPOSITORY:
        raise RuntimeError(f"run this command from the repository root: {REPOSITORY}")


def prepare_device(name: str) -> torch.device:
    """Require the requested backend, with no silent CPU fallback."""
    check_repository()
    if name not in ("cuda", "mps", "cpu"):
        raise ValueError("device must be cuda, mps, or cpu")
    if name == "cuda":
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8") != ":4096:8":
            raise RuntimeError("set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python")
        if torch.cuda.is_initialized() and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
            raise RuntimeError("set CUBLAS_WORKSPACE_CONFIG before initializing CUDA")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use a GPU allocation and a CUDA PyTorch build")
        if not torch.cuda.is_bf16_supported() or torch.cuda.get_device_capability()[0] < 8:
            raise RuntimeError("this study requires native BF16 CUDA support (Ampere or newer)")
    elif name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable on this machine")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    seed_everything(1337)
    return torch.device(name)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def allocation_metrics(device: torch.device) -> dict[str, int]:
    if device.type == "cuda":
        return {"cuda_allocated_bytes": torch.cuda.memory_allocated(device),
                "cuda_reserved_bytes": torch.cuda.memory_reserved(device),
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if device.type == "mps":
        return {"mps_allocated_bytes": torch.mps.current_allocated_memory(),
                "mps_driver_bytes": torch.mps.driver_allocated_memory()}
    return {}
