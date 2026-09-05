"""Device selection and provenance for relocated opaque-association runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import sys
from typing import Any

import torch

from tinymem.utils.seed import seed_everything


REPOSITORY = Path(__file__).resolve().parents[3]
ORIGINAL_REPOSITORY = Path("/Users/caleb/TinyMem")
PORTABLE_SOURCES = (
    "scripts/opaque/__init__.py",
    "scripts/opaque/train.py",
    "scripts/opaque/baselines.py",
    "scripts/opaque/memory.py",
    "scripts/opaque/training_fit.py",
    "scripts/opaque/oracle.py",
    "scripts/opaque/aggregate.py",
    "scripts/opaque/smoke.py",
    "scripts/report_opaque_study.py",
    "src/tinymem/research/study_runtime.py",
    "src/tinymem/research/native_memory_oracle.py",
    "src/tinymem/evaluation/association_study.py",
    "src/tinymem/evaluation/association_diagnostics.py",
    "src/tinymem/utils/seed.py",
    "src/tinymem/utils/experiment.py",
)


def repository_path(value: str | Path, *, root: Path = REPOSITORY) -> Path:
    """Relocate only the recorded repository prefix; return a relative path."""
    path = Path(value)
    if ".." in path.parts:
        raise ValueError("parent traversal is not a repository path")
    if path.is_absolute():
        if path.is_relative_to(root):
            path = path.relative_to(root)
        elif path.is_relative_to(ORIGINAL_REPOSITORY):
            path = path.relative_to(ORIGINAL_REPOSITORY)
        else:
            raise ValueError(f"path is outside the current or original repository: {value}")
    if not (root / path).resolve().is_relative_to(root.resolve()):
        raise ValueError(f"repository path escapes through a symlink: {value}")
    return path


def sha256(path: str | Path) -> str:
    return hashlib.sha256((REPOSITORY / repository_path(path)).read_bytes()).hexdigest()


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
            raise RuntimeError("CUDA is unavailable; use a GPU Slurm allocation and a CUDA PyTorch build")
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


def execution_record(device: torch.device) -> dict[str, Any]:
    """Separate portable execution identity from the unchanged study protocol."""
    return {
        "version": "opaque_portable_execution_v1",
        "source_sha256": {name: sha256(name) for name in PORTABLE_SOURCES},
        "runtime": {
            "device": device.type, "python": platform.python_version(),
            "platform": platform.system(),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "peft", "numpy", "safetensors", "accelerate", "tokenizers")},
            "cuda_build": torch.version.cuda if device.type == "cuda" else None,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "gpu_capability": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None,
            "reader_dtype": "bfloat16", "writer_dtype": "float32",
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG") if device.type == "cuda" else None,
            "attention_implementation": "sdpa",
        },
    }


def validate_execution(record: dict[str, Any], expected: dict[str, Any] | None = None) -> None:
    if record.get("version") != "opaque_portable_execution_v1":
        raise ValueError("expected a portable execution, not an original MPS result")
    hashes = record.get("source_sha256", {})
    if hashes != {name: sha256(name) for name in PORTABLE_SOURCES}:
        raise ValueError("portable execution sources changed")
    runtime = record.get("runtime")
    fields = {"device", "python", "platform", "packages", "cuda_build", "gpu_name", "gpu_capability",
              "reader_dtype", "writer_dtype", "float32_matmul_precision", "deterministic_algorithms",
              "deterministic_warn_only", "tf32_matmul", "tf32_cudnn", "cublas_workspace_config", "attention_implementation"}
    if not isinstance(runtime, dict) or runtime.keys() != fields:
        raise ValueError("portable execution runtime is incomplete")
    if (runtime["device"] not in ("cuda", "mps", "cpu") or runtime["reader_dtype"] != "bfloat16"
            or runtime["writer_dtype"] != "float32" or runtime["attention_implementation"] != "sdpa"):
        raise ValueError("unsupported execution device, dtype, or attention implementation")
    packages = runtime["packages"]
    if (not isinstance(packages, dict) or packages.keys() != {"torch", "transformers", "peft", "numpy", "safetensors", "accelerate", "tokenizers"}
            or any(not isinstance(value, str) or not value for value in packages.values())):
        raise ValueError("execution package versions are incomplete")
    for name in ("deterministic_algorithms", "deterministic_warn_only", "tf32_matmul", "tf32_cudnn"):
        if not isinstance(runtime[name], bool):
            raise ValueError(f"execution setting must be boolean: {name}")
    if runtime["device"] == "cuda" and (not runtime["cuda_build"] or not runtime["gpu_name"]
            or not isinstance(runtime["gpu_capability"], list) or len(runtime["gpu_capability"]) != 2
            or runtime["cublas_workspace_config"] != ":4096:8"):
        raise ValueError("CUDA runtime metadata is incomplete")
    if expected is not None and record != expected:
        raise ValueError("execution differs: do not mix backends, packages, hardware, or numeric settings")


def attach_execution(protocol: dict[str, Any], sources: list[Path], device: torch.device) -> None:
    protocol["execution"] = execution_record(device)
    seen = {path.resolve() for path in sources}
    for name in PORTABLE_SOURCES:
        path = Path(name)
        if path.resolve() not in seen:
            sources.append(path)
            seen.add(path.resolve())
    if len({path.name for path in sources}) != len(sources):
        raise ValueError("source snapshot basenames collide")
    protocol["source_sha256"] = {str(path): sha256(path) for path in sources}
