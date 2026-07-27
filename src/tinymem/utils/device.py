"""PyTorch compute-device selection."""

import torch


SUPPORTED_DEVICES = ("auto", "cpu", "cuda", "mps")


def select_device(preference: str = "auto") -> torch.device:
    """Select a compute device without hiding unavailable explicit requests."""
    if not isinstance(preference, str):
        raise TypeError("preference must be a string")
    if preference not in SUPPORTED_DEVICES:
        supported = ", ".join(SUPPORTED_DEVICES)
        raise ValueError(f"unsupported device {preference!r}; choose from: {supported}")

    if preference == "cpu":
        return torch.device("cpu")

    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")

    if preference == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
