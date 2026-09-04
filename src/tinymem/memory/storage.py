"""Measure retained tensor allocations, including backing storage of views."""

from collections.abc import Iterable

import torch


def tensor_storage_bytes(tensors: Iterable[torch.Tensor]) -> int:
    """Count unique dense allocations once, not just the visible tensor elements."""
    allocations = {}
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("state must contain tensors")
        if tensor.layout != torch.strided or tensor.device.type == "meta":
            raise ValueError("storage accounting requires materialized dense tensors")
        storage = tensor.untyped_storage()
        key = (tensor.device, storage.data_ptr())
        allocations[key] = storage.nbytes()
    return sum(allocations.values())
