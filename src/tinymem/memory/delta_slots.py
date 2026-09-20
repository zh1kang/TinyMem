"""Fixed-size two-token state backed by a differentiable delta memory update."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from tinymem.memory.recurrent_slots import LatentSlotState


def _require_tensor(name: str, value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    return value


def _require_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _validate_fixed_beta(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError("fixed_beta must be a finite float in [0, 1]")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("fixed_beta must be a finite float in [0, 1]")
    return value


def _validate_normalize_hidden(value: bool) -> bool:
    if type(value) is not bool:
        raise TypeError("normalize_hidden must be a bool")
    return value


def delta_update(
    matrix: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Apply one delta-rule write and return a newly allocated matrix.

    The batch equation is ``S' = S + k outer(beta * (v - k @ S))``.
    Inputs must share a floating-point dtype and device.  Float32 and float64
    are supported so that the function can also be used with ``gradcheck``.
    """

    matrix = _require_tensor("matrix", matrix)
    key = _require_tensor("key", key)
    value = _require_tensor("value", value)
    beta = _require_tensor("beta", beta)
    if matrix.ndim != 3 or key.ndim != 2 or value.ndim != 2 or beta.ndim != 1:
        raise ValueError("matrix, key, value, and beta must have ranks 3, 2, 2, and 1")
    batch, key_width, value_width = matrix.shape
    if batch <= 0 or key_width <= 0 or value_width <= 0:
        raise ValueError("matrix dimensions must be positive")
    if key.shape != (batch, key_width):
        raise ValueError("key must have shape [batch, key_width]")
    if value.shape != (batch, value_width):
        raise ValueError("value must have shape [batch, value_width]")
    if beta.shape != (batch,):
        raise ValueError("beta must have shape [batch]")
    tensors = (matrix, key, value, beta)
    if any(not tensor.is_floating_point() for tensor in tensors):
        raise TypeError("matrix, key, value, and beta must be floating point")
    if any(tensor.dtype != matrix.dtype for tensor in tensors):
        raise TypeError("matrix, key, value, and beta must share a dtype")
    if any(tensor.device != matrix.device for tensor in tensors):
        raise ValueError("matrix, key, value, and beta must share a device")
    if matrix.dtype not in (torch.float32, torch.float64):
        raise TypeError("delta_update supports float32 and float64")
    for name, tensor in zip(("matrix", "key", "value", "beta"), tensors):
        _require_finite(name, tensor)
    if bool((beta < 0).any()) or bool((beta > 1).any()):
        raise ValueError("beta must lie in [0, 1]")
    norm = torch.linalg.vector_norm(key, dim=-1)
    tolerance = 16 * torch.finfo(key.dtype).eps
    if bool((norm > 1 + tolerance).any()):
        raise ValueError("key norm must be at most one")

    residual = value - torch.bmm(key.unsqueeze(1), matrix).squeeze(1)
    correction = beta.unsqueeze(-1) * residual
    return matrix + key.unsqueeze(-1) * correction.unsqueeze(1)


class DeltaSlotWriter(nn.Module):
    """Encode each statement into one delta write over two retained tokens."""

    def __init__(
        self,
        reader_width: int,
        memory_width: int,
        *,
        key_width: int,
        hidden_width: int = 64,
        fixed_beta: float | None = None,
        normalize_hidden: bool = False,
    ) -> None:
        super().__init__()
        self.reader_width = _require_positive_int("reader_width", reader_width)
        self.memory_width = _require_positive_int("memory_width", memory_width)
        self.key_width = _require_positive_int("key_width", key_width)
        self.hidden_width = _require_positive_int("hidden_width", hidden_width)
        self.fixed_beta = _validate_fixed_beta(fixed_beta)
        self.normalize_hidden = _validate_normalize_hidden(normalize_hidden)
        total_width = 2 * self.memory_width
        if total_width % self.key_width:
            raise ValueError("2 * memory_width must be divisible by key_width")
        self.slots = 2
        self.value_width = total_width // self.key_width

        self.attention_query = nn.Parameter(torch.empty(self.reader_width))
        self.input_projection = nn.Linear(self.reader_width, self.hidden_width)
        self.key_projection = nn.Linear(self.hidden_width, self.key_width)
        self.value_projection = nn.Linear(self.hidden_width, self.value_width)
        self.beta_projection = nn.Linear(self.hidden_width, 1)
        nn.init.normal_(self.attention_query, std=self.reader_width**-0.5)
        if self.fixed_beta is not None:
            del self.beta_projection

    def empty(self, batch_size: int) -> LatentSlotState:
        batch_size = _require_positive_int("batch_size", batch_size)
        parameter = self.attention_query
        values = torch.zeros(
            batch_size,
            self.slots,
            self.memory_width,
            dtype=torch.float32,
            device=parameter.device,
        )
        valid = torch.zeros(batch_size, self.slots, dtype=torch.bool, device=parameter.device)
        return LatentSlotState(values, valid)

    def _validate_hidden(self, hidden: torch.Tensor, valid: torch.Tensor) -> None:
        if any(parameter.dtype != torch.float32 for parameter in self.parameters()):
            raise TypeError("writer parameters must use float32")
        if not isinstance(hidden, torch.Tensor) or not isinstance(valid, torch.Tensor):
            raise TypeError("hidden and valid must be tensors")
        if hidden.ndim != 3 or hidden.shape[-1] != self.reader_width or hidden.shape[1] == 0:
            raise ValueError("hidden must have shape [batch, nonempty tokens, reader_width]")
        if valid.shape != hidden.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("valid must be boolean and match hidden tokens")
        if hidden.dtype != torch.float32:
            raise TypeError("hidden must use float32")
        if hidden.device != self.attention_query.device or valid.device != hidden.device:
            raise ValueError("hidden, valid, and writer must share a device")
        valid_hidden = hidden.masked_select(valid.unsqueeze(-1))
        if valid_hidden.numel() and not bool(torch.isfinite(valid_hidden).all()):
            raise ValueError("valid hidden features must be finite")

    def encode_statement(
        self, hidden: torch.Tensor, valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a masked statement into ``(key, value, beta)``."""

        self._validate_hidden(hidden, valid)
        mask = valid.unsqueeze(-1)
        safe_hidden = torch.where(mask, hidden, torch.zeros_like(hidden))
        scores = torch.sum(safe_hidden * self.attention_query, dim=-1)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1) * valid
        denominator = weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        weights = weights / denominator
        pooled = torch.sum(weights.unsqueeze(-1) * safe_hidden, dim=1)
        projected = self.input_projection(pooled)
        if self.normalize_hidden:
            projected = F.layer_norm(projected, (self.hidden_width,), eps=1e-5)
        features = F.gelu(projected)
        key = F.normalize(self.key_projection(features), dim=-1, eps=1e-6)
        value = torch.tanh(self.value_projection(features))
        if self.fixed_beta is None:
            beta = torch.sigmoid(self.beta_projection(features)).squeeze(-1)
        else:
            beta = features.new_full((features.shape[0],), self.fixed_beta)
        return key, value, beta

    def _validate_state(self, state: LatentSlotState, batch_size: int, device: torch.device) -> None:
        if not isinstance(state, LatentSlotState):
            raise TypeError("state must be a LatentSlotState")
        if state.values.shape != (batch_size, self.slots, self.memory_width):
            raise ValueError("state shape must match writer and batch")
        if state.values.dtype != torch.float32:
            raise TypeError("state values must use float32")
        if state.values.device != device or state.valid.device != device:
            raise ValueError("state and hidden must share a device")
        if not bool(torch.equal(state.valid[:, 0], state.valid[:, 1])):
            raise ValueError("both state valid flags must agree")
        valid_values = state.values.masked_select(state.valid.unsqueeze(-1))
        if valid_values.numel() and not bool(torch.isfinite(valid_values).all()):
            raise ValueError("valid state values must be finite")

    def forward(
        self, state: LatentSlotState, hidden: torch.Tensor, valid: torch.Tensor,
    ) -> LatentSlotState:
        self._validate_hidden(hidden, valid)
        self._validate_state(state, hidden.shape[0], hidden.device)
        key, value, beta = self.encode_statement(hidden, valid)

        old_values = torch.where(state.valid.unsqueeze(-1), state.values, torch.zeros_like(state.values))
        old_matrix = old_values.reshape(hidden.shape[0], self.key_width, self.value_width)
        updated = delta_update(old_matrix, key, value, beta).reshape_as(state.values)
        has_statement = valid.any(dim=1, keepdim=True)
        values = torch.where(has_statement.unsqueeze(-1), updated, state.values)
        next_valid = state.valid | has_statement.expand_as(state.valid)
        return LatentSlotState(values, next_valid)
