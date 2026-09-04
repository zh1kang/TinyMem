"""Content-aware replacement of one slot in a continuous memory bank."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import torch
from torch import nn
from torch.nn import functional as F

from tinymem.model.memory_input import AttentionMemory


@dataclass(frozen=True)
class SlotReplacementOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    assignments: torch.Tensor
    slots: torch.Tensor


class SlotReplacementController(nn.Module):
    """Select the memory slot whose entity matches a correction summary."""

    def __init__(
        self,
        model_width: int,
        *,
        hidden_width: int | None = None,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if isinstance(model_width, bool) or not isinstance(model_width, Integral):
            raise TypeError("model_width must be an integer")
        if model_width <= 0:
            raise ValueError("model_width must be positive")
        if hidden_width is None:
            hidden_width = int(model_width)
        if isinstance(hidden_width, bool) or not isinstance(hidden_width, Integral):
            raise TypeError("hidden_width must be an integer or None")
        if hidden_width <= 0:
            raise ValueError("hidden_width must be positive")
        self.model_width = int(model_width)
        self.hidden_width = int(hidden_width)
        self.pair_projection = nn.Linear(4 * self.model_width, self.hidden_width)
        self.score_projection = nn.Linear(self.hidden_width, 1)
        self.register_buffer(
            "_temperature",
            torch.tensor(self._validate_temperature(temperature)),
        )

    @staticmethod
    def _validate_temperature(temperature: float) -> float:
        if isinstance(temperature, bool) or not isinstance(temperature, Real):
            raise TypeError("temperature must be a real number")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        return float(temperature)

    @property
    def temperature(self) -> float:
        return float(self._temperature.item())

    def set_temperature(self, temperature: float) -> None:
        self._temperature.fill_(self._validate_temperature(temperature))

    def forward(
        self,
        correction: torch.Tensor,
        memory: torch.Tensor,
        memory_valid: torch.Tensor,
    ) -> SlotReplacementOutput:
        """Return a hard slot choice with a soft straight-through gradient."""
        for name, tensor in (
            ("correction", correction),
            ("memory", memory),
            ("memory_valid", memory_valid),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
        if correction.ndim != 2 or correction.shape[1] != self.model_width:
            raise ValueError(
                f"correction must have shape [batch, {self.model_width}]"
            )
        expected_memory_shape = (
            correction.shape[0],
            memory.shape[1],
            self.model_width,
        )
        if memory.ndim != 3 or memory.shape != expected_memory_shape:
            raise ValueError(
                "memory must have shape [batch, slots, model_width]"
            )
        if memory.shape[1] < 2:
            raise ValueError("memory must contain at least two slots")
        if memory_valid.shape != memory.shape[:2]:
            raise ValueError(f"memory_valid must have shape {memory.shape[:2]}")
        if not correction.is_floating_point() or not memory.is_floating_point():
            raise TypeError("correction and memory must be floating point")
        if memory_valid.dtype != torch.bool:
            raise TypeError("memory_valid must be boolean")
        if correction.shape[0] == 0:
            raise ValueError("controller inputs must contain a batch row")
        if not memory_valid.any(dim=1).all():
            raise ValueError("each row must contain at least one valid memory slot")
        if any(
            tensor.device != correction.device
            for tensor in (memory, memory_valid, self.pair_projection.weight)
        ):
            raise ValueError("controller inputs and parameters must share a device")
        if memory.dtype != correction.dtype:
            raise TypeError("correction and memory must share a dtype")
        if correction.dtype != self.pair_projection.weight.dtype:
            raise TypeError("controller inputs and parameters must share a dtype")

        expanded = correction.unsqueeze(1).expand(-1, memory.shape[1], -1)
        features = torch.cat(
            (expanded, memory, expanded * memory, (expanded - memory).abs()),
            dim=-1,
        )
        logits = self.score_projection(F.silu(self.pair_projection(features))).squeeze(-1)
        logits = logits.masked_fill(~memory_valid, torch.finfo(logits.dtype).min)
        probabilities = torch.softmax(logits / self.temperature, dim=-1)
        slots = probabilities.argmax(dim=-1)
        hard = F.one_hot(slots, num_classes=memory.shape[1]).to(
            dtype=probabilities.dtype
        )
        assignments = (
            hard - probabilities.detach() + probabilities
            if self.training
            else hard
        )
        return SlotReplacementOutput(
            logits=logits,
            probabilities=probabilities,
            assignments=assignments,
            slots=slots,
        )


def replace_memory_slot(
    memory: AttentionMemory,
    replacement: torch.Tensor,
    replacement_valid: torch.Tensor,
    replacement_positions: torch.Tensor,
    assignments: torch.Tensor,
) -> AttentionMemory:
    """Functionally replace one selected slot in each memory row."""
    if not isinstance(memory, AttentionMemory):
        raise TypeError("memory must be an AttentionMemory")
    for name, tensor in (
        ("replacement", replacement),
        ("replacement_valid", replacement_valid),
        ("replacement_positions", replacement_positions),
        ("assignments", assignments),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
    batch_size, slot_count, model_width = memory.values.shape
    if replacement.shape != (batch_size, model_width):
        raise ValueError(
            f"replacement must have shape {(batch_size, model_width)}"
        )
    if replacement_valid.shape != (batch_size,):
        raise ValueError(f"replacement_valid must have shape {(batch_size,)}")
    if replacement_positions.shape != (batch_size,):
        raise ValueError(
            f"replacement_positions must have shape {(batch_size,)}"
        )
    if assignments.shape != (batch_size, slot_count):
        raise ValueError(
            f"assignments must have shape {(batch_size, slot_count)}"
        )
    if replacement_valid.dtype != torch.bool:
        raise TypeError("replacement_valid must be boolean")
    if replacement_positions.dtype not in (torch.int32, torch.int64):
        raise TypeError("replacement_positions must be integer")
    if not replacement.is_floating_point() or not assignments.is_floating_point():
        raise TypeError("replacement and assignments must be floating point")
    tensors = (
        replacement,
        replacement_valid,
        replacement_positions,
        assignments,
    )
    if any(tensor.device != memory.values.device for tensor in tensors):
        raise ValueError("replacement inputs and memory must share a device")
    if replacement.dtype != memory.values.dtype or assignments.dtype != memory.values.dtype:
        raise TypeError("replacement values and assignments must match memory dtype")
    if not torch.isfinite(assignments).all() or (assignments < 0).any():
        raise ValueError("assignments must be finite and nonnegative")
    if not torch.allclose(
        assignments.sum(dim=1),
        torch.ones(batch_size, device=assignments.device, dtype=assignments.dtype),
    ):
        raise ValueError("assignments must sum to one in each row")
    if (replacement_positions[replacement_valid] < 0).any():
        raise ValueError("valid replacement positions must be nonnegative")

    active = assignments * replacement_valid.unsqueeze(-1).to(assignments.dtype)
    next_values = memory.values + active.unsqueeze(-1) * (
        replacement.unsqueeze(1) - memory.values
    )
    selected = assignments.argmax(dim=-1)
    hard_selected = F.one_hot(selected, num_classes=slot_count).to(torch.bool)
    hard_selected = hard_selected & replacement_valid.unsqueeze(-1)
    next_valid = torch.where(hard_selected, replacement_valid.unsqueeze(-1), memory.valid)
    next_positions = torch.where(
        hard_selected,
        replacement_positions.unsqueeze(-1),
        memory.positions,
    )
    return AttentionMemory(
        values=next_values,
        valid=next_valid,
        positions=next_positions,
    )
