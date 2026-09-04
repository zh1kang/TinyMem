"""Query-independent, fixed-width latent memory with differentiable recurrence."""

from dataclasses import dataclass

import torch
from torch import nn

from tinymem.memory.storage import tensor_storage_bytes


@dataclass(frozen=True)
class LatentSlotState:
    values: torch.Tensor
    valid: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.values, torch.Tensor) or not isinstance(self.valid, torch.Tensor):
            raise TypeError("state values and valid must be tensors")
        if self.values.ndim != 3 or min(self.values.shape) <= 0:
            raise ValueError("values must have nonempty [batch, slots, memory_width] shape")
        if not self.values.is_floating_point():
            raise TypeError("values must be floating point")
        if self.valid.shape != self.values.shape[:2] or self.valid.dtype != torch.bool:
            raise ValueError("valid must be boolean with shape [batch, slots]")
        if self.valid.device != self.values.device:
            raise ValueError("state tensors must share a device")

    @property
    def nbytes(self) -> int:
        return tensor_storage_bytes((self.values, self.valid))

    def detached(self) -> "LatentSlotState":
        return LatentSlotState(self.values.detach(), self.valid)


class RecurrentSlotWriter(nn.Module):
    """Attend from old slots to old state plus new segment, then replace the bank."""

    def __init__(self, reader_width: int, memory_width: int, slots: int) -> None:
        super().__init__()
        for name, value in (("reader_width", reader_width), ("memory_width", memory_width), ("slots", slots)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.reader_width = reader_width
        self.memory_width = memory_width
        self.slots = slots
        self.input_projection = nn.Linear(reader_width, memory_width)
        self.queries = nn.Parameter(torch.empty(slots, memory_width))
        nn.init.normal_(self.queries, std=0.02)
        self.query_projection = nn.Linear(memory_width, memory_width)
        self.key_projection = nn.Linear(memory_width, memory_width)
        self.value_projection = nn.Linear(memory_width, memory_width)
        self.output_projection = nn.Linear(memory_width, memory_width)
        self.gate = nn.Linear(2 * memory_width, memory_width)

    def empty(self, batch_size: int) -> LatentSlotState:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        return LatentSlotState(
            self.queries.new_zeros(batch_size, self.slots, self.memory_width),
            torch.zeros(batch_size, self.slots, device=self.queries.device, dtype=torch.bool),
        )

    def forward(
        self, state: LatentSlotState, hidden: torch.Tensor, valid: torch.Tensor,
    ) -> LatentSlotState:
        if hidden.ndim != 3 or hidden.shape[-1] != self.reader_width or hidden.shape[1] == 0:
            raise ValueError("hidden must have shape [batch, nonempty tokens, reader_width]")
        if valid.shape != hidden.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("valid must be boolean and match hidden tokens")
        if state.values.shape != (hidden.shape[0], self.slots, self.memory_width):
            raise ValueError("state shape must match writer and batch")
        if hidden.dtype != self.queries.dtype or state.values.dtype != hidden.dtype:
            raise TypeError("hidden, state, and writer must share a dtype")
        if any(tensor.device != self.queries.device for tensor in (hidden, valid, state.values)):
            raise ValueError("hidden, state, and writer must share a device")

        old = state.values.masked_fill(~state.valid.unsqueeze(-1), 0)
        current = self.input_projection(hidden.masked_fill(~valid.unsqueeze(-1), 0))
        source = torch.cat((old, current), dim=1)
        source_valid = torch.cat((state.valid, valid), dim=1)
        queries = self.query_projection(old + self.queries)
        scores = queries @ self.key_projection(source).transpose(-1, -2)
        scores = scores * self.memory_width**-0.5
        scores = scores.masked_fill(~source_valid.unsqueeze(1), torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1) * source_valid.unsqueeze(1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        attended = weights @ self.value_projection(source)
        candidate = torch.tanh(self.output_projection(attended))
        gate = torch.sigmoid(self.gate(torch.cat((old, attended), dim=-1)))
        proposed = old + gate * (candidate - old)
        has_segment = valid.any(dim=1, keepdim=True)
        values = torch.where(has_segment.unsqueeze(-1), proposed, state.values)
        next_valid = state.valid | has_segment
        return LatentSlotState(values, next_valid)
