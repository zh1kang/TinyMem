"""Trainable masked-mean summaries retained in a fixed-size FIFO bank."""

import torch
from torch import nn
from torch.nn import functional as F

from tinymem.memory.recurrent_slots import LatentSlotState


class MeanPoolSlotWriter(nn.Module):
    """Append one compressed chunk mean, without learned selection or replacement."""

    def __init__(self, reader_width: int, memory_width: int, slots: int, *, hidden_width: int = 64) -> None:
        super().__init__()
        for name, value in (("reader_width", reader_width), ("memory_width", memory_width),
                            ("slots", slots), ("hidden_width", hidden_width)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.reader_width = reader_width
        self.memory_width = memory_width
        self.slots = slots
        self.hidden_width = hidden_width
        self.input_projection = nn.Linear(reader_width, hidden_width)
        self.output_projection = nn.Linear(hidden_width, memory_width)

    def empty(self, batch_size: int) -> LatentSlotState:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        parameter = self.input_projection.weight
        return LatentSlotState(
            parameter.new_zeros(batch_size, self.slots, self.memory_width),
            torch.zeros(batch_size, self.slots, device=parameter.device, dtype=torch.bool),
        )

    def forward(self, state: LatentSlotState, hidden: torch.Tensor, valid: torch.Tensor) -> LatentSlotState:
        if hidden.ndim != 3 or hidden.shape[-1] != self.reader_width or hidden.shape[1] == 0:
            raise ValueError("hidden must have shape [batch, nonempty tokens, reader_width]")
        if valid.shape != hidden.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("valid must be boolean and match hidden tokens")
        if state.values.shape != (hidden.shape[0], self.slots, self.memory_width):
            raise ValueError("state shape must match writer and batch")
        parameter = self.input_projection.weight
        if hidden.dtype != parameter.dtype or state.values.dtype != hidden.dtype:
            raise TypeError("hidden, state, and writer must share a dtype")
        if any(tensor.device != parameter.device for tensor in (hidden, valid, state.values)):
            raise ValueError("hidden, state, and writer must share a device")

        masked = hidden.masked_fill(~valid.unsqueeze(-1), 0)
        count = valid.sum(dim=1, keepdim=True).unsqueeze(-1).clamp_min(1)
        pooled = masked.sum(dim=1, keepdim=True) / count
        summary = self.output_projection(F.gelu(self.input_projection(pooled))).tanh()
        old = state.values.masked_fill(~state.valid.unsqueeze(-1), 0)
        proposed = torch.cat((old[:, 1:], summary), dim=1)
        has_segment = valid.any(dim=1, keepdim=True)
        next_valid = torch.cat((state.valid[:, 1:], has_segment), dim=1)
        return LatentSlotState(
            torch.where(has_segment.unsqueeze(-1), proposed, state.values),
            torch.where(has_segment, next_valid, state.valid),
        )
