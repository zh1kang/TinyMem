"""Full-width query pooling followed by narrow, query-independent slot storage."""

import torch
from torch import nn
from torch.nn import functional as F

from tinymem.memory.recurrent_slots import LatentSlotState


class QueryPoolSlotWriter(nn.Module):
    """Keep attention wide; compress only the pooled features into recurrent state."""

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
        self.queries = nn.Parameter(torch.empty(slots, reader_width))
        nn.init.normal_(self.queries, std=0.02)
        self.state_projection = nn.Linear(memory_width, reader_width, bias=False)
        self.query_projection = nn.Linear(memory_width, reader_width, bias=False)
        self.input_projection = nn.Linear(reader_width, hidden_width)
        self.output_projection = nn.Linear(hidden_width, memory_width)
        self.gate = nn.Linear(2 * memory_width, memory_width)

    def empty(self, batch_size: int) -> LatentSlotState:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        return LatentSlotState(
            self.queries.new_zeros(batch_size, self.slots, self.memory_width),
            torch.zeros(batch_size, self.slots, device=self.queries.device, dtype=torch.bool),
        )

    def forward(self, state: LatentSlotState, hidden: torch.Tensor, valid: torch.Tensor) -> LatentSlotState:
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
        current = hidden.masked_fill(~valid.unsqueeze(-1), 0)
        source = torch.cat((self.state_projection(old), current), dim=1)
        source_valid = torch.cat((state.valid, valid), dim=1)
        queries = self.queries + self.query_projection(old)
        scores = (queries @ source.transpose(-1, -2)) * self.reader_width**-0.5
        scores = scores.masked_fill(~source_valid.unsqueeze(1), torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1) * source_valid.unsqueeze(1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        pooled = weights @ source
        compressed = self.output_projection(F.gelu(self.input_projection(pooled)))
        candidate = torch.tanh(compressed)
        gate = torch.sigmoid(self.gate(torch.cat((old, compressed), dim=-1)))
        proposed = old + gate * (candidate - old)
        has_segment = valid.any(dim=1, keepdim=True)
        values = torch.where(has_segment.unsqueeze(-1), proposed, state.values)
        return LatentSlotState(values, state.valid | has_segment)
