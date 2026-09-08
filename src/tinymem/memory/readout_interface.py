"""One-shot compression and paired readout at a fixed 66-byte state boundary."""

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from tinymem.memory.recurrent_slots import LatentSlotState


ReadoutKind = Literal["affine", "gelu"]
STATE_BYTES = 66


def check_readout_state(state: LatentSlotState) -> None:
    """Require independently owned, bounded single-history payloads."""
    if not isinstance(state, LatentSlotState):
        raise TypeError("state must be a LatentSlotState")
    if state.values.shape != (1, 2, 8) or state.valid.shape != (1, 2):
        raise ValueError("state must contain two width-eight slots for one history")
    if state.values.dtype != torch.float32 or state.nbytes != STATE_BYTES:
        raise ValueError("state must own exactly 66 bytes with FP32 values")
    if not torch.isfinite(state.values).all() or (state.values.abs() > 1).any():
        raise ValueError("state values must be finite and bounded to [-1, 1]")


class OneShotEncoder(nn.Module):
    """Pool history features only; no recurrent state, query, or reader ownership."""

    def __init__(self, reader_width: int) -> None:
        super().__init__()
        if type(reader_width) is not int or reader_width <= 0:
            raise ValueError("reader_width must be a positive integer")
        self.reader_width = reader_width
        self.queries = nn.Parameter(torch.empty(2, reader_width))
        nn.init.normal_(self.queries, std=0.02)
        self.input_projection = nn.Linear(reader_width, 64)
        self.output_projection = nn.Linear(64, 8)

    def forward(self, hidden: torch.Tensor, valid: torch.Tensor) -> LatentSlotState:
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] == 0 or hidden.shape[2] != self.reader_width:
            raise ValueError("hidden must have shape [1, nonempty tokens, reader_width]")
        if valid.shape != hidden.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("valid must be boolean and match history tokens")
        if hidden.device != self.queries.device or valid.device != hidden.device:
            raise ValueError("features, mask, and encoder must share a device")
        if hidden.dtype != torch.float32 or any(p.dtype != torch.float32 for p in self.parameters()):
            raise TypeError("features and encoder parameters must be FP32")
        current = hidden.masked_fill(~valid.unsqueeze(-1), 0)
        if not torch.isfinite(current).all():
            raise ValueError("valid history features must be finite")
        scores = (self.queries @ current.transpose(-1, -2)) * self.reader_width**-0.5
        scores = scores.masked_fill(~valid.unsqueeze(1), torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1) * valid.unsqueeze(1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        pooled = weights @ current
        values = self.output_projection(F.gelu(self.input_projection(pooled))).tanh()
        occupied = valid.any(dim=1, keepdim=True).expand(1, 2).clone()
        values = values.masked_fill(~occupied.unsqueeze(-1), 0)
        state = LatentSlotState(values, occupied)
        check_readout_state(state)
        return state


class ReadoutBridge(nn.Module):
    """Equal parameter shapes; the only architectural difference is GELU."""

    def __init__(self, reader_width: int, kind: ReadoutKind) -> None:
        super().__init__()
        if type(reader_width) is not int or reader_width <= 0:
            raise ValueError("reader_width must be a positive integer")
        if kind not in ("affine", "gelu"):
            raise ValueError("kind must be affine or gelu")
        self.kind = kind
        self.input_projection = nn.Linear(8, 32)
        self.output_projection = nn.Linear(32, reader_width)

    def forward(self, state: LatentSlotState) -> torch.Tensor:
        check_readout_state(state)
        if state.values.device != self.input_projection.weight.device:
            raise ValueError("state and bridge must share a device")
        if any(p.dtype != torch.float32 for p in self.parameters()):
            raise TypeError("bridge parameters must be FP32")
        indices = state.valid[0].nonzero(as_tuple=True)[0].tolist()
        if not indices:
            return state.values.new_empty(0, self.output_projection.out_features)
        # Basic slices preserve the existing deterministic MPS read convention.
        values = torch.cat([state.values[0, index:index + 1] for index in indices])
        hidden = self.input_projection(values)
        if self.kind == "gelu":
            hidden = F.gelu(hidden)
        memory = self.output_projection(hidden)
        if not torch.isfinite(memory).all():
            raise ValueError("projected memory must be finite")
        return memory
