"""A narrow byte-research adapter with no persistent raw-token or KV path."""

import torch
from torch import nn

from tinymem.memory.recurrent_slots import LatentSlotState, RecurrentSlotWriter
from tinymem.model.memory_input import AttentionMemory
from tinymem.model.transformer import DecoderOnlyTransformer


class RecurrentSlotDecoder(nn.Module):
    """Write history without questions; read final narrow memory with the decoder."""

    def __init__(
        self, reader: DecoderOnlyTransformer, *, memory_width: int,
        slots: int, segment_length: int,
    ) -> None:
        super().__init__()
        if not isinstance(reader, DecoderOnlyTransformer):
            raise TypeError("reader must be a DecoderOnlyTransformer")
        if isinstance(segment_length, bool) or not isinstance(segment_length, int) or segment_length <= 0:
            raise ValueError("segment_length must be a positive integer")
        self.writer = RecurrentSlotWriter(reader.config.d_model, memory_width, slots)
        if slots + segment_length > reader.config.max_local_tokens:
            raise ValueError("virtual memory positions plus segment exceed reader capacity")
        self.reader = reader
        self.read_projection = nn.Linear(memory_width, reader.config.d_model, bias=False)
        self.segment_length = segment_length

    def attention_memory(self, state: LatentSlotState) -> AttentionMemory:
        if state.values.shape[1:] != (self.writer.slots, self.writer.memory_width):
            raise ValueError("state shape must match writer")
        positions = torch.arange(self.writer.slots, device=state.values.device)
        return AttentionMemory(
            values=self.read_projection(state.values),
            valid=state.valid,
            positions=positions.unsqueeze(0).expand(state.values.shape[0], -1),
        )

    def _check_length(self, input_ids: torch.Tensor) -> None:
        if input_ids.ndim != 2 or not 0 < input_ids.shape[1] <= self.segment_length:
            raise ValueError("input_ids must be a nonempty batch within segment_length")

    def write(
        self, input_ids: torch.Tensor, valid: torch.Tensor, state: LatentSlotState,
    ) -> LatentSlotState:
        self._check_length(input_ids)
        if valid.shape != input_ids.shape or valid.dtype != torch.bool:
            raise ValueError("valid must be boolean and match input_ids")
        if (valid[:, 1:] & ~valid[:, :-1]).any():
            raise ValueError("writer inputs must be right padded")
        hidden = self.reader.forward_hidden(
            input_ids, position_offset=self.writer.slots,
            memory=self.attention_memory(state),
        )
        return self.writer(state, hidden, valid)

    def forward(self, input_ids: torch.Tensor, state: LatentSlotState) -> torch.Tensor:
        self._check_length(input_ids)
        return self.reader(
            input_ids, position_offset=self.writer.slots,
            memory=self.attention_memory(state),
        )
