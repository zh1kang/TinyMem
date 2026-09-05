"""Query-blind narrow recurrent state around the frozen native research reader."""

from typing import Literal

import torch
from torch import nn

from tinymem.memory.mean_pool_slots import MeanPoolSlotWriter
from tinymem.memory.query_pool_slots import QueryPoolSlotWriter
from tinymem.memory.recurrent_slots import LatentSlotState, RecurrentSlotWriter
from tinymem.research.pretrained import PretrainedReader


class NativeRecurrentMemory(nn.Module):
    """Own only shared writer/projection weights, never a reader or stream state."""

    def __init__(
        self, reader_width: int, *, memory_width: int, slots: int, segment_length: int,
        writer_kind: Literal["narrow", "query_pool", "mean_pool"] = "narrow", aggregation_width: int | None = None,
    ) -> None:
        super().__init__()
        if isinstance(segment_length, bool) or not isinstance(segment_length, int) or segment_length <= 0:
            raise ValueError("segment_length must be a positive integer")
        self.writer: RecurrentSlotWriter | QueryPoolSlotWriter | MeanPoolSlotWriter
        if writer_kind == "narrow":
            if aggregation_width is not None:
                raise ValueError("aggregation_width is only supported by query_pool and mean_pool")
            self.writer = RecurrentSlotWriter(reader_width, memory_width, slots)
        elif writer_kind == "query_pool":
            width = 64 if aggregation_width is None else aggregation_width
            self.writer = QueryPoolSlotWriter(reader_width, memory_width, slots, hidden_width=width)
        elif writer_kind == "mean_pool":
            width = 64 if aggregation_width is None else aggregation_width
            self.writer = MeanPoolSlotWriter(reader_width, memory_width, slots, hidden_width=width)
        else:
            raise ValueError("writer_kind must be narrow, query_pool, or mean_pool")
        self.writer_kind = writer_kind
        self.read_projection = nn.Linear(memory_width, reader_width, bias=False)
        self.segment_length = segment_length

    def memory_vectors(self, state: LatentSlotState) -> torch.Tensor:
        """Expand valid slots only; the result is temporary reader input."""
        if not isinstance(state, LatentSlotState):
            raise TypeError("state must be a LatentSlotState")
        if state.values.shape != (1, self.writer.slots, self.writer.memory_width):
            raise ValueError("state must match the single-stream writer shape")
        if state.values.device != self.read_projection.weight.device or state.values.dtype != self.read_projection.weight.dtype:
            raise ValueError("state and writer must share device and dtype")
        indices = state.valid[0].nonzero(as_tuple=True)[0].tolist()
        if not indices:
            return self.read_projection.weight.new_empty(0, self.writer.reader_width)
        # Basic slices avoid nondeterministic MPS advanced-index backward.
        values = torch.cat([state.values[0, index:index + 1] for index in indices])
        if not torch.isfinite(values).all():
            raise ValueError("valid memory must be finite")
        return self.read_projection(values)

    def write(
        self, reader: PretrainedReader, state: LatentSlotState, input_ids: torch.Tensor,
    ) -> LatentSlotState:
        """Read only this history chunk plus old state; preserve recurrent gradients."""
        if input_ids.ndim != 1 or not 0 < input_ids.numel() <= self.segment_length:
            raise ValueError("input_ids must be a nonempty vector within segment_length")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must be int32 or int64")
        parameter = next(self.writer.parameters())
        if input_ids.device != reader.model.device or input_ids.device != parameter.device:
            raise ValueError("input_ids, reader, and writer must share a device")
        if ((input_ids < 0) | (input_ids >= reader.model.config.vocab_size)).any():
            raise ValueError("input_ids contains an out-of-vocabulary token")
        if any(parameter.requires_grad for parameter in reader.model.parameters()):
            raise ValueError("the shared reader must be frozen during memory training")
        embedding = reader.model.get_input_embeddings()
        if embedding.embedding_dim != self.writer.reader_width:
            raise ValueError("reader embedding width must match writer")
        memory = self.memory_vectors(state).to(embedding.weight.dtype)
        if not torch.isfinite(memory).all():
            raise ValueError("projected memory must be finite in the reader dtype")
        length = memory.shape[0] + input_ids.numel()
        if length > reader.model.config.max_position_embeddings:
            raise ValueError("memory plus chunk exceeds reader context; truncation is forbidden")
        inputs = torch.cat((memory, embedding(input_ids))).unsqueeze(0)
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        output = reader.model.get_decoder()(
            inputs_embeds=inputs, attention_mask=torch.ones_like(positions),
            position_ids=positions, use_cache=False,
        )
        hidden = output.last_hidden_state[:, memory.shape[0]:].to(parameter.dtype)
        if not torch.isfinite(hidden).all():
            raise ValueError("history hidden states must be finite in the writer dtype")
        valid = torch.ones(1, input_ids.numel(), dtype=torch.bool, device=input_ids.device)
        return self.writer(state, hidden, valid)
