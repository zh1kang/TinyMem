"""Read compact retained IDs by recomputing context, without a persistent KV cache."""

import torch
from torch import nn

from tinymem.memory.compact_tokens import CompactTokenRetention, CompactTokenState
from tinymem.model.transformer import DecoderOnlyTransformer


class RawRetentionDecoder(nn.Module):
    def __init__(
        self, reader: DecoderOnlyTransformer, *, capacity: int,
        query_length: int, history_vocab_size: int = 256,
    ) -> None:
        super().__init__()
        if not isinstance(reader, DecoderOnlyTransformer):
            raise TypeError("reader must be a DecoderOnlyTransformer")
        self.retention = CompactTokenRetention(capacity, history_vocab_size)
        if isinstance(query_length, bool) or not isinstance(query_length, int) or query_length <= 0:
            raise ValueError("query_length must be a positive integer")
        if capacity + query_length > reader.config.max_local_tokens:
            raise ValueError("retained prefix plus query exceeds reader capacity")
        if history_vocab_size > reader.config.vocab_size:
            raise ValueError("history vocabulary must fit the reader vocabulary")
        self.reader = reader
        self.query_length = query_length

    def empty(self, batch_size: int) -> CompactTokenState:
        return self.retention.empty(batch_size, device=self.reader.token_embedding.weight.device)

    def write(
        self, input_ids: torch.Tensor, valid: torch.Tensor, state: CompactTokenState,
    ) -> CompactTokenState:
        if valid.shape != input_ids.shape or valid.dtype != torch.bool:
            raise ValueError("valid must be boolean and match input IDs")
        return self.retention.append(state, input_ids.masked_fill(~valid, 0), valid)

    def forward(self, query_ids: torch.Tensor, state: CompactTokenState) -> torch.Tensor:
        if query_ids.ndim != 2 or not 0 < query_ids.shape[1] <= self.query_length:
            raise ValueError("query IDs must fit [batch, query_length]")
        if query_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("query IDs must be integer")
        history_ids, _ = self.retention.materialize(state, pad_id=0)
        if query_ids.shape[0] != history_ids.shape[0] or query_ids.device != history_ids.device:
            raise ValueError("query and state must share batch and device")
        if ((query_ids < 0) | (query_ids >= self.reader.config.vocab_size)).any():
            raise ValueError("query IDs must fit the reader vocabulary")
        total_length = int(state.lengths.max()) + query_ids.shape[1]
        combined = query_ids.new_zeros(query_ids.shape[0], total_length)
        for row in range(query_ids.shape[0]):
            length = int(state.lengths[row])
            combined[row, :length] = history_ids[row, :length]
            combined[row, length:length + query_ids.shape[1]] = query_ids[row]
        logits = self.reader(combined)
        query_positions = state.lengths.unsqueeze(1) + torch.arange(query_ids.shape[1], device=query_ids.device)
        return logits.gather(1, query_positions.unsqueeze(-1).expand(-1, -1, logits.shape[-1]))
