"""Differentiable segment-level decoding with continuous recurrent memory."""

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral

import torch
from torch import nn

from tinymem.memory.continuous import ContinuousMemoryCompressor
from tinymem.memory.recurrent_memory import RecurrentMemoryBank
from tinymem.model.kv_cache import KVCache
from tinymem.model.memory_input import AttentionMemory
from tinymem.model.transformer import DecoderOnlyTransformer


MemoryIntervention = Callable[[AttentionMemory], AttentionMemory]


@dataclass(frozen=True)
class SegmentedContinuousOutput:
    """Hold logits and the final explicit continuous-memory state."""

    logits: torch.Tensor
    memory: torch.Tensor
    memory_valid: torch.Tensor
    memory_positions: torch.Tensor


class SegmentedContinuousDecoder(nn.Module):
    """Process segments causally and carry differentiable memory between them."""

    def __init__(
        self,
        model: DecoderOnlyTransformer,
        compressor: ContinuousMemoryCompressor,
        bank: RecurrentMemoryBank,
        *,
        segment_length: int,
    ) -> None:
        super().__init__()
        if not isinstance(model, DecoderOnlyTransformer):
            raise TypeError("model must be a DecoderOnlyTransformer")
        if not isinstance(compressor, ContinuousMemoryCompressor):
            raise TypeError("compressor must be a ContinuousMemoryCompressor")
        if not isinstance(bank, RecurrentMemoryBank):
            raise TypeError("bank must be a RecurrentMemoryBank")
        if isinstance(segment_length, bool) or not isinstance(
            segment_length,
            Integral,
        ):
            raise TypeError("segment_length must be an integer")
        if segment_length <= 0:
            raise ValueError("segment_length must be positive")
        if segment_length > model.config.max_local_tokens:
            raise ValueError("segment_length must not exceed max_local_tokens")
        if compressor.model_width != model.config.d_model:
            raise ValueError("compressor width must match the model width")
        if bank.model_width != model.config.d_model:
            raise ValueError("bank width must match the model width")

        self.model = model
        self.compressor = compressor
        self.bank = bank
        self.segment_length = int(segment_length)

    def _empty_memory(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        memory = torch.zeros(
            batch_size,
            self.bank.capacity,
            self.model.config.d_model,
            device=device,
            dtype=dtype,
        )
        memory_valid = torch.zeros(
            batch_size,
            self.bank.capacity,
            device=device,
            dtype=torch.bool,
        )
        memory_positions = torch.full(
            (batch_size, self.bank.capacity),
            -1,
            device=device,
            dtype=torch.long,
        )
        return memory, memory_valid, memory_positions

    @staticmethod
    def _summary_positions(
        segment_valid: torch.Tensor,
        *,
        position_offset: int,
    ) -> torch.Tensor:
        local_positions = torch.arange(
            position_offset,
            position_offset + segment_valid.shape[1],
            device=segment_valid.device,
        )
        positions = local_positions.unsqueeze(0).expand(
            segment_valid.shape[0],
            -1,
        )
        return positions.masked_fill(~segment_valid, -1).amax(
            dim=1,
            keepdim=True,
        )

    @staticmethod
    def _update_positions(
        memory_positions: torch.Tensor,
        summary_positions: torch.Tensor,
        summary_valid: torch.Tensor,
    ) -> torch.Tensor:
        write_count = summary_positions.shape[1]
        shifted_positions = torch.cat(
            (memory_positions[:, write_count:], summary_positions),
            dim=1,
        )
        return torch.where(
            summary_valid[:, :1],
            shifted_positions,
            memory_positions,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_valid: torch.Tensor,
        *,
        memory_intervention: MemoryIntervention | None = None,
    ) -> SegmentedContinuousOutput:
        """Return logits and final memory without breaking the autograd graph."""
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("input_ids must be a torch.Tensor")
        if not isinstance(token_valid, torch.Tensor):
            raise TypeError("token_valid must be a torch.Tensor")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, tokens]")
        if input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must contain at least one token")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must be an integer tensor")
        if token_valid.shape != input_ids.shape:
            raise ValueError(f"token_valid must have shape {input_ids.shape}")
        if token_valid.dtype != torch.bool:
            raise TypeError("token_valid must be a boolean tensor")
        if token_valid.device != input_ids.device:
            raise ValueError("token_valid and input_ids must share a device")
        if input_ids.device != self.model.token_embedding.weight.device:
            raise ValueError("input_ids and decoder must share a device")
        if memory_intervention is not None and not callable(memory_intervention):
            raise TypeError("memory_intervention must be callable or None")
        if (
            token_valid.shape[1] > 1
            and (token_valid[:, 1:] & ~token_valid[:, :-1]).any()
        ):
            raise ValueError("token_valid must describe right-padded rows")

        memory, memory_valid, memory_positions = self._empty_memory(
            batch_size=input_ids.shape[0],
            device=input_ids.device,
            dtype=self.model.token_embedding.weight.dtype,
        )
        segment_logits = []
        for offset in range(0, input_ids.shape[1], self.segment_length):
            end = offset + self.segment_length
            segment_ids = input_ids[:, offset:end]
            segment_valid = token_valid[:, offset:end]
            attention_memory = AttentionMemory(
                values=memory,
                valid=memory_valid,
                positions=memory_positions,
            )
            if memory_intervention is not None:
                attention_memory = memory_intervention(attention_memory)
                if not isinstance(attention_memory, AttentionMemory):
                    raise TypeError(
                        "memory_intervention must return AttentionMemory"
                    )
            caches = [
                KVCache(
                    max_length=self.segment_length,
                    start_position=offset,
                )
                for _ in range(self.model.config.n_layers)
            ]
            hidden_states = self.model.forward_hidden(
                segment_ids,
                position_offset=offset,
                caches=caches,
                memory=attention_memory,
            )
            segment_logits.append(self.model.lm_head(hidden_states))

            summary, summary_valid = self.compressor(
                hidden_states,
                segment_valid,
            )
            summary_positions = self._summary_positions(
                segment_valid,
                position_offset=offset,
            ).expand(-1, summary.shape[1])
            memory, memory_valid = self.bank(
                memory,
                memory_valid,
                summary,
                summary_valid,
            )
            memory_positions = self._update_positions(
                memory_positions,
                summary_positions,
                summary_valid,
            )

        return SegmentedContinuousOutput(
            logits=torch.cat(segment_logits, dim=1),
            memory=memory,
            memory_valid=memory_valid,
            memory_positions=memory_positions,
        )
