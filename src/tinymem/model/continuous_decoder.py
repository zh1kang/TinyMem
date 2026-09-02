"""Differentiable segment-level decoding with continuous recurrent memory."""

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral

import torch
from torch import nn

from tinymem.memory.continuous import ContinuousMemoryCompressor
from tinymem.memory.discrete_compressor import DiscreteMemoryCompressor
from tinymem.memory.recurrent_memory import (
    GatedRecurrentMemoryBank,
    RecurrentMemoryBank,
)
from tinymem.memory.write_gate import TokenSegmentWriteGate
from tinymem.model.kv_cache import KVCache
from tinymem.model.memory_input import AttentionMemory
from tinymem.model.transformer import DecoderOnlyTransformer


MemoryIntervention = Callable[[AttentionMemory], AttentionMemory]
MEMORY_POSITION_MODES = frozenset(("absolute", "virtual"))


@dataclass(frozen=True)
class SegmentedContinuousOutput:
    """Hold logits and the final explicit segmented-memory state."""

    logits: torch.Tensor
    memory: torch.Tensor
    memory_valid: torch.Tensor
    memory_positions: torch.Tensor
    writes_applied: torch.Tensor
    write_logits: torch.Tensor | None
    memory_codes: torch.Tensor | None
    proposed_code_indices: torch.Tensor | None
    code_probabilities: torch.Tensor | None
    proposed_code_valid: torch.Tensor | None


class SegmentedContinuousDecoder(nn.Module):
    """Process segments causally and carry differentiable memory between them."""

    def __init__(
        self,
        model: DecoderOnlyTransformer,
        compressor: ContinuousMemoryCompressor,
        bank: RecurrentMemoryBank,
        *,
        segment_length: int,
        write_gate: TokenSegmentWriteGate | None = None,
        memory_position_mode: str = "absolute",
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
        if write_gate is not None and not isinstance(
            write_gate,
            TokenSegmentWriteGate,
        ):
            raise TypeError("write_gate must be a TokenSegmentWriteGate or None")
        if write_gate is not None and not isinstance(
            bank,
            GatedRecurrentMemoryBank,
        ):
            raise ValueError("write_gate requires a gated recurrent memory bank")
        if (
            write_gate is not None
            and write_gate.model_width != model.config.d_model
        ):
            raise ValueError("write gate width must match the model width")
        if not isinstance(memory_position_mode, str):
            raise TypeError("memory_position_mode must be a string")
        if memory_position_mode not in MEMORY_POSITION_MODES:
            raise ValueError(
                "memory_position_mode must be 'absolute' or 'virtual'"
            )

        self.model = model
        self.compressor = compressor
        self.bank = bank
        self.segment_length = int(segment_length)
        self.write_gate = write_gate
        self.memory_position_mode = memory_position_mode

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

    def _attention_memory_positions(
        self,
        memory_positions: torch.Tensor,
        memory_valid: torch.Tensor,
        *,
        position_offset: int,
    ) -> torch.Tensor:
        if self.memory_position_mode == "absolute":
            return memory_positions
        ranks = memory_valid.cumsum(dim=1)
        valid_count = memory_valid.sum(dim=1, keepdim=True)
        virtual_positions = position_offset - valid_count + ranks - 1
        return virtual_positions.clamp_min(0).masked_fill(~memory_valid, -1)

    @staticmethod
    def _update_slot_state(
        state: torch.Tensor,
        proposed: torch.Tensor,
        write_applied: torch.Tensor,
    ) -> torch.Tensor:
        write_count = proposed.shape[1]
        shifted = torch.cat((state[:, write_count:], proposed), dim=1)
        expanded_write = write_applied.reshape(
            write_applied.shape[0],
            *((1,) * (state.ndim - 1)),
        )
        return torch.where(expanded_write, shifted, state)

    @staticmethod
    def _update_assignment_state(
        state: torch.Tensor,
        proposed: torch.Tensor,
        write_applied: torch.Tensor,
        write_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        write_count = proposed.shape[1]
        shifted = torch.cat((state[:, write_count:], proposed), dim=1)
        if write_logits is None:
            return torch.where(write_applied.unsqueeze(-1), shifted, state)
        write_probability = torch.sigmoid(write_logits)
        straight_through_write = (
            write_applied.to(dtype=write_probability.dtype)
            + write_probability
            - write_probability.detach()
        )
        return state + straight_through_write.unsqueeze(-1) * (
            shifted - state
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
        segment_writes = []
        segment_write_logits = []
        discrete_compressor = (
            self.compressor
            if isinstance(self.compressor, DiscreteMemoryCompressor)
            else None
        )
        memory_codes = (
            torch.full(
                (input_ids.shape[0], self.bank.capacity),
                -1,
                dtype=torch.long,
                device=input_ids.device,
            )
            if discrete_compressor is not None
            else None
        )
        memory_assignments = (
            torch.zeros(
                input_ids.shape[0],
                self.bank.capacity,
                discrete_compressor.codebook_size,
                dtype=memory.dtype,
                device=input_ids.device,
            )
            if discrete_compressor is not None and self.training
            else None
        )
        proposed_code_indices = []
        code_probabilities = []
        proposed_code_valid = []
        for offset in range(0, input_ids.shape[1], self.segment_length):
            end = offset + self.segment_length
            segment_ids = input_ids[:, offset:end]
            segment_valid = token_valid[:, offset:end]
            attention_memory = AttentionMemory(
                values=memory,
                valid=memory_valid,
                positions=self._attention_memory_positions(
                    memory_positions,
                    memory_valid,
                    position_offset=offset,
                ),
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

            discrete_output = None
            if discrete_compressor is None:
                summary, summary_valid = self.compressor(
                    hidden_states,
                    segment_valid,
                )
            else:
                discrete_output = discrete_compressor.compress(
                    hidden_states,
                    segment_valid,
                )
                summary = discrete_output.values
                summary_valid = discrete_output.valid
            summary_positions = self._summary_positions(
                segment_valid,
                position_offset=offset,
            ).expand(-1, summary.shape[1])
            if self.write_gate is None:
                next_memory, memory_valid, write_applied, write_logits = self.bank(
                    memory,
                    memory_valid,
                    summary,
                    summary_valid,
                )
            else:
                external_write_logits = self.write_gate(
                    self.model.token_embedding(segment_ids).detach(),
                    segment_valid,
                )
                assert isinstance(self.bank, GatedRecurrentMemoryBank)
                next_memory, memory_valid, write_applied, write_logits = self.bank(
                    memory,
                    memory_valid,
                    summary,
                    summary_valid,
                    external_write_logits=external_write_logits,
                )
            if discrete_output is None:
                memory = next_memory
            else:
                assert memory_codes is not None
                memory_codes = self._update_slot_state(
                    memory_codes,
                    discrete_output.indices,
                    write_applied,
                )
                if self.training:
                    assert memory_assignments is not None
                    memory_assignments = self._update_assignment_state(
                        memory_assignments,
                        discrete_output.assignments,
                        write_applied,
                        write_logits,
                    )
                    memory = (
                        memory_assignments
                        @ discrete_compressor.codebook.embedding.weight
                    )
                else:
                    memory = discrete_compressor.codebook.embedding(
                        memory_codes.clamp_min(0)
                    )
                    memory = memory * memory_valid.unsqueeze(-1)
                proposed_code_indices.append(discrete_output.indices)
                code_probabilities.append(discrete_output.probabilities)
                proposed_code_valid.append(discrete_output.valid)
            memory_positions = self._update_positions(
                memory_positions,
                summary_positions,
                write_applied,
            )
            segment_writes.append(write_applied)
            if write_logits is not None:
                segment_write_logits.append(write_logits)

        return SegmentedContinuousOutput(
            logits=torch.cat(segment_logits, dim=1),
            memory=memory,
            memory_valid=memory_valid,
            memory_positions=memory_positions,
            writes_applied=torch.cat(segment_writes, dim=1),
            write_logits=(
                torch.cat(segment_write_logits, dim=1)
                if segment_write_logits
                else None
            ),
            memory_codes=memory_codes,
            proposed_code_indices=(
                torch.cat(proposed_code_indices, dim=1)
                if proposed_code_indices
                else None
            ),
            code_probabilities=(
                torch.cat(code_probabilities, dim=1)
                if code_probabilities
                else None
            ),
            proposed_code_valid=(
                torch.cat(proposed_code_valid, dim=1)
                if proposed_code_valid
                else None
            ),
        )
