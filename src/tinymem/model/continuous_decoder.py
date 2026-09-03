"""Differentiable segment-level decoding with continuous recurrent memory."""

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral

import torch
from torch import nn

from tinymem.memory.controller import AdaptiveWriteController
from tinymem.memory.continuous import ContinuousMemoryCompressor
from tinymem.memory.discrete_compressor import DiscreteMemoryCompressor
from tinymem.memory.recurrent_memory import (
    GatedRecurrentMemoryBank,
    RecurrentMemoryBank,
)
from tinymem.memory.write_gate import TokenSegmentWriteGate
from tinymem.model.memory_input import AttentionMemory
from tinymem.model.multi_token_prediction import MultiTokenPredictionHeads
from tinymem.model.transformer import DecoderOnlyTransformer


MemoryIntervention = Callable[[AttentionMemory], AttentionMemory]
MEMORY_POSITION_MODES = frozenset(("absolute", "virtual"))


@dataclass(frozen=True)
class SegmentedContinuousOutput:
    """Hold logits and the final explicit segmented-memory state."""

    logits: torch.Tensor
    mtp_logits: dict[int, torch.Tensor] | None
    memory: torch.Tensor
    memory_valid: torch.Tensor
    memory_positions: torch.Tensor
    writes_applied: torch.Tensor
    write_logits: torch.Tensor | None
    memory_codes: torch.Tensor | None
    proposed_code_indices: torch.Tensor | None
    code_probabilities: torch.Tensor | None
    code_assignments: torch.Tensor | None
    proposed_code_valid: torch.Tensor | None
    prequantized_codes: torch.Tensor | None
    controller_action_logits: torch.Tensor | None
    controller_probabilities: torch.Tensor | None
    controller_assignments: torch.Tensor | None
    controller_surprise: torch.Tensor | None
    controller_valid: torch.Tensor | None


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
        write_controller: AdaptiveWriteController | None = None,
        mtp_heads: MultiTokenPredictionHeads | None = None,
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
        if write_controller is not None and not isinstance(
            write_controller,
            AdaptiveWriteController,
        ):
            raise TypeError(
                "write_controller must be an AdaptiveWriteController or None"
            )
        if write_gate is not None and write_controller is not None:
            raise ValueError("write_gate and write_controller are mutually exclusive")
        if mtp_heads is not None and not isinstance(
            mtp_heads,
            MultiTokenPredictionHeads,
        ):
            raise TypeError("mtp_heads must be MultiTokenPredictionHeads or None")
        if mtp_heads is not None and (
            mtp_heads.model_width != model.config.d_model
            or mtp_heads.vocab_size != model.config.vocab_size
        ):
            raise ValueError("MTP heads must match the model width and vocabulary")
        if write_gate is not None and not isinstance(
            bank,
            GatedRecurrentMemoryBank,
        ):
            raise ValueError("write_gate requires a gated recurrent memory bank")
        if write_controller is not None and not isinstance(
            bank,
            GatedRecurrentMemoryBank,
        ):
            raise ValueError(
                "write_controller requires a gated recurrent memory bank"
            )
        if (
            write_gate is not None
            and write_gate.model_width != model.config.d_model
        ):
            raise ValueError("write gate width must match the model width")
        if (
            write_controller is not None
            and write_controller.model_width != model.config.d_model
        ):
            raise ValueError("write controller width must match the model width")
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
        self.write_controller = write_controller
        self.mtp_heads = mtp_heads
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
    def _masked_mean(
        values: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        weights = valid.unsqueeze(-1).to(dtype=values.dtype)
        counts = weights.sum(dim=1).clamp_min(1)
        return (values * weights).sum(dim=1) / counts

    @staticmethod
    def _prediction_surprise(
        logits: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=-1)
        log_probabilities = torch.log_softmax(logits, dim=-1)
        entropy = -(probabilities * log_probabilities).sum(dim=-1)
        weights = valid.to(dtype=entropy.dtype)
        counts = weights.sum(dim=1, keepdim=True).clamp_min(1)
        return (entropy * weights).sum(dim=1, keepdim=True) / counts

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
        write_strength: torch.Tensor | None = None,
    ) -> torch.Tensor:
        write_count = proposed.shape[1]
        shifted = torch.cat((state[:, write_count:], proposed), dim=1)
        if write_strength is not None:
            return state + write_strength.unsqueeze(-1) * (shifted - state)
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
        forced_writes: torch.Tensor | None = None,
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
        if forced_writes is not None:
            if not isinstance(forced_writes, torch.Tensor):
                raise TypeError("forced_writes must be a torch.Tensor or None")
            segment_count = (
                input_ids.shape[1] + self.segment_length - 1
            ) // self.segment_length
            expected_write_shape = (input_ids.shape[0], segment_count)
            if forced_writes.shape != expected_write_shape:
                raise ValueError(
                    f"forced_writes must have shape {expected_write_shape}"
                )
            if forced_writes.dtype != torch.bool:
                raise TypeError("forced_writes must be a boolean tensor")
            if forced_writes.device != input_ids.device:
                raise ValueError("forced_writes and input_ids must share a device")
            if self.training:
                raise ValueError("forced_writes are available only during evaluation")
            if self.write_gate is None and self.write_controller is None:
                raise ValueError("forced_writes require an external write decider")
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
        segment_mtp_logits = (
            {horizon: [] for horizon in self.mtp_heads.horizons}
            if self.mtp_heads is not None
            else None
        )
        segment_writes = []
        segment_write_logits = []
        controller_action_logits = []
        controller_probabilities = []
        controller_assignments = []
        controller_surprises = []
        controller_valid = []
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
            if (
                discrete_compressor is not None
                and (
                    self.training
                    or discrete_compressor.codebook.evaluation_mode == "soft"
                )
            )
            else None
        )
        proposed_code_indices = []
        code_probabilities = []
        code_assignments = []
        proposed_code_valid = []
        prequantized_codes = []
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
            caches = self.model.create_caches(
                max_length=self.segment_length,
                start_position=offset,
            )
            hidden_states = self.model.forward_hidden(
                segment_ids,
                position_offset=offset,
                caches=caches,
                memory=attention_memory,
            )
            current_logits = self.model.lm_head(hidden_states)
            segment_logits.append(current_logits)
            if self.mtp_heads is not None:
                assert segment_mtp_logits is not None
                for horizon, logits in self.mtp_heads(hidden_states).items():
                    segment_mtp_logits[horizon].append(logits)

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
            controller_output = None
            external_write_strength = None
            if self.write_controller is not None:
                controller_surprise = self._prediction_surprise(
                    current_logits,
                    segment_valid,
                )
                controller_output = self.write_controller(
                    self._masked_mean(hidden_states, segment_valid),
                    self._masked_mean(memory, memory_valid),
                    controller_surprise,
                    segment_valid.any(dim=1),
                )
                external_write_logits = (
                    controller_output.logits[:, 1:2]
                    - controller_output.logits[:, 0:1]
                ) / self.write_controller.temperature
                external_write_strength = controller_output.write_strength
            elif self.write_gate is not None:
                external_write_logits = self.write_gate(
                    self.model.token_embedding(segment_ids).detach(),
                    segment_valid,
                )
            if forced_writes is not None:
                segment_index = offset // self.segment_length
                external_write_strength = forced_writes[
                    :,
                    segment_index : segment_index + 1,
                ].to(dtype=memory.dtype)

            if self.write_gate is None and self.write_controller is None:
                next_memory, memory_valid, write_applied, write_logits = self.bank(
                    memory,
                    memory_valid,
                    summary,
                    summary_valid,
                )
            else:
                assert isinstance(self.bank, GatedRecurrentMemoryBank)
                next_memory, memory_valid, write_applied, write_logits = self.bank(
                    memory,
                    memory_valid,
                    summary,
                    summary_valid,
                    external_write_logits=external_write_logits,
                    external_write_strength=external_write_strength,
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
                if memory_assignments is not None:
                    memory_assignments = self._update_assignment_state(
                        memory_assignments,
                        discrete_output.assignments,
                        write_applied,
                        write_logits,
                        external_write_strength,
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
                code_assignments.append(discrete_output.assignments)
                proposed_code_valid.append(discrete_output.valid)
                prequantized_codes.append(discrete_output.prequantized)
            memory_positions = self._update_positions(
                memory_positions,
                summary_positions,
                write_applied,
            )
            segment_writes.append(write_applied)
            if write_logits is not None:
                segment_write_logits.append(write_logits)
            if controller_output is not None:
                controller_action_logits.append(controller_output.logits)
                controller_probabilities.append(controller_output.probabilities)
                controller_assignments.append(controller_output.assignments)
                controller_surprises.append(controller_surprise)
                controller_valid.append(segment_valid.any(dim=1))

        return SegmentedContinuousOutput(
            logits=torch.cat(segment_logits, dim=1),
            mtp_logits=(
                {
                    horizon: torch.cat(logits, dim=1)
                    for horizon, logits in segment_mtp_logits.items()
                }
                if segment_mtp_logits is not None
                else None
            ),
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
            code_assignments=(
                torch.cat(code_assignments, dim=1)
                if code_assignments
                else None
            ),
            proposed_code_valid=(
                torch.cat(proposed_code_valid, dim=1)
                if proposed_code_valid
                else None
            ),
            prequantized_codes=(
                torch.cat(prequantized_codes, dim=1)
                if prequantized_codes
                else None
            ),
            controller_action_logits=(
                torch.stack(controller_action_logits, dim=1)
                if controller_action_logits
                else None
            ),
            controller_probabilities=(
                torch.stack(controller_probabilities, dim=1)
                if controller_probabilities
                else None
            ),
            controller_assignments=(
                torch.stack(controller_assignments, dim=1)
                if controller_assignments
                else None
            ),
            controller_surprise=(
                torch.cat(controller_surprises, dim=1)
                if controller_surprises
                else None
            ),
            controller_valid=(
                torch.stack(controller_valid, dim=1)
                if controller_valid
                else None
            ),
        )
