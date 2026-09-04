"""Training utilities for content-aware correction replacement."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from numbers import Integral, Real

import torch
from torch.nn import functional as F
from torch.optim import Optimizer

from tinymem.data.replacement_qa import ReplacementQAExample
from tinymem.memory.replacement import (
    SlotReplacementController,
    SlotReplacementOutput,
    replace_memory_slot,
)
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.memory_input import AttentionMemory
from tinymem.tokenization.byte_tokenizer import ByteTokenizer
from tinymem.training.losses import next_token_cross_entropy


StepCallback = Callable[[int, float], None]


@dataclass(frozen=True)
class ReplacementMemoryBuild:
    initial_memory: AttentionMemory
    corrected_memory: AttentionMemory
    correction_summary: torch.Tensor
    correction_valid: torch.Tensor
    correction_positions: torch.Tensor
    controller_output: SlotReplacementOutput


@dataclass(frozen=True)
class ReplacementQATrainingHistory:
    losses: tuple[float, ...]
    answer_losses: tuple[float, ...]
    replacement_losses: tuple[float, ...]
    replacement_accuracies: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_examples(
    examples: Sequence[ReplacementQAExample],
    decoder: SegmentedContinuousDecoder,
) -> None:
    if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)):
        raise TypeError("examples must be a sequence")
    if not examples or not all(
        isinstance(example, ReplacementQAExample) for example in examples
    ):
        raise ValueError("examples must contain ReplacementQAExample values")
    capacities = {example.memory_capacity for example in examples}
    if capacities != {decoder.bank.capacity}:
        raise ValueError("example capacity must match the decoder memory bank")
    lengths = {example.segment_length for example in examples}
    if lengths != {decoder.segment_length}:
        raise ValueError("example and decoder segment lengths must match")


def _collate_sequences(
    sequences: Sequence[tuple[int, ...]],
    *,
    pad_id: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_length = max(map(len, sequences))
    input_ids = torch.full(
        (len(sequences), max_length),
        int(pad_id),
        dtype=torch.long,
        device=device,
    )
    valid = torch.zeros_like(input_ids, dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        input_ids[row, : len(sequence)] = torch.tensor(
            sequence,
            dtype=torch.long,
            device=device,
        )
        valid[row, : len(sequence)] = True
    return input_ids, valid


def collate_replacement_query(
    examples: Sequence[ReplacementQAExample],
    *,
    pad_id: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return query sequences with loss only on answer bytes and newline."""
    if not examples or not all(
        isinstance(example, ReplacementQAExample) for example in examples
    ):
        raise ValueError("examples must contain ReplacementQAExample values")
    sequences = tuple(
        example.query_ids + example.answer_ids + (ord("\n"),)
        for example in examples
    )
    input_ids, valid = _collate_sequences(
        sequences,
        pad_id=pad_id,
        device=device,
    )
    targets = torch.full_like(input_ids, -100)
    for row, example in enumerate(examples):
        answer_start = len(example.query_ids)
        answer_end = answer_start + len(example.answer_ids) + 1
        targets[row, answer_start:answer_end] = input_ids[
            row,
            answer_start:answer_end,
        ]
    return input_ids, targets, valid


def _summarize_sequences(
    decoder: SegmentedContinuousDecoder,
    sequences: Sequence[tuple[int, ...]],
    *,
    position_offset: int,
    pad_id: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids, valid = _collate_sequences(
        sequences,
        pad_id=pad_id,
        device=device,
    )
    hidden = decoder.model.forward_hidden(
        input_ids,
        position_offset=position_offset,
        caches=decoder.model.create_caches(
            max_length=decoder.segment_length,
            start_position=position_offset,
        ),
    )
    summary, summary_valid = decoder.compressor(hidden, valid)
    if summary.shape[1] != 1 or summary_valid.shape != (len(sequences), 1):
        raise ValueError("replacement QA requires one summary per segment")
    positions = (
        valid.sum(dim=1, dtype=torch.long) + int(position_offset) - 1
    )
    return summary[:, 0], summary_valid[:, 0], positions


def build_replacement_memory(
    decoder: SegmentedContinuousDecoder,
    controller: SlotReplacementController,
    examples: Sequence[ReplacementQAExample],
    *,
    pad_id: int,
    device: torch.device | str,
    forced_slots: torch.Tensor | None = None,
) -> ReplacementMemoryBuild:
    """Encode a full bank and replace the slot selected for one correction."""
    if not isinstance(decoder, SegmentedContinuousDecoder):
        raise TypeError("decoder must be a SegmentedContinuousDecoder")
    if not isinstance(controller, SlotReplacementController):
        raise TypeError("controller must be a SlotReplacementController")
    _validate_examples(examples, decoder)
    if controller.model_width != decoder.model.config.d_model:
        raise ValueError("controller and decoder widths must match")
    if isinstance(pad_id, bool) or not isinstance(pad_id, Integral):
        raise TypeError("pad_id must be an integer")
    if not 0 <= pad_id < ByteTokenizer.vocab_size:
        raise ValueError("pad_id must be in the byte-token vocabulary")

    slot_values = []
    slot_valid = []
    slot_positions = []
    for slot in range(decoder.bank.capacity):
        values, valid, positions = _summarize_sequences(
            decoder,
            tuple(example.initial_fact_ids[slot] for example in examples),
            position_offset=slot * decoder.segment_length,
            pad_id=int(pad_id),
            device=device,
        )
        slot_values.append(values)
        slot_valid.append(valid)
        slot_positions.append(positions)
    initial_memory = AttentionMemory(
        values=torch.stack(slot_values, dim=1),
        valid=torch.stack(slot_valid, dim=1),
        positions=torch.stack(slot_positions, dim=1),
    )
    correction, correction_valid, correction_positions = _summarize_sequences(
        decoder,
        tuple(example.correction_ids for example in examples),
        position_offset=decoder.bank.capacity * decoder.segment_length,
        pad_id=int(pad_id),
        device=device,
    )
    controller_output = controller(
        correction,
        initial_memory.values,
        initial_memory.valid,
    )
    assignments = controller_output.assignments
    if forced_slots is not None:
        if not isinstance(forced_slots, torch.Tensor):
            raise TypeError("forced_slots must be a torch.Tensor or None")
        if forced_slots.shape != (len(examples),):
            raise ValueError(f"forced_slots must have shape {(len(examples),)}")
        if forced_slots.dtype not in (torch.int32, torch.int64):
            raise TypeError("forced_slots must be integer")
        if forced_slots.device != initial_memory.values.device:
            raise ValueError("forced_slots and memory must share a device")
        if ((forced_slots < 0) | (forced_slots >= decoder.bank.capacity)).any():
            raise ValueError("forced_slots contains an out-of-range slot")
        assignments = F.one_hot(
            forced_slots.to(torch.long),
            num_classes=decoder.bank.capacity,
        ).to(dtype=initial_memory.values.dtype)
    corrected_memory = replace_memory_slot(
        initial_memory,
        correction,
        correction_valid,
        correction_positions,
        assignments,
    )
    return ReplacementMemoryBuild(
        initial_memory=initial_memory,
        corrected_memory=corrected_memory,
        correction_summary=correction,
        correction_valid=correction_valid,
        correction_positions=correction_positions,
        controller_output=controller_output,
    )


def train_replacement_qa(
    decoder: SegmentedContinuousDecoder,
    controller: SlotReplacementController,
    optimizer: Optimizer,
    examples: Sequence[ReplacementQAExample],
    *,
    steps: int,
    slot_pretrain_steps: int,
    batch_size: int,
    replacement_loss_weight: float,
    gradient_clip_norm: float,
    pad_id: int,
    device: torch.device | str,
    seed: int,
    on_step: StepCallback | None = None,
) -> ReplacementQATrainingHistory:
    """Train slot matching first, then the full delayed-answer path."""
    if not isinstance(decoder, SegmentedContinuousDecoder):
        raise TypeError("decoder must be a SegmentedContinuousDecoder")
    if not isinstance(controller, SlotReplacementController):
        raise TypeError("controller must be a SlotReplacementController")
    if not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be a torch Optimizer")
    _validate_examples(examples, decoder)
    for name, value in (
        ("steps", steps),
        ("slot_pretrain_steps", slot_pretrain_steps),
        ("batch_size", batch_size),
        ("seed", seed),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    if steps <= 0 or batch_size <= 0 or seed < 0:
        raise ValueError("training counts must be positive and seed nonnegative")
    if not 0 <= slot_pretrain_steps < steps:
        raise ValueError("slot_pretrain_steps must be in [0, steps)")
    for name, value in (
        ("replacement_loss_weight", replacement_loss_weight),
        ("gradient_clip_norm", gradient_clip_norm),
    ):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number")
    if replacement_loss_weight < 0 or gradient_clip_norm <= 0:
        raise ValueError("loss weight must be nonnegative and clip norm positive")
    if on_step is not None and not callable(on_step):
        raise TypeError("on_step must be callable or None")

    generator = torch.Generator().manual_seed(int(seed))
    losses = []
    answer_losses = []
    replacement_losses = []
    replacement_accuracies = []
    decoder.train()
    controller.train()
    for step in range(1, int(steps) + 1):
        indices = torch.randint(
            len(examples),
            (int(batch_size),),
            generator=generator,
        ).tolist()
        batch = [examples[index] for index in indices]
        optimizer.zero_grad(set_to_none=True)
        built = build_replacement_memory(
            decoder,
            controller,
            batch,
            pad_id=int(pad_id),
            device=device,
        )
        input_ids, target_ids, valid = collate_replacement_query(
            batch,
            pad_id=int(pad_id),
            device=device,
        )
        output = decoder(
            input_ids,
            valid,
            initial_memory=built.corrected_memory,
            position_offset=batch[0].query_position_offset,
            update_memory=False,
        )
        answer_loss = next_token_cross_entropy(output.logits, target_ids)
        targets = torch.tensor(
            [example.correction_slot for example in batch],
            dtype=torch.long,
            device=device,
        )
        replacement_loss = F.cross_entropy(
            built.controller_output.logits / controller.temperature,
            targets,
        )
        loss = (
            replacement_loss
            if step <= int(slot_pretrain_steps)
            else answer_loss + float(replacement_loss_weight) * replacement_loss
        )
        if not torch.isfinite(loss):
            raise RuntimeError("replacement QA training produced nonfinite loss")
        loss.backward()
        parameters = tuple(decoder.parameters()) + tuple(controller.parameters())
        torch.nn.utils.clip_grad_norm_(parameters, float(gradient_clip_norm))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        answer_losses.append(float(answer_loss.detach().cpu()))
        replacement_losses.append(float(replacement_loss.detach().cpu()))
        replacement_accuracies.append(
            float(
                (built.controller_output.slots == targets)
                .to(torch.float32)
                .mean()
                .detach()
                .cpu()
            )
        )
        if on_step is not None:
            on_step(step, losses[-1])
    return ReplacementQATrainingHistory(
        losses=tuple(losses),
        answer_losses=tuple(answer_losses),
        replacement_losses=tuple(replacement_losses),
        replacement_accuracies=tuple(replacement_accuracies),
    )
