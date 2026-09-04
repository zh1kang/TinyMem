"""Training and gradient diagnostics for segmented byte QA."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from numbers import Integral, Real

import torch
from torch.optim import Optimizer

from tinymem.data.memory_required_qa import MemoryRequiredQAExample
from tinymem.model.continuous_decoder import (
    SegmentedContinuousDecoder,
    SegmentedContinuousOutput,
)
from tinymem.model.memory_input import AttentionMemory
from tinymem.tokenization.byte_tokenizer import ByteTokenizer
from tinymem.training.losses import next_token_cross_entropy


MEMORY_REQUIRED_MODES = ("oracle", "learned")
StepCallback = Callable[[int, float], None]


@dataclass(frozen=True)
class MemoryRequiredQATrainingHistory:
    losses: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CrossSegmentGradientAudit:
    condition: str
    answer_loss: float
    support_hidden_grad_norm: float
    summary_grad_norm: float
    writer_grad_norm: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def collate_memory_required_query(
    examples: Sequence[MemoryRequiredQAExample],
    *,
    pad_id: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return byte IDs with loss only on answer bytes and the final newline."""
    if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)):
        raise TypeError("examples must be a sequence")
    if not examples or not all(
        isinstance(example, MemoryRequiredQAExample) for example in examples
    ):
        raise ValueError("examples must contain MemoryRequiredQAExample values")
    if isinstance(pad_id, bool) or not isinstance(pad_id, Integral):
        raise TypeError("pad_id must be an integer")
    if not 0 <= pad_id < ByteTokenizer.vocab_size:
        raise ValueError("pad_id must be in the byte-token vocabulary")
    segment_lengths = {example.segment_length for example in examples}
    if len(segment_lengths) != 1:
        raise ValueError("all examples in a batch must share a segment length")

    prefixes = [example.query_ids for example in examples]
    sequences = [
        prefix + example.answer_ids + (ord("\n"),)
        for prefix, example in zip(prefixes, examples, strict=True)
    ]
    max_length = max(map(len, sequences))
    input_ids = torch.full(
        (len(examples), max_length),
        int(pad_id),
        dtype=torch.long,
        device=device,
    )
    target_ids = torch.full_like(input_ids, -100)
    token_valid = torch.zeros_like(input_ids, dtype=torch.bool)
    for row, (prefix, sequence) in enumerate(
        zip(prefixes, sequences, strict=True)
    ):
        length = len(sequence)
        input_ids[row, :length] = torch.tensor(
            sequence,
            dtype=torch.long,
            device=device,
        )
        token_valid[row, :length] = True
        target_ids[row, len(prefix) : length] = input_ids[
            row,
            len(prefix) : length,
        ]
    return input_ids, target_ids, token_valid


def collate_memory_required_support(
    examples: Sequence[MemoryRequiredQAExample],
    *,
    pad_id: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad support facts while keeping padding invalid for compression."""
    if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)):
        raise TypeError("examples must be a sequence")
    if not examples or not all(
        isinstance(example, MemoryRequiredQAExample) for example in examples
    ):
        raise ValueError("examples must contain MemoryRequiredQAExample values")
    if isinstance(pad_id, bool) or not isinstance(pad_id, Integral):
        raise TypeError("pad_id must be an integer")
    if not 0 <= pad_id < ByteTokenizer.vocab_size:
        raise ValueError("pad_id must be in the byte-token vocabulary")
    max_length = max(len(example.support_ids) for example in examples)
    input_ids = torch.full(
        (len(examples), max_length),
        int(pad_id),
        dtype=torch.long,
        device=device,
    )
    token_valid = torch.zeros_like(input_ids, dtype=torch.bool)
    for row, example in enumerate(examples):
        length = len(example.support_ids)
        input_ids[row, :length] = torch.tensor(
            example.support_ids,
            dtype=torch.long,
            device=device,
        )
        token_valid[row, :length] = True
    return input_ids, token_valid


def collate_memory_required_distractor(
    examples: Sequence[MemoryRequiredQAExample],
    distractor_index: int,
    *,
    pad_id: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad one aligned distractor segment for a batch."""
    if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)):
        raise TypeError("examples must be a sequence")
    if not examples or not all(
        isinstance(example, MemoryRequiredQAExample) for example in examples
    ):
        raise ValueError("examples must contain MemoryRequiredQAExample values")
    if isinstance(distractor_index, bool) or not isinstance(
        distractor_index,
        Integral,
    ):
        raise TypeError("distractor_index must be an integer")
    distractor_counts = {len(example.distractor_ids) for example in examples}
    if len(distractor_counts) != 1:
        raise ValueError("all examples in a batch must have equal distractor counts")
    distractor_count = next(iter(distractor_counts))
    if not 0 <= distractor_index < distractor_count:
        raise ValueError("distractor_index is out of range")
    if isinstance(pad_id, bool) or not isinstance(pad_id, Integral):
        raise TypeError("pad_id must be an integer")
    if not 0 <= pad_id < ByteTokenizer.vocab_size:
        raise ValueError("pad_id must be in the byte-token vocabulary")

    sequences = [example.distractor_ids[distractor_index] for example in examples]
    max_length = max(map(len, sequences))
    input_ids = torch.full(
        (len(examples), max_length),
        int(pad_id),
        dtype=torch.long,
        device=device,
    )
    token_valid = torch.zeros_like(input_ids, dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        length = len(sequence)
        input_ids[row, :length] = torch.tensor(
            sequence,
            dtype=torch.long,
            device=device,
        )
        token_valid[row, :length] = True
    return input_ids, token_valid


def _output_memory(output: SegmentedContinuousOutput) -> AttentionMemory:
    return AttentionMemory(
        values=output.memory,
        valid=output.memory_valid,
        positions=output.memory_positions,
    )


def unroll_memory_required_prefix(
    decoder: SegmentedContinuousDecoder,
    examples: Sequence[MemoryRequiredQAExample],
    *,
    pad_id: int,
    device: torch.device | str,
    initial_memory: AttentionMemory | None = None,
    write_distractors: bool = True,
) -> AttentionMemory:
    """Write support and each distractor while preserving the autograd graph."""
    if not isinstance(decoder, SegmentedContinuousDecoder):
        raise TypeError("decoder must be a SegmentedContinuousDecoder")
    if not isinstance(write_distractors, bool):
        raise TypeError("write_distractors must be a boolean")
    if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)):
        raise TypeError("examples must be a sequence")
    if not examples or not all(
        isinstance(example, MemoryRequiredQAExample) for example in examples
    ):
        raise ValueError("examples must contain MemoryRequiredQAExample values")
    distractor_counts = {len(example.distractor_ids) for example in examples}
    if len(distractor_counts) != 1:
        raise ValueError("all examples in a batch must have equal distractor counts")
    distractor_count = next(iter(distractor_counts))

    if initial_memory is None:
        support_ids, support_valid = collate_memory_required_support(
            examples,
            pad_id=pad_id,
            device=device,
        )
        memory = _output_memory(decoder(support_ids, support_valid))
    else:
        if not isinstance(initial_memory, AttentionMemory):
            raise TypeError("initial_memory must be an AttentionMemory or None")
        memory = initial_memory

    for distractor_index in range(distractor_count):
        distractor_ids, distractor_valid = collate_memory_required_distractor(
            examples,
            distractor_index,
            pad_id=pad_id,
            device=device,
        )
        memory = _output_memory(
            decoder(
                distractor_ids,
                distractor_valid,
                initial_memory=memory,
                position_offset=(distractor_index + 1) * decoder.segment_length,
                update_memory=write_distractors,
            )
        )
    return memory


def attention_memory_from_values(
    values: torch.Tensor,
    *,
    segment_length: int,
) -> AttentionMemory:
    """Mark the newest slot in each row as valid oracle memory."""
    if not isinstance(values, torch.Tensor):
        raise TypeError("values must be a torch.Tensor")
    if values.ndim != 3 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("values must have shape [batch, slots, model_width]")
    if not values.is_floating_point():
        raise TypeError("values must be floating point")
    if isinstance(segment_length, bool) or not isinstance(segment_length, Integral):
        raise TypeError("segment_length must be an integer")
    if segment_length <= 0:
        raise ValueError("segment_length must be positive")
    valid = torch.zeros(values.shape[:2], dtype=torch.bool, device=values.device)
    valid[:, -1] = True
    positions = torch.full(
        values.shape[:2],
        -1,
        dtype=torch.long,
        device=values.device,
    )
    positions[:, -1] = int(segment_length) - 1
    return AttentionMemory(values=values, valid=valid, positions=positions)


@torch.no_grad()
def precompute_oracle_memories(
    decoder: SegmentedContinuousDecoder,
    examples: Sequence[MemoryRequiredQAExample],
    *,
    device: torch.device | str,
    batch_size: int = 128,
) -> torch.Tensor:
    """Pool only the ground-truth support span into one frozen memory slot."""
    if not isinstance(decoder, SegmentedContinuousDecoder):
        raise TypeError("decoder must be a SegmentedContinuousDecoder")
    if not examples or not all(
        isinstance(example, MemoryRequiredQAExample) for example in examples
    ):
        raise ValueError("examples must contain MemoryRequiredQAExample values")
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    by_length: dict[int, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        by_length[len(example.support_ids)].append(index)
    memories: list[torch.Tensor | None] = [None] * len(examples)
    was_training = decoder.training
    decoder.eval()
    try:
        for length, indices in by_length.items():
            for start in range(0, len(indices), int(batch_size)):
                batch_indices = indices[start : start + int(batch_size)]
                support_ids = torch.tensor(
                    [examples[index].support_ids for index in batch_indices],
                    dtype=torch.long,
                    device=device,
                )
                hidden = decoder.model.forward_hidden(support_ids)
                summary, summary_valid = decoder.compressor(
                    hidden,
                    torch.ones(
                        len(batch_indices),
                        length,
                        dtype=torch.bool,
                        device=device,
                    ),
                )
                if summary.shape[1] != 1 or not summary_valid.all():
                    raise ValueError("oracle span pooling must produce one valid slot")
                for row, example_index in enumerate(batch_indices):
                    values = torch.zeros(
                        decoder.bank.capacity,
                        decoder.model.config.d_model,
                        dtype=summary.dtype,
                        device="cpu",
                    )
                    values[-1] = summary[row, 0].cpu()
                    memories[example_index] = values
    finally:
        decoder.train(was_training)
    if any(memory is None for memory in memories):
        raise RuntimeError("oracle memory precomputation missed an example")
    return torch.stack([memory for memory in memories if memory is not None])


def train_memory_required_qa(
    decoder: SegmentedContinuousDecoder,
    optimizer: Optimizer,
    examples: Sequence[MemoryRequiredQAExample],
    *,
    mode: str,
    steps: int,
    batch_size: int,
    gradient_clip_norm: float,
    pad_id: int,
    device: torch.device | str,
    seed: int,
    oracle_values: torch.Tensor | None = None,
    write_distractors: bool = True,
    on_step: StepCallback | None = None,
) -> MemoryRequiredQATrainingHistory:
    """Train either the reader with oracle slots or the complete memory path."""
    if not isinstance(decoder, SegmentedContinuousDecoder):
        raise TypeError("decoder must be a SegmentedContinuousDecoder")
    if not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be a torch Optimizer")
    if not examples or not all(
        isinstance(example, MemoryRequiredQAExample) for example in examples
    ):
        raise ValueError("examples must contain MemoryRequiredQAExample values")
    if mode not in MEMORY_REQUIRED_MODES:
        raise ValueError(f"mode must be one of {MEMORY_REQUIRED_MODES}")
    for name, value in (("steps", steps), ("batch_size", batch_size), ("seed", seed)):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    if steps <= 0 or batch_size <= 0 or seed < 0:
        raise ValueError("training counts must be positive and seed nonnegative")
    if isinstance(gradient_clip_norm, bool) or not isinstance(
        gradient_clip_norm,
        Real,
    ):
        raise TypeError("gradient_clip_norm must be a real number")
    if gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive")
    if mode == "oracle":
        expected_shape = (
            len(examples),
            decoder.bank.capacity,
            decoder.model.config.d_model,
        )
        if (
            not isinstance(oracle_values, torch.Tensor)
            or oracle_values.shape != expected_shape
        ):
            raise ValueError(f"oracle_values must have shape {expected_shape}")
    elif oracle_values is not None:
        raise ValueError("oracle_values are valid only in oracle mode")
    if not isinstance(write_distractors, bool):
        raise TypeError("write_distractors must be a boolean")
    if on_step is not None and not callable(on_step):
        raise TypeError("on_step must be callable or None")

    generator = torch.Generator().manual_seed(int(seed))
    losses = []
    decoder.train()
    for step in range(1, int(steps) + 1):
        indices = torch.randint(
            len(examples),
            (int(batch_size),),
            generator=generator,
        ).tolist()
        batch = [examples[index] for index in indices]
        input_ids, target_ids, token_valid = collate_memory_required_query(
            batch,
            pad_id=pad_id,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        if mode == "oracle":
            assert oracle_values is not None
            values = oracle_values[indices].to(device=device)
            memory = unroll_memory_required_prefix(
                decoder,
                batch,
                pad_id=pad_id,
                device=device,
                initial_memory=attention_memory_from_values(
                    values,
                    segment_length=decoder.segment_length,
                ),
                write_distractors=write_distractors,
            )
            output = decoder(
                input_ids,
                token_valid,
                initial_memory=memory,
                position_offset=batch[0].query_position_offset,
                update_memory=False,
            )
        else:
            memory = unroll_memory_required_prefix(
                decoder,
                batch,
                pad_id=pad_id,
                device=device,
                write_distractors=write_distractors,
            )
            output = decoder(
                input_ids,
                token_valid,
                initial_memory=memory,
                position_offset=batch[0].query_position_offset,
                update_memory=False,
            )
        loss = next_token_cross_entropy(output.logits, target_ids)
        if not torch.isfinite(loss):
            raise RuntimeError("memory-required QA training produced nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), gradient_clip_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if on_step is not None:
            on_step(step, losses[-1])
    return MemoryRequiredQATrainingHistory(losses=tuple(losses))


def audit_cross_segment_gradients(
    decoder: SegmentedContinuousDecoder,
    example: MemoryRequiredQAExample,
    *,
    condition: str,
    pad_id: int,
    device: torch.device | str,
    write_distractors: bool = True,
) -> CrossSegmentGradientAudit:
    """Measure whether answer loss reaches the support state and writer."""
    if not isinstance(decoder, SegmentedContinuousDecoder):
        raise TypeError("decoder must be a SegmentedContinuousDecoder")
    if not isinstance(example, MemoryRequiredQAExample):
        raise TypeError("example must be a MemoryRequiredQAExample")
    if condition not in ("normal", "drop"):
        raise ValueError("condition must be 'normal' or 'drop'")
    if not isinstance(write_distractors, bool):
        raise TypeError("write_distractors must be a boolean")

    captured: list[tuple[torch.Tensor, torch.Tensor]] = []

    def hide_memory(memory: AttentionMemory) -> AttentionMemory:
        return AttentionMemory(
            values=memory.values,
            valid=torch.zeros_like(memory.valid),
            positions=torch.full_like(memory.positions, -1),
        )

    def capture(
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        del module
        captured.append((inputs[0], output[0]))

    input_ids, target_ids, token_valid = collate_memory_required_query(
        [example],
        pad_id=pad_id,
        device=device,
    )
    was_training = decoder.training
    decoder.train()
    decoder.zero_grad(set_to_none=True)
    handle = decoder.compressor.register_forward_hook(capture)
    try:
        memory = unroll_memory_required_prefix(
            decoder,
            [example],
            pad_id=pad_id,
            device=device,
            write_distractors=write_distractors,
        )
        output = decoder(
            input_ids,
            token_valid,
            initial_memory=memory,
            position_offset=example.query_position_offset,
            update_memory=False,
            memory_intervention=hide_memory if condition == "drop" else None,
        )
        expected_segments = 2 + len(example.distractor_ids)
        if len(captured) != expected_segments:
            raise RuntimeError(
                f"gradient audit requires exactly {expected_segments} segments"
            )
        support_hidden, summary = captured[0]
        support_hidden.retain_grad()
        summary.retain_grad()
        loss = next_token_cross_entropy(output.logits, target_ids)
        loss.backward()
        writer_squared_norm = sum(
            float(parameter.grad.detach().square().sum().cpu())
            for parameter in decoder.compressor.parameters()
            if parameter.grad is not None
        )
        result = CrossSegmentGradientAudit(
            condition=condition,
            answer_loss=float(loss.detach().cpu()),
            support_hidden_grad_norm=(
                float(support_hidden.grad.detach().norm().cpu())
                if support_hidden.grad is not None
                else 0.0
            ),
            summary_grad_norm=(
                float(summary.grad.detach().norm().cpu())
                if summary.grad is not None
                else 0.0
            ),
            writer_grad_norm=writer_squared_norm**0.5,
        )
    finally:
        handle.remove()
        decoder.zero_grad(set_to_none=True)
        decoder.train(was_training)
    return result
