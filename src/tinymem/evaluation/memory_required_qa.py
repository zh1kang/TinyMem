"""Causal interventions for segmented byte memory experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from numbers import Integral

import torch
from torch.nn import functional as F

from tinymem.data.memory_required_qa import MemoryRequiredQAExample
from tinymem.evaluation.continuous_memory import drop_memory, zero_memory
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.memory_input import AttentionMemory
from tinymem.tokenization.byte_tokenizer import ByteTokenizer
from tinymem.training.memory_required_qa import (
    MEMORY_REQUIRED_MODES,
    attention_memory_from_values,
    unroll_memory_required_prefix,
)


@dataclass(frozen=True)
class MemoryRequiredQAPrediction:
    source_example_id: str
    prediction: str
    reference: str
    exact_match: bool
    first_byte_correct: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryRequiredQAConditionResult:
    condition: str
    count: int
    exact_accuracy: float
    first_byte_accuracy: float
    answer_byte_nll: float
    mean_first_byte_logit_linf_from_normal: float
    predictions: tuple[MemoryRequiredQAPrediction, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "predictions": [prediction.to_dict() for prediction in self.predictions],
        }


def _empty_memory(
    decoder: SegmentedContinuousDecoder,
    *,
    batch_size: int,
    device: torch.device | str,
) -> AttentionMemory:
    return AttentionMemory(
        values=torch.zeros(
            batch_size,
            decoder.bank.capacity,
            decoder.model.config.d_model,
            dtype=decoder.model.token_embedding.weight.dtype,
            device=device,
        ),
        valid=torch.zeros(
            batch_size,
            decoder.bank.capacity,
            dtype=torch.bool,
            device=device,
        ),
        positions=torch.full(
            (batch_size, decoder.bank.capacity),
            -1,
            dtype=torch.long,
            device=device,
        ),
    )


@torch.no_grad()
def _learned_memories(
    decoder: SegmentedContinuousDecoder,
    examples: Sequence[MemoryRequiredQAExample],
    *,
    device: torch.device | str,
    batch_size: int,
) -> AttentionMemory:
    values = []
    valid = []
    positions = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        memory = unroll_memory_required_prefix(
            decoder,
            batch,
            pad_id=ByteTokenizer.special_tokens["<pad>"],
            device=device,
        )
        values.append(memory.values)
        valid.append(memory.valid)
        positions.append(memory.positions)
    return AttentionMemory(
        values=torch.cat(values),
        valid=torch.cat(valid),
        positions=torch.cat(positions),
    )


def _mismatched_indices(
    examples: Sequence[MemoryRequiredQAExample],
    *,
    device: torch.device | str,
) -> torch.Tensor:
    answers = [example.answer_ids for example in examples]
    if len(set(answers)) < 2:
        raise ValueError("shuffled evaluation requires at least two answer classes")
    indices = []
    for index, answer in enumerate(answers):
        for offset in range(1, len(answers)):
            candidate = (index + offset) % len(answers)
            if answers[candidate] != answer:
                indices.append(candidate)
                break
        else:
            raise RuntimeError("could not construct an answer-mismatched shuffle")
    return torch.tensor(indices, dtype=torch.long, device=device)


def _select_condition(
    memory: AttentionMemory,
    *,
    condition: str,
    shuffle_indices: torch.Tensor,
    decoder: SegmentedContinuousDecoder,
    device: torch.device | str,
) -> AttentionMemory:
    if condition == "normal":
        return memory
    if condition == "drop":
        return drop_memory(memory)
    if condition == "zero":
        return zero_memory(memory)
    if condition == "shuffle":
        return AttentionMemory(
            values=memory.values[shuffle_indices],
            valid=memory.valid[shuffle_indices],
            positions=memory.positions[shuffle_indices],
        )
    if condition == "no_writes":
        return _empty_memory(
            decoder,
            batch_size=memory.values.shape[0],
            device=device,
        )
    raise ValueError("unknown memory condition")


@torch.no_grad()
def _score_and_generate(
    decoder: SegmentedContinuousDecoder,
    example: MemoryRequiredQAExample,
    memory: AttentionMemory,
    *,
    device: torch.device | str,
    max_new_tokens: int,
) -> tuple[str, float, bool, torch.Tensor]:
    sequence = example.query_ids + example.answer_ids + (ord("\n"),)
    input_ids = torch.tensor(sequence, dtype=torch.long, device=device).unsqueeze(0)
    output = decoder(
        input_ids,
        torch.ones_like(input_ids, dtype=torch.bool),
        initial_memory=memory,
        position_offset=example.query_position_offset,
        update_memory=False,
    )
    answer_start = len(example.query_ids)
    answer_logits = output.logits[:, answer_start - 1 : -1]
    answer_targets = input_ids[:, answer_start:]
    total_nll = float(
        F.cross_entropy(
            answer_logits.reshape(-1, answer_logits.shape[-1]),
            answer_targets.reshape(-1),
            reduction="sum",
        ).cpu()
    )
    first_logits = output.logits[0, answer_start - 1].detach().cpu()
    first_byte_correct = int(first_logits.argmax()) == example.answer_ids[0]

    generated: list[int] = []
    for _ in range(max_new_tokens):
        current = torch.tensor(
            example.query_ids + tuple(generated),
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        generated_output = decoder(
            current,
            torch.ones_like(current, dtype=torch.bool),
            initial_memory=memory,
            position_offset=example.query_position_offset,
            update_memory=False,
        )
        next_id = int(generated_output.logits[0, -1].argmax())
        if next_id in (ord("\n"), ByteTokenizer.special_tokens["<eos>"]):
            break
        if not 0 <= next_id <= 255:
            break
        generated.append(next_id)
    prediction = bytes(generated).decode("utf-8", errors="replace").strip()
    return prediction, total_nll, first_byte_correct, first_logits


@torch.no_grad()
def evaluate_memory_required_qa(
    decoder: SegmentedContinuousDecoder,
    examples: Sequence[MemoryRequiredQAExample],
    *,
    mode: str,
    device: torch.device | str,
    max_new_tokens: int,
    oracle_values: torch.Tensor | None = None,
    memory_batch_size: int = 128,
) -> dict[str, MemoryRequiredQAConditionResult]:
    """Evaluate correct, absent, zero, wrong, and unwritten memory states."""
    if not isinstance(decoder, SegmentedContinuousDecoder):
        raise TypeError("decoder must be a SegmentedContinuousDecoder")
    if not examples or not all(
        isinstance(example, MemoryRequiredQAExample) for example in examples
    ):
        raise ValueError("examples must contain MemoryRequiredQAExample values")
    if mode not in MEMORY_REQUIRED_MODES:
        raise ValueError(f"mode must be one of {MEMORY_REQUIRED_MODES}")
    for name, value in (
        ("max_new_tokens", max_new_tokens),
        ("memory_batch_size", memory_batch_size),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    segment_lengths = {example.segment_length for example in examples}
    if segment_lengths != {decoder.segment_length}:
        raise ValueError("example and decoder segment lengths must match")
    if any(
        len(example.query_ids) + int(max_new_tokens) > decoder.segment_length
        for example in examples
    ):
        raise ValueError("max_new_tokens would exceed the query segment")

    was_training = decoder.training
    decoder.eval()
    try:
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
            oracle_memory = attention_memory_from_values(
                oracle_values.to(device=device),
                segment_length=decoder.segment_length,
            )
            memory = unroll_memory_required_prefix(
                decoder,
                examples,
                pad_id=ByteTokenizer.special_tokens["<pad>"],
                device=device,
                initial_memory=oracle_memory,
            )
            conditions = ("normal", "drop", "zero", "shuffle")
        else:
            if oracle_values is not None:
                raise ValueError("oracle_values are valid only in oracle mode")
            memory = _learned_memories(
                decoder,
                examples,
                device=device,
                batch_size=int(memory_batch_size),
            )
            conditions = ("normal", "drop", "zero", "shuffle", "no_writes")

        shuffle_indices = _mismatched_indices(examples, device=device)
        raw_results: dict[
            str,
            tuple[
                tuple[MemoryRequiredQAPrediction, ...],
                float,
                int,
                tuple[torch.Tensor, ...],
            ],
        ] = {}
        for condition in conditions:
            conditioned = _select_condition(
                memory,
                condition=condition,
                shuffle_indices=shuffle_indices,
                decoder=decoder,
                device=device,
            )
            predictions = []
            total_nll = 0.0
            total_bytes = 0
            first_logits = []
            for index, example in enumerate(examples):
                row_memory = AttentionMemory(
                    values=conditioned.values[index : index + 1],
                    valid=conditioned.valid[index : index + 1],
                    positions=conditioned.positions[index : index + 1],
                )
                prediction, nll, first_correct, logits = _score_and_generate(
                    decoder,
                    example,
                    row_memory,
                    device=device,
                    max_new_tokens=int(max_new_tokens),
                )
                reference = ByteTokenizer().decode(example.answer_ids)
                predictions.append(
                    MemoryRequiredQAPrediction(
                        source_example_id=example.source_example_id,
                        prediction=prediction,
                        reference=reference,
                        exact_match=prediction.casefold() == reference.casefold(),
                        first_byte_correct=first_correct,
                    )
                )
                total_nll += nll
                total_bytes += len(example.answer_ids) + 1
                first_logits.append(logits)
            raw_results[condition] = (
                tuple(predictions),
                total_nll,
                total_bytes,
                tuple(first_logits),
            )

        normal_logits = raw_results["normal"][3]
        results = {}
        for condition, (
            predictions,
            total_nll,
            total_bytes,
            first_logits,
        ) in raw_results.items():
            results[condition] = MemoryRequiredQAConditionResult(
                condition=condition,
                count=len(predictions),
                exact_accuracy=sum(item.exact_match for item in predictions)
                / len(predictions),
                first_byte_accuracy=sum(
                    item.first_byte_correct for item in predictions
                )
                / len(predictions),
                answer_byte_nll=total_nll / total_bytes,
                mean_first_byte_logit_linf_from_normal=sum(
                    float((current - normal).abs().max())
                    for current, normal in zip(
                        first_logits,
                        normal_logits,
                        strict=True,
                    )
                )
                / len(predictions),
                predictions=predictions,
            )
        return results
    finally:
        decoder.train(was_training)
