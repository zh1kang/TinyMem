"""Counterfactual evaluations for learned continuous memory."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from tinymem.data.schema import ReasoningExample
from tinymem.data.vocabulary import ControlledVocabulary
from tinymem.evaluation.forgetting_curve import (
    DEFAULT_DELAY_LIMITS,
    DelayResult,
    delay_label,
    qa1_evidence_delay_tokens,
)
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.memory_input import AttentionMemory
from tinymem.training.controlled_qa import format_qa_prompt


MemoryIntervention = Callable[[AttentionMemory], AttentionMemory]


@dataclass(frozen=True)
class ContinuousMemoryResult:
    """Store one continuous-memory forgetting curve and summary accuracy."""

    intervention: str
    correct: int
    count: int
    outside_window_correct: int
    outside_window_count: int
    curve: tuple[DelayResult, ...]

    @property
    def accuracy(self) -> float:
        return self.correct / self.count

    @property
    def outside_window_accuracy(self) -> float:
        if self.outside_window_count == 0:
            return 0.0
        return self.outside_window_correct / self.outside_window_count

    def to_dict(self) -> dict[str, object]:
        return {
            "intervention": self.intervention,
            "correct": self.correct,
            "count": self.count,
            "accuracy": self.accuracy,
            "outside_window_correct": self.outside_window_correct,
            "outside_window_count": self.outside_window_count,
            "outside_window_accuracy": self.outside_window_accuracy,
            "curve": [bucket.to_dict() for bucket in self.curve],
        }


def drop_memory(memory: AttentionMemory) -> AttentionMemory:
    """Make every slot unavailable while preserving allocated shapes."""
    return AttentionMemory(
        values=memory.values,
        valid=torch.zeros_like(memory.valid),
        positions=torch.full_like(memory.positions, -1),
    )


def zero_memory(memory: AttentionMemory) -> AttentionMemory:
    """Remove stored content while preserving occupancy and positions."""
    return AttentionMemory(
        values=torch.zeros_like(memory.values),
        valid=memory.valid,
        positions=memory.positions,
    )


def shuffle_memory(memory: AttentionMemory) -> AttentionMemory:
    """Rotate complete memory rows across batch items."""
    if memory.values.shape[0] < 2:
        raise ValueError("shuffled-memory evaluation requires batch size above one")
    return AttentionMemory(
        values=memory.values.roll(1, dims=0),
        valid=memory.valid.roll(1, dims=0),
        positions=memory.positions.roll(1, dims=0),
    )


def _build_curve(
    counts: dict[str, list[int]],
    limits: tuple[int, ...],
) -> tuple[DelayResult, ...]:
    results = []
    lower = 0
    for limit in limits:
        label = f"{lower}-{limit}"
        correct, count = counts.get(label, [0, 0])
        results.append(DelayResult(label, lower, limit, correct, count))
        lower = limit + 1
    label = f">{limits[-1]}"
    correct, count = counts.get(label, [0, 0])
    results.append(DelayResult(label, lower, None, correct, count))
    return tuple(results)


@torch.no_grad()
def evaluate_continuous_qa1(
    decoder: SegmentedContinuousDecoder,
    vocabulary: ControlledVocabulary,
    examples: Sequence[ReasoningExample],
    *,
    batch_size: int,
    device: torch.device | str,
    intervention_name: str = "normal",
    memory_intervention: MemoryIntervention | None = None,
    limits: tuple[int, ...] = DEFAULT_DELAY_LIMITS,
) -> ContinuousMemoryResult:
    """Evaluate exact qa1 answers through the segmented decoder."""
    if not isinstance(decoder, SegmentedContinuousDecoder):
        raise TypeError("decoder must be a SegmentedContinuousDecoder")
    if not isinstance(vocabulary, ControlledVocabulary):
        raise TypeError("vocabulary must be a ControlledVocabulary")
    if not examples:
        raise ValueError("examples must be nonempty")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not isinstance(intervention_name, str):
        raise TypeError("intervention_name must be a string")
    if not intervention_name:
        raise ValueError("intervention_name must be nonempty")
    if memory_intervention is not None and not callable(memory_intervention):
        raise TypeError("memory_intervention must be callable or None")
    if memory_intervention is shuffle_memory and batch_size < 2:
        raise ValueError("shuffled-memory evaluation requires batch size above one")
    if not limits or any(limit <= 0 for limit in limits):
        raise ValueError("limits must contain positive values")
    if tuple(sorted(set(limits))) != limits:
        raise ValueError("limits must be unique and strictly increasing")

    prepared = []
    for example in examples:
        prompt_ids = tuple(
            vocabulary.encode(format_qa_prompt(example), add_bos=True)
        )
        answer_ids = vocabulary.encode(example.answer)
        if len(answer_ids) != 1:
            raise ValueError("controlled answer must encode to exactly one token")
        prepared.append(
            (
                prompt_ids,
                answer_ids[0],
                qa1_evidence_delay_tokens(example, vocabulary),
            )
        )
    prepared.sort(key=lambda item: len(item[0]))

    was_training = decoder.training
    decoder.eval()
    correct = 0
    outside_window_correct = 0
    outside_window_count = 0
    delay_counts: dict[str, list[int]] = {}
    try:
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start : start + batch_size]
            max_length = max(len(item[0]) for item in batch)
            input_ids = torch.full(
                (len(batch), max_length),
                vocabulary.token_to_id["<pad>"],
                dtype=torch.long,
                device=device,
            )
            token_valid = torch.zeros_like(input_ids, dtype=torch.bool)
            prompt_lengths = []
            for row, (prompt_ids, _, _) in enumerate(batch):
                length = len(prompt_ids)
                input_ids[row, :length] = torch.tensor(
                    prompt_ids,
                    dtype=torch.long,
                    device=device,
                )
                token_valid[row, :length] = True
                prompt_lengths.append(length)

            output = decoder(
                input_ids,
                token_valid,
                memory_intervention=memory_intervention,
            )
            rows = torch.arange(len(batch), device=device)
            final_positions = torch.tensor(
                [length - 1 for length in prompt_lengths],
                device=device,
            )
            predictions = output.logits[rows, final_positions].argmax(dim=-1).cpu()
            for prediction, (_, answer_id, delay) in zip(
                predictions,
                batch,
                strict=True,
            ):
                is_correct = int(prediction == answer_id)
                correct += is_correct
                if delay > decoder.segment_length:
                    outside_window_correct += is_correct
                    outside_window_count += 1
                label = delay_label(delay, limits)
                totals = delay_counts.setdefault(label, [0, 0])
                totals[0] += is_correct
                totals[1] += 1
    finally:
        decoder.train(was_training)

    return ContinuousMemoryResult(
        intervention=intervention_name,
        correct=correct,
        count=len(prepared),
        outside_window_correct=outside_window_correct,
        outside_window_count=outside_window_count,
        curve=_build_curve(delay_counts, limits),
    )
