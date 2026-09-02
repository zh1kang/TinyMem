"""Counterfactual evaluations for learned continuous memory."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import comb, floor

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
from tinymem.training.continuous import (
    collate_segmented_answer_supervision,
    evidence_segment_write_targets,
)
from tinymem.training.controlled_qa import EncodedQAExample, format_qa_prompt


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
    correctness: tuple[bool, ...]
    outside_window_correctness: tuple[bool, ...]
    writes: WriteDecisionResult

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
            "writes": self.writes.to_dict(),
        }


@dataclass(frozen=True)
class WriteDecisionResult:
    """Store binary write-selection counts for labeled segments."""

    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    @property
    def accuracy(self) -> float:
        total = (
            self.true_positive
            + self.false_positive
            + self.false_negative
            + self.true_negative
        )
        return (self.true_positive + self.true_negative) / total

    @property
    def precision(self) -> float:
        predicted_positive = self.true_positive + self.false_positive
        if predicted_positive == 0:
            return 0.0
        return self.true_positive / predicted_positive

    @property
    def recall(self) -> float:
        actual_positive = self.true_positive + self.false_negative
        if actual_positive == 0:
            return 0.0
        return self.true_positive / actual_positive

    @property
    def write_rate(self) -> float:
        total = (
            self.true_positive
            + self.false_positive
            + self.false_negative
            + self.true_negative
        )
        return (self.true_positive + self.false_positive) / total

    def to_dict(self) -> dict[str, object]:
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "true_negative": self.true_negative,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "write_rate": self.write_rate,
        }


@dataclass(frozen=True)
class WriteThresholdCalibration:
    """Store a threshold selected under a background-write budget."""

    threshold: float
    max_false_positive_rate: float
    allowed_false_positives: int
    writes: WriteDecisionResult

    def to_dict(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "max_false_positive_rate": self.max_false_positive_rate,
            "allowed_false_positives": self.allowed_false_positives,
            "writes": self.writes.to_dict(),
        }


@dataclass(frozen=True)
class PairedAccuracyResult:
    """Store a one-sided exact paired comparison against normal memory."""

    normal_only_correct: int
    intervention_only_correct: int
    ties: int
    alpha: float

    @property
    def count(self) -> int:
        return (
            self.normal_only_correct
            + self.intervention_only_correct
            + self.ties
        )

    @property
    def accuracy_difference(self) -> float:
        return (
            self.normal_only_correct - self.intervention_only_correct
        ) / self.count

    @property
    def one_sided_p_value(self) -> float:
        discordant = self.normal_only_correct + self.intervention_only_correct
        if discordant == 0:
            return 1.0
        numerator = sum(
            comb(discordant, successes)
            for successes in range(self.normal_only_correct, discordant + 1)
        )
        return numerator / (2**discordant)

    @property
    def significant(self) -> bool:
        return (
            self.normal_only_correct > self.intervention_only_correct
            and self.one_sided_p_value <= self.alpha
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "normal_only_correct": self.normal_only_correct,
            "intervention_only_correct": self.intervention_only_correct,
            "ties": self.ties,
            "count": self.count,
            "accuracy_difference": self.accuracy_difference,
            "one_sided_p_value": self.one_sided_p_value,
            "alpha": self.alpha,
            "significant": self.significant,
        }


@dataclass(frozen=True)
class ContinuousAnswerResult:
    """Store exact answer accuracy for one memory intervention."""

    intervention: str
    correct: int
    count: int
    correctness: tuple[bool, ...]
    writes: WriteDecisionResult | None = None

    @property
    def accuracy(self) -> float:
        return self.correct / self.count

    def to_dict(self) -> dict[str, object]:
        document: dict[str, object] = {
            "intervention": self.intervention,
            "correct": self.correct,
            "count": self.count,
            "accuracy": self.accuracy,
        }
        if self.writes is not None:
            document["writes"] = self.writes.to_dict()
        return document


def paired_accuracy_test(
    normal: Sequence[bool],
    intervention: Sequence[bool],
    *,
    alpha: float,
) -> PairedAccuracyResult:
    """Compare paired correctness with an exact one-sided sign test."""
    if not isinstance(normal, Sequence) or isinstance(normal, (str, bytes)):
        raise TypeError("normal must be a sequence of booleans")
    if not isinstance(intervention, Sequence) or isinstance(
        intervention,
        (str, bytes),
    ):
        raise TypeError("intervention must be a sequence of booleans")
    if len(normal) != len(intervention):
        raise ValueError("paired correctness sequences must have equal length")
    if not normal:
        raise ValueError("paired correctness sequences must be nonempty")
    if not all(isinstance(value, bool) for value in (*normal, *intervention)):
        raise TypeError("paired correctness sequences must contain booleans")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be a real number")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")

    normal_only = 0
    intervention_only = 0
    ties = 0
    for normal_correct, intervention_correct in zip(
        normal,
        intervention,
        strict=True,
    ):
        if normal_correct and not intervention_correct:
            normal_only += 1
        elif intervention_correct and not normal_correct:
            intervention_only += 1
        else:
            ties += 1
    return PairedAccuracyResult(
        normal_only_correct=normal_only,
        intervention_only_correct=intervention_only,
        ties=ties,
        alpha=float(alpha),
    )


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


def _evaluation_batches(
    count: int,
    batch_size: int,
    *,
    require_pairs: bool,
) -> tuple[slice, ...]:
    if require_pairs and count < 2:
        raise ValueError("shuffled-memory evaluation requires at least two examples")
    if require_pairs and batch_size < 2:
        raise ValueError("shuffled-memory evaluation requires batch size above one")
    batches = [
        slice(start, min(start + batch_size, count))
        for start in range(0, count, batch_size)
    ]
    if require_pairs and len(batches) > 1:
        final = batches[-1]
        if final.stop - final.start == 1:
            previous = batches[-2]
            batches[-2:] = [slice(previous.start, final.stop)]
    return tuple(batches)


def _write_decision_counts(
    actual: torch.Tensor,
    expected: Sequence[bool],
) -> tuple[int, int, int, int]:
    expected_tensor = torch.tensor(expected, dtype=torch.bool)
    return (
        int((actual & expected_tensor).sum()),
        int((actual & ~expected_tensor).sum()),
        int((~actual & expected_tensor).sum()),
        int((~actual & ~expected_tensor).sum()),
    )


@torch.no_grad()
def calibrate_write_threshold(
    decoder: SegmentedContinuousDecoder,
    examples: Sequence[EncodedQAExample],
    *,
    batch_size: int,
    pad_id: int,
    device: torch.device | str,
    max_false_positive_rate: float,
    minimum_threshold: float = 0.5,
) -> WriteThresholdCalibration:
    """Select the lowest threshold that meets a false-positive budget."""
    if not isinstance(decoder, SegmentedContinuousDecoder):
        raise TypeError("decoder must be a SegmentedContinuousDecoder")
    if not examples:
        raise ValueError("examples must be nonempty")
    if not all(isinstance(example, EncodedQAExample) for example in examples):
        raise TypeError("examples must contain EncodedQAExample values")
    if not all(example.segment_write_targets is not None for example in examples):
        raise ValueError("calibration requires write targets for every example")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for name, value in (
        ("max_false_positive_rate", max_false_positive_rate),
        ("minimum_threshold", minimum_threshold),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real number")
    if not 0 <= max_false_positive_rate < 1:
        raise ValueError("max_false_positive_rate must be in [0, 1)")
    if not 0 < minimum_threshold <= 1:
        raise ValueError("minimum_threshold must be in (0, 1]")

    probabilities = []
    targets = []
    was_training = decoder.training
    decoder.eval()
    try:
        for batch_slice in _evaluation_batches(
            len(examples),
            batch_size,
            require_pairs=False,
        ):
            batch = examples[batch_slice]
            input_ids, _, token_valid = collate_segmented_answer_supervision(
                batch,
                pad_id=pad_id,
                device=device,
            )
            output = decoder(input_ids, token_valid)
            if output.write_logits is None:
                raise ValueError("calibration requires a learned write gate")
            for row, example in enumerate(batch):
                write_targets = example.segment_write_targets
                assert write_targets is not None
                expected_count = (
                    len(example.input_ids) + decoder.segment_length - 1
                ) // decoder.segment_length
                if len(write_targets) != expected_count:
                    raise ValueError(
                        "segment write targets must match the encoded sequence"
                    )
                probabilities.append(
                    output.write_logits[row, :expected_count].sigmoid().cpu()
                )
                targets.append(torch.tensor(write_targets, dtype=torch.bool))
    finally:
        decoder.train(was_training)

    probability_tensor = torch.cat(probabilities)
    target_tensor = torch.cat(targets)
    positive_count = int(target_tensor.sum())
    negative_count = int((~target_tensor).sum())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("calibration requires positive and negative write targets")
    allowed_false_positives = floor(
        max_false_positive_rate * negative_count
    )
    descending_negatives = probability_tensor[~target_tensor].sort(
        descending=True
    ).values
    threshold = float(minimum_threshold)
    if allowed_false_positives < negative_count:
        boundary = descending_negatives[allowed_false_positives]
        strict_boundary = torch.nextafter(
            boundary,
            torch.tensor(float("inf"), dtype=boundary.dtype),
        )
        threshold = max(threshold, min(float(strict_boundary), 1.0))

    predicted = probability_tensor >= threshold
    true_positive = int((predicted & target_tensor).sum())
    false_positive = int((predicted & ~target_tensor).sum())
    false_negative = int((~predicted & target_tensor).sum())
    true_negative = int((~predicted & ~target_tensor).sum())
    return WriteThresholdCalibration(
        threshold=threshold,
        max_false_positive_rate=float(max_false_positive_rate),
        allowed_false_positives=allowed_false_positives,
        writes=WriteDecisionResult(
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
            true_negative=true_negative,
        ),
    )


@torch.no_grad()
def evaluate_continuous_answers(
    decoder: SegmentedContinuousDecoder,
    examples: Sequence[EncodedQAExample],
    *,
    batch_size: int,
    pad_id: int,
    device: torch.device | str,
    intervention_name: str = "normal",
    memory_intervention: MemoryIntervention | None = None,
    forced_writes: torch.Tensor | None = None,
) -> ContinuousAnswerResult:
    """Evaluate exact answer accuracy on encoded controlled examples."""
    if not isinstance(decoder, SegmentedContinuousDecoder):
        raise TypeError("decoder must be a SegmentedContinuousDecoder")
    if not examples:
        raise ValueError("examples must be nonempty")
    if not all(isinstance(example, EncodedQAExample) for example in examples):
        raise TypeError("examples must contain EncodedQAExample values")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not isinstance(intervention_name, str) or not intervention_name:
        raise ValueError("intervention_name must be a nonempty string")
    if memory_intervention is not None and not callable(memory_intervention):
        raise TypeError("memory_intervention must be callable or None")
    if forced_writes is not None:
        if not isinstance(forced_writes, torch.Tensor):
            raise TypeError("forced_writes must be a torch.Tensor or None")
        if forced_writes.ndim != 2 or forced_writes.shape[0] != len(examples):
            raise ValueError(
                "forced_writes must have shape [examples, padded_segments]"
            )
        if forced_writes.dtype != torch.bool:
            raise TypeError("forced_writes must be a boolean tensor")
        required_segments = max(
            (len(example.input_ids) + decoder.segment_length - 1)
            // decoder.segment_length
            for example in examples
        )
        if forced_writes.shape[1] < required_segments:
            raise ValueError("forced_writes does not cover every example segment")

    batches = _evaluation_batches(
        len(examples),
        batch_size,
        require_pairs=memory_intervention is shuffle_memory,
    )
    was_training = decoder.training
    decoder.eval()
    correct = 0
    correctness = []
    has_write_targets = all(
        example.segment_write_targets is not None for example in examples
    )
    if not has_write_targets and any(
        example.segment_write_targets is not None for example in examples
    ):
        raise ValueError("write targets must be present for every example or none")
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    try:
        for batch_slice in batches:
            batch = examples[batch_slice]
            input_ids, _, token_valid = collate_segmented_answer_supervision(
                batch,
                pad_id=pad_id,
                device=device,
            )
            output = decoder(
                input_ids,
                token_valid,
                memory_intervention=memory_intervention,
                forced_writes=(
                    forced_writes[
                        batch_slice,
                        : (
                            input_ids.shape[1]
                            + decoder.segment_length
                            - 1
                        )
                        // decoder.segment_length,
                    ].to(device=device)
                    if forced_writes is not None
                    else None
                ),
            )
            rows = torch.arange(len(batch), device=device)
            prompt_positions = torch.tensor(
                [len(example.input_ids) - 2 for example in batch],
                device=device,
            )
            predictions = output.logits[
                rows,
                prompt_positions,
            ].argmax(dim=-1).cpu()
            batch_correctness = tuple(
                bool(prediction == example.answer_id)
                for prediction, example in zip(predictions, batch, strict=True)
            )
            correctness.extend(batch_correctness)
            correct += sum(batch_correctness)
            if has_write_targets:
                for row, example in enumerate(batch):
                    targets = example.segment_write_targets
                    assert targets is not None
                    expected_count = (
                        len(example.input_ids) + decoder.segment_length - 1
                    ) // decoder.segment_length
                    if len(targets) != expected_count:
                        raise ValueError(
                            "segment write targets must match the encoded sequence"
                        )
                    actual = output.writes_applied[row, :expected_count].cpu()
                    counts = _write_decision_counts(actual, targets)
                    true_positive += counts[0]
                    false_positive += counts[1]
                    false_negative += counts[2]
                    true_negative += counts[3]
    finally:
        decoder.train(was_training)

    writes = None
    if has_write_targets:
        writes = WriteDecisionResult(
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
            true_negative=true_negative,
        )
    return ContinuousAnswerResult(
        intervention=intervention_name,
        correct=correct,
        count=len(examples),
        correctness=tuple(correctness),
        writes=writes,
    )


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
                evidence_segment_write_targets(
                    example,
                    vocabulary,
                    segment_length=decoder.segment_length,
                    prompt_ids=prompt_ids,
                ),
            )
        )
    prepared.sort(key=lambda item: len(item[0]))

    was_training = decoder.training
    decoder.eval()
    correct = 0
    outside_window_correct = 0
    outside_window_count = 0
    correctness = []
    outside_window_correctness = []
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    delay_counts: dict[str, list[int]] = {}
    try:
        for batch_slice in _evaluation_batches(
            len(prepared),
            batch_size,
            require_pairs=memory_intervention is shuffle_memory,
        ):
            batch = prepared[batch_slice]
            max_length = max(len(item[0]) for item in batch)
            input_ids = torch.full(
                (len(batch), max_length),
                vocabulary.token_to_id["<pad>"],
                dtype=torch.long,
                device=device,
            )
            token_valid = torch.zeros_like(input_ids, dtype=torch.bool)
            prompt_lengths = []
            for row, (prompt_ids, _, _, _) in enumerate(batch):
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
            paired_batch = zip(predictions, batch, strict=True)
            for row, (
                prediction,
                (_, answer_id, delay, write_targets),
            ) in enumerate(paired_batch):
                is_correct = bool(prediction == answer_id)
                correctness.append(is_correct)
                correct += is_correct
                if delay > decoder.segment_length:
                    outside_window_correct += is_correct
                    outside_window_count += 1
                    outside_window_correctness.append(is_correct)
                label = delay_label(delay, limits)
                totals = delay_counts.setdefault(label, [0, 0])
                totals[0] += is_correct
                totals[1] += 1
                actual_writes = output.writes_applied[
                    row,
                    : len(write_targets),
                ].cpu()
                counts = _write_decision_counts(actual_writes, write_targets)
                true_positive += counts[0]
                false_positive += counts[1]
                false_negative += counts[2]
                true_negative += counts[3]
    finally:
        decoder.train(was_training)

    return ContinuousMemoryResult(
        intervention=intervention_name,
        correct=correct,
        count=len(prepared),
        outside_window_correct=outside_window_correct,
        outside_window_count=outside_window_count,
        curve=_build_curve(delay_counts, limits),
        correctness=tuple(correctness),
        outside_window_correctness=tuple(outside_window_correctness),
        writes=WriteDecisionResult(
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
            true_negative=true_negative,
        ),
    )
