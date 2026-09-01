"""Matched-budget evaluation for fixed qa1 memory baselines."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from tinymem.data.schema import ReasoningExample
from tinymem.data.vocabulary import ControlledVocabulary
from tinymem.evaluation.forgetting_curve import (
    DEFAULT_DELAY_LIMITS,
    DelayResult,
    delay_label,
    qa1_evidence_delay_tokens,
    qa1_evidence_token_positions,
)
from tinymem.memory.heavy_hitter import HeavyHitterMemory
from tinymem.memory.importance import ExtractiveImportanceMemory
from tinymem.memory.interfaces import MemoryPolicy
from tinymem.memory.oracle import OracleMemory
from tinymem.memory.recent_tokens import RecentTokenMemory
from tinymem.memory.reservoir import RandomReservoirMemory
from tinymem.model.streaming import StreamingDecoder
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.training.controlled_qa import format_qa_prompt


BASELINE_NAMES = (
    "local",
    "recent",
    "reservoir",
    "importance",
    "heavy_hitter",
    "oracle",
)


@dataclass(frozen=True)
class BaselineResult:
    """Store exact qa1 accuracy and per-example memory cost."""

    baseline: str
    capacity: int
    memory_bytes: int
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
            "baseline": self.baseline,
            "capacity": self.capacity,
            "memory_bytes": self.memory_bytes,
            "correct": self.correct,
            "count": self.count,
            "accuracy": self.accuracy,
            "outside_window_correct": self.outside_window_correct,
            "outside_window_count": self.outside_window_count,
            "outside_window_accuracy": self.outside_window_accuracy,
            "curve": [bucket.to_dict() for bucket in self.curve],
        }


@dataclass(frozen=True)
class _PreparedExample:
    prompt_ids: tuple[int, ...]
    answer_id: int
    delay: int
    evidence_positions: tuple[int, ...]


def _prepare_examples(
    examples: Sequence[ReasoningExample],
    vocabulary: ControlledVocabulary,
) -> list[_PreparedExample]:
    prepared = []
    for example in examples:
        prompt_ids = tuple(
            vocabulary.encode(format_qa_prompt(example), add_bos=True)
        )
        answer_ids = vocabulary.encode(example.answer)
        if len(answer_ids) != 1:
            raise ValueError("controlled answer must encode to exactly one token")
        prepared.append(
            _PreparedExample(
                prompt_ids=prompt_ids,
                answer_id=answer_ids[0],
                delay=qa1_evidence_delay_tokens(example, vocabulary),
                evidence_positions=qa1_evidence_token_positions(
                    example,
                    vocabulary,
                ),
            )
        )
    prepared.sort(key=lambda item: len(item.prompt_ids))
    return prepared


def _target_position_tensor(
    batch: Sequence[_PreparedExample],
    *,
    device: torch.device | str,
) -> torch.Tensor:
    width = max(len(item.evidence_positions) for item in batch)
    positions = torch.full(
        (len(batch), width),
        -1,
        dtype=torch.long,
        device=device,
    )
    for row, item in enumerate(batch):
        positions[row, : len(item.evidence_positions)] = torch.tensor(
            item.evidence_positions,
            dtype=torch.long,
            device=device,
        )
    return positions


def _make_policy(
    baseline: str,
    capacity: int,
    target_positions: torch.Tensor,
) -> MemoryPolicy | None:
    if baseline == "local":
        return None
    if baseline == "recent":
        return RecentTokenMemory(capacity)
    if baseline == "reservoir":
        return RandomReservoirMemory(capacity)
    if baseline == "importance":
        return ExtractiveImportanceMemory(capacity)
    if baseline == "heavy_hitter":
        if capacity < 2:
            raise ValueError("heavy_hitter requires capacity of at least two")
        return HeavyHitterMemory(capacity, recent_slots=max(1, capacity // 2))
    if baseline == "oracle":
        return OracleMemory(capacity, target_positions=target_positions)
    supported = ", ".join(BASELINE_NAMES)
    raise ValueError(f"unsupported baseline {baseline!r}; choose from: {supported}")


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
def evaluate_qa1_baseline(
    model: DecoderOnlyTransformer,
    vocabulary: ControlledVocabulary,
    examples: Sequence[ReasoningExample],
    *,
    baseline: str,
    capacity: int,
    batch_size: int,
    device: torch.device | str,
    seed: int,
    limits: tuple[int, ...] = DEFAULT_DELAY_LIMITS,
) -> BaselineResult:
    """Evaluate one fixed-memory policy through the cached streaming path."""
    if not isinstance(model, DecoderOnlyTransformer):
        raise TypeError("model must be a DecoderOnlyTransformer")
    if not isinstance(vocabulary, ControlledVocabulary):
        raise TypeError("vocabulary must be a ControlledVocabulary")
    if not examples:
        raise ValueError("examples must be nonempty")
    for name, value in (
        ("capacity", capacity),
        ("batch_size", batch_size),
        ("seed", seed),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if capacity <= 0 or batch_size <= 0 or seed < 0:
        raise ValueError("capacity and batch_size must be positive and seed nonnegative")
    if baseline not in BASELINE_NAMES:
        supported = ", ".join(BASELINE_NAMES)
        raise ValueError(f"unsupported baseline {baseline!r}; choose from: {supported}")

    prepared = _prepare_examples(examples, vocabulary)
    if baseline == "oracle":
        required_capacity = max(len(item.evidence_positions) for item in prepared)
        if capacity < required_capacity:
            raise ValueError(
                f"oracle capacity must be at least {required_capacity} for exact evidence"
            )

    generator = torch.Generator(device=torch.device(device)).manual_seed(seed)
    was_training = model.training
    model.eval()
    correct = 0
    outside_window_correct = 0
    outside_window_count = 0
    delay_counts: dict[str, list[int]] = {}
    observed_memory_bytes: set[int] = set()
    try:
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start : start + batch_size]
            max_length = max(len(item.prompt_ids) for item in batch)
            input_ids = torch.full(
                (len(batch), max_length),
                vocabulary.token_to_id["<pad>"],
                dtype=torch.long,
                device=device,
            )
            prompt_lengths = []
            for row, item in enumerate(batch):
                input_ids[row, : len(item.prompt_ids)] = torch.tensor(
                    item.prompt_ids,
                    dtype=torch.long,
                    device=device,
                )
                prompt_lengths.append(len(item.prompt_ids))

            target_positions = _target_position_tensor(batch, device=device)
            policy = _make_policy(baseline, capacity, target_positions)
            stream = StreamingDecoder(
                model,
                segment_length=1,
                memory_policy=policy,
                generator=generator,
            )
            predictions = [-1] * len(batch)
            for position in range(max_length):
                logits = stream.process_segment(
                    input_ids[:, position : position + 1]
                )
                for row, prompt_length in enumerate(prompt_lengths):
                    if position + 1 == prompt_length:
                        predictions[row] = int(logits[row, 0].argmax().cpu())

            if any(prediction < 0 for prediction in predictions):
                raise RuntimeError("failed to capture every streamed answer prediction")
            if stream.memory_bytes % len(batch) != 0:
                raise RuntimeError("memory byte count is not divisible by batch size")
            observed_memory_bytes.add(stream.memory_bytes // len(batch))

            for prediction, item in zip(predictions, batch, strict=True):
                is_correct = int(prediction == item.answer_id)
                correct += is_correct
                if item.delay > model.config.max_local_tokens:
                    outside_window_correct += is_correct
                    outside_window_count += 1
                label = delay_label(item.delay, limits)
                totals = delay_counts.setdefault(label, [0, 0])
                totals[0] += is_correct
                totals[1] += 1
    finally:
        model.train(was_training)

    if len(observed_memory_bytes) != 1:
        raise RuntimeError("per-example memory allocation changed across batches")
    memory_bytes = observed_memory_bytes.pop()
    return BaselineResult(
        baseline=baseline,
        capacity=0 if baseline == "local" else capacity,
        memory_bytes=memory_bytes,
        correct=correct,
        count=len(prepared),
        outside_window_correct=outside_window_correct,
        outside_window_count=outside_window_count,
        curve=_build_curve(delay_counts, limits),
    )


def require_equal_memory_budget(results: Sequence[BaselineResult]) -> int:
    """Return the shared nonzero byte budget or reject an unfair comparison."""
    if not results:
        raise ValueError("results must be nonempty")
    budgets = {
        result.memory_bytes
        for result in results
        if result.baseline != "local"
    }
    if not budgets:
        raise ValueError("results must contain at least one memory baseline")
    if len(budgets) != 1:
        raise ValueError("memory baselines do not use the same byte budget")
    return budgets.pop()
