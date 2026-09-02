"""Matched write-policy evaluation for adaptive recurrent memory."""

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import torch

from tinymem.evaluation.continuous_memory import (
    PairedAccuracyResult,
    WriteDecisionResult,
    evaluate_continuous_answers,
    paired_accuracy_test,
)
from tinymem.memory.controller import AdaptiveWriteController, WRITE_ACTION
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.training.continuous import collate_segmented_answer_supervision
from tinymem.training.controlled_qa import EncodedQAExample


@dataclass(frozen=True)
class ControllerEventTrace:
    """Store inspectable write signals for one encoded example."""

    source_example_id: str
    surprises: tuple[float, ...]
    write_probabilities: tuple[float, ...]
    learned_writes: tuple[bool, ...]
    relevant_segments: tuple[bool, ...]
    event_types: tuple[tuple[str, ...], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "source_example_id": self.source_example_id,
            "events": [
                {
                    "segment": index,
                    "surprise": surprise,
                    "write_probability": probability,
                    "learned_write": write,
                    "relevant": relevant,
                    "event_types": list(event_types),
                }
                for index, (
                    surprise,
                    probability,
                    write,
                    relevant,
                    event_types,
                ) in enumerate(
                    zip(
                        self.surprises,
                        self.write_probabilities,
                        self.learned_writes,
                        self.relevant_segments,
                        self.event_types,
                        strict=True,
                    )
                )
            ],
        }


@dataclass(frozen=True)
class ControllerPolicyResult:
    """Store answer quality and write cost for one fixed policy."""

    policy: str
    correct: int
    count: int
    correctness: tuple[bool, ...]
    writes: int
    valid_segments: int
    tokens: int
    decisions: WriteDecisionResult

    @property
    def accuracy(self) -> float:
        return self.correct / self.count

    @property
    def write_rate(self) -> float:
        return self.writes / self.valid_segments

    @property
    def writes_per_1000_tokens(self) -> float:
        return 1000 * self.writes / self.tokens

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "correct": self.correct,
            "count": self.count,
            "accuracy": self.accuracy,
            "writes": self.writes,
            "valid_segments": self.valid_segments,
            "write_rate": self.write_rate,
            "tokens": self.tokens,
            "writes_per_1000_tokens": self.writes_per_1000_tokens,
            "decisions": self.decisions.to_dict(),
        }


@dataclass(frozen=True)
class ControllerComparison:
    """Store matched policy results and milestone exit diagnostics."""

    policies: tuple[ControllerPolicyResult, ...]
    learned_vs_random: PairedAccuracyResult
    relevant_write_rate: float
    background_write_rate: float
    event_write_rates: dict[str, float]
    event_counts: dict[str, int]
    surprise_threshold: float | None
    traces: tuple[ControllerEventTrace, ...]

    @property
    def learned_beats_random(self) -> bool:
        indexed = {result.policy: result for result in self.policies}
        return indexed["learned"].accuracy > indexed["random_matched"].accuracy

    @property
    def responds_to_relevance(self) -> bool:
        return self.relevant_write_rate > self.background_write_rate

    @property
    def responds_to_corrections(self) -> bool:
        return (
            self.event_counts.get("correction", 0) > 0
            and self.event_write_rates["correction"] > self.background_write_rate
        )

    @property
    def exit_criteria_met(self) -> bool:
        return (
            self.learned_beats_random
            and self.responds_to_relevance
            and self.responds_to_corrections
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "policies": [result.to_dict() for result in self.policies],
            "learned_vs_random": self.learned_vs_random.to_dict(),
            "relevant_write_rate": self.relevant_write_rate,
            "background_write_rate": self.background_write_rate,
            "event_write_rates": self.event_write_rates,
            "event_counts": self.event_counts,
            "surprise_threshold": self.surprise_threshold,
            "learned_beats_random": self.learned_beats_random,
            "responds_to_relevance": self.responds_to_relevance,
            "responds_to_corrections": self.responds_to_corrections,
            "exit_criteria_met": self.exit_criteria_met,
        }


@dataclass(frozen=True)
class _ControllerSignals:
    learned: torch.Tensor
    surprise: torch.Tensor
    valid: torch.Tensor
    relevant: torch.Tensor
    traces: tuple[ControllerEventTrace, ...]


def periodic_write_mask(valid: torch.Tensor, *, interval: int) -> torch.Tensor:
    """Write every fixed number of valid segments in each stream."""
    _validate_valid_mask(valid)
    if isinstance(interval, bool) or not isinstance(interval, Integral):
        raise TypeError("interval must be an integer")
    if interval <= 0:
        raise ValueError("interval must be positive")
    positions = torch.arange(valid.shape[1]).unsqueeze(0)
    return valid & ((positions + 1) % int(interval) == 0)


def matched_random_write_mask(
    valid: torch.Tensor,
    *,
    writes: int,
    seed: int,
) -> torch.Tensor:
    """Select an exact random write count from valid segments."""
    _validate_valid_mask(valid)
    for name, value in (("writes", writes), ("seed", seed)):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    valid_count = int(valid.sum())
    if not 0 <= writes <= valid_count:
        raise ValueError("writes must be between zero and the valid segment count")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    output = torch.zeros_like(valid)
    if writes == 0:
        return output
    positions = valid.nonzero(as_tuple=False)
    generator = torch.Generator().manual_seed(int(seed))
    selected = positions[
        torch.randperm(valid_count, generator=generator)[: int(writes)]
    ]
    output[selected[:, 0], selected[:, 1]] = True
    return output


def matched_surprise_write_mask(
    surprise: torch.Tensor,
    valid: torch.Tensor,
    *,
    writes: int,
) -> torch.Tensor:
    """Write the exact matched count of highest-surprise segments."""
    _validate_valid_mask(valid)
    if not isinstance(surprise, torch.Tensor):
        raise TypeError("surprise must be a torch.Tensor")
    if surprise.shape != valid.shape:
        raise ValueError(f"surprise must have shape {valid.shape}")
    if not surprise.is_floating_point():
        raise TypeError("surprise must be floating point")
    if isinstance(writes, bool) or not isinstance(writes, Integral):
        raise TypeError("writes must be an integer")
    valid_count = int(valid.sum())
    if not 0 <= writes <= valid_count:
        raise ValueError("writes must be between zero and the valid segment count")
    output = torch.zeros_like(valid)
    if writes == 0:
        return output
    scores = surprise.masked_fill(~valid, float("-inf")).flatten()
    selected = scores.topk(int(writes)).indices
    output.flatten()[selected] = True
    return output


def _validate_valid_mask(valid: torch.Tensor) -> None:
    if not isinstance(valid, torch.Tensor):
        raise TypeError("valid must be a torch.Tensor")
    if valid.ndim != 2 or valid.numel() == 0:
        raise ValueError("valid must have nonempty shape [examples, segments]")
    if valid.dtype != torch.bool:
        raise TypeError("valid must be a boolean tensor")
    if valid.device.type != "cpu":
        raise ValueError("policy masks must be built on CPU")


@torch.no_grad()
def _collect_controller_signals(
    decoder: SegmentedContinuousDecoder,
    examples: Sequence[EncodedQAExample],
    *,
    batch_size: int,
    pad_id: int,
    device: torch.device | str,
) -> _ControllerSignals:
    if not isinstance(decoder.write_controller, AdaptiveWriteController):
        raise ValueError("controller comparison requires an adaptive controller")
    max_segments = max(
        (len(example.input_ids) + decoder.segment_length - 1)
        // decoder.segment_length
        for example in examples
    )
    shape = (len(examples), max_segments)
    learned = torch.zeros(shape, dtype=torch.bool)
    surprise = torch.zeros(shape)
    valid = torch.zeros(shape, dtype=torch.bool)
    relevant = torch.zeros(shape, dtype=torch.bool)
    traces = []
    was_training = decoder.training
    decoder.eval()
    try:
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            input_ids, _, token_valid = collate_segmented_answer_supervision(
                batch,
                pad_id=pad_id,
                device=device,
            )
            output = decoder(input_ids, token_valid)
            if (
                output.controller_probabilities is None
                or output.controller_surprise is None
                or output.controller_valid is None
            ):
                raise RuntimeError("adaptive decoder did not return controller signals")
            for row, example in enumerate(batch):
                targets = example.segment_write_targets
                if targets is None:
                    raise ValueError(
                        "controller comparison requires write targets"
                    )
                count = (
                    len(example.input_ids) + decoder.segment_length - 1
                ) // decoder.segment_length
                if len(targets) != count:
                    raise ValueError(
                        "segment write targets must match the encoded sequence"
                    )
                destination = start + row
                row_writes = output.writes_applied[row, :count].cpu()
                row_surprise = output.controller_surprise[row, :count].cpu()
                row_probabilities = output.controller_probabilities[
                    row,
                    :count,
                    WRITE_ACTION,
                ].cpu()
                row_valid = output.controller_valid[row, :count].cpu()
                target_tensor = torch.tensor(targets, dtype=torch.bool)
                event_types = example.segment_event_types
                if event_types is None:
                    event_types = tuple(
                        ("relevant_fact",) if target else ("background",)
                        for target in targets
                    )
                if len(event_types) != count:
                    raise ValueError(
                        "segment event types must match the encoded sequence"
                    )
                learned[destination, :count] = row_writes
                surprise[destination, :count] = row_surprise
                valid[destination, :count] = row_valid
                relevant[destination, :count] = target_tensor
                traces.append(
                    ControllerEventTrace(
                        source_example_id=example.source_example_id,
                        surprises=tuple(float(value) for value in row_surprise),
                        write_probabilities=tuple(
                            float(value) for value in row_probabilities
                        ),
                        learned_writes=tuple(bool(value) for value in row_writes),
                        relevant_segments=tuple(
                            bool(value) for value in target_tensor
                        ),
                        event_types=event_types,
                    )
                )
    finally:
        decoder.train(was_training)
    return _ControllerSignals(
        learned=learned,
        surprise=surprise,
        valid=valid,
        relevant=relevant,
        traces=tuple(traces),
    )


def _policy_result(
    decoder: SegmentedContinuousDecoder,
    examples: Sequence[EncodedQAExample],
    mask: torch.Tensor,
    *,
    policy: str,
    batch_size: int,
    pad_id: int,
    device: torch.device | str,
) -> ControllerPolicyResult:
    evaluated = evaluate_continuous_answers(
        decoder,
        examples,
        batch_size=batch_size,
        pad_id=pad_id,
        device=device,
        intervention_name=policy,
        forced_writes=mask,
    )
    if evaluated.writes is None:
        raise RuntimeError("policy evaluation did not return write diagnostics")
    return ControllerPolicyResult(
        policy=policy,
        correct=evaluated.correct,
        count=evaluated.count,
        correctness=evaluated.correctness,
        writes=int(mask.sum()),
        valid_segments=sum(
            (len(example.input_ids) + decoder.segment_length - 1)
            // decoder.segment_length
            for example in examples
        ),
        tokens=sum(len(example.input_ids) for example in examples),
        decisions=evaluated.writes,
    )


def evaluate_controller_policies(
    decoder: SegmentedContinuousDecoder,
    examples: Sequence[EncodedQAExample],
    *,
    batch_size: int,
    pad_id: int,
    device: torch.device | str,
    periodic_interval: int,
    random_seed: int,
) -> ControllerComparison:
    """Compare all required binary write policies on one frozen decoder."""
    if not isinstance(decoder, SegmentedContinuousDecoder):
        raise TypeError("decoder must be a SegmentedContinuousDecoder")
    if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)):
        raise TypeError("examples must be a sequence")
    if not examples or not all(
        isinstance(example, EncodedQAExample) for example in examples
    ):
        raise ValueError("examples must contain EncodedQAExample values")
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    signals = _collect_controller_signals(
        decoder,
        examples,
        batch_size=int(batch_size),
        pad_id=pad_id,
        device=device,
    )
    learned_writes = int(signals.learned.sum())
    surprise_mask = matched_surprise_write_mask(
        signals.surprise,
        signals.valid,
        writes=learned_writes,
    )
    surprise_threshold = (
        float(signals.surprise[surprise_mask].min())
        if learned_writes
        else None
    )
    masks = (
        ("periodic", periodic_write_mask(
            signals.valid,
            interval=periodic_interval,
        )),
        ("random_matched", matched_random_write_mask(
            signals.valid,
            writes=learned_writes,
            seed=random_seed,
        )),
        ("surprise_threshold", surprise_mask),
        ("learned", signals.learned),
        ("oracle", signals.relevant & signals.valid),
    )
    policies = tuple(
        _policy_result(
            decoder,
            examples,
            mask,
            policy=name,
            batch_size=int(batch_size),
            pad_id=pad_id,
            device=device,
        )
        for name, mask in masks
    )
    indexed = {result.policy: result for result in policies}
    relevant_count = int((signals.relevant & signals.valid).sum())
    background = ~signals.relevant & signals.valid
    background_count = int(background.sum())
    relevant_write_rate = (
        float((signals.learned & signals.relevant & signals.valid).sum())
        / relevant_count
        if relevant_count
        else 0.0
    )
    background_write_rate = (
        float((signals.learned & background).sum()) / background_count
        if background_count
        else 0.0
    )
    event_counts: dict[str, int] = {}
    event_writes: dict[str, int] = {}
    for trace in signals.traces:
        for wrote, labels in zip(
            trace.learned_writes,
            trace.event_types,
            strict=True,
        ):
            for label in labels:
                event_counts[label] = event_counts.get(label, 0) + 1
                event_writes[label] = event_writes.get(label, 0) + int(wrote)
    event_write_rates = {
        label: event_writes[label] / count
        for label, count in event_counts.items()
    }
    return ControllerComparison(
        policies=policies,
        learned_vs_random=paired_accuracy_test(
            indexed["learned"].correctness,
            indexed["random_matched"].correctness,
            alpha=0.05,
        ),
        relevant_write_rate=relevant_write_rate,
        background_write_rate=background_write_rate,
        event_write_rates=event_write_rates,
        event_counts=event_counts,
        surprise_threshold=surprise_threshold,
        traces=signals.traces,
    )
