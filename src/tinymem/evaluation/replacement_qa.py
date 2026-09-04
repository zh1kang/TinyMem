"""Causal evaluation for content-aware correction replacement."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from numbers import Integral

import torch
from torch.nn import functional as F

from tinymem.data.replacement_qa import ReplacementQAExample, replacement_history_id
from tinymem.evaluation.continuous_memory import (
    drop_memory,
    zero_memory,
)
from tinymem.memory.replacement import (
    SlotReplacementController,
    replace_memory_slot,
)
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.memory_input import AttentionMemory
from tinymem.tokenization.byte_tokenizer import ByteTokenizer
from tinymem.training.replacement_qa import (
    build_replacement_memory,
    collate_replacement_query,
)


REPLACEMENT_CONDITIONS = (
    "normal",
    "oracle_slot",
    "fifo_slot",
    "wrong_slot",
    "frozen",
    "drop",
    "zero",
    "shuffle",
)


@dataclass(frozen=True)
class ReplacementQAPrediction:
    source_example_id: str
    prediction: str
    reference: str
    exact_match: bool
    first_byte_correct: bool
    query_requires_correction: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ReplacementQAConditionResult:
    condition: str
    count: int
    exact_accuracy: float
    corrected_query_accuracy: float
    unchanged_query_accuracy: float
    first_byte_accuracy: float
    answer_byte_nll: float
    mean_first_byte_logit_linf_from_normal: float
    predictions: tuple[ReplacementQAPrediction, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "predictions": [prediction.to_dict() for prediction in self.predictions],
        }


@dataclass(frozen=True)
class ReplacementQAEvaluation:
    replacement_slot_accuracy: float
    predicted_slots: tuple[int, ...]
    target_slots: tuple[int, ...]
    conditions: dict[str, ReplacementQAConditionResult]

    def to_dict(self) -> dict[str, object]:
        return {
            "replacement_slot_accuracy": self.replacement_slot_accuracy,
            "predicted_slots": self.predicted_slots,
            "target_slots": self.target_slots,
            "conditions": {
                name: result.to_dict() for name, result in self.conditions.items()
            },
        }


def _forced_memory(
    initial: AttentionMemory,
    correction: torch.Tensor,
    correction_valid: torch.Tensor,
    correction_positions: torch.Tensor,
    slots: torch.Tensor,
) -> AttentionMemory:
    assignments = F.one_hot(
        slots.to(torch.long),
        num_classes=initial.slot_count,
    ).to(dtype=initial.values.dtype)
    return replace_memory_slot(
        initial,
        correction,
        correction_valid,
        correction_positions,
        assignments,
    )


def _row_memory(memory: AttentionMemory, index: int) -> AttentionMemory:
    return AttentionMemory(
        values=memory.values[index : index + 1],
        valid=memory.valid[index : index + 1],
        positions=memory.positions[index : index + 1],
    )


def mismatched_history_indices(examples: Sequence[ReplacementQAExample]) -> list[int]:
    """Derange history groups so alternate questions never keep their own memory."""
    histories = [replacement_history_id(example) for example in examples]
    if len(set(histories)) < 2:
        raise ValueError("memory shuffle requires at least two distinct histories")
    order = sorted(range(len(histories)), key=histories.__getitem__)
    largest_group = max(histories.count(history) for history in set(histories))
    if largest_group * 2 > len(histories):
        raise ValueError("history groups are too imbalanced for a memory derangement")
    indices = [0] * len(order)
    for rank, source in enumerate(order):
        indices[source] = order[(rank + largest_group) % len(order)]
    return indices


def _score_and_generate(
    decoder: SegmentedContinuousDecoder,
    example: ReplacementQAExample,
    memory: AttentionMemory,
    *,
    device: torch.device | str,
    max_new_tokens: int,
) -> tuple[str, float, bool, torch.Tensor]:
    pad_id = ByteTokenizer.special_tokens["<pad>"]
    input_ids, target_ids, valid = collate_replacement_query(
        [example],
        pad_id=pad_id,
        device=device,
    )
    scored = decoder(
        input_ids,
        valid,
        initial_memory=memory,
        position_offset=example.query_position_offset,
        update_memory=False,
    )
    shifted_targets = target_ids[:, 1:]
    answer_nll = F.cross_entropy(
        scored.logits[:, :-1].reshape(-1, scored.logits.shape[-1]),
        shifted_targets.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    first_logits = scored.logits[0, len(example.query_ids) - 1].detach().cpu()
    first_byte_correct = int(first_logits.argmax()) == example.answer_ids[0]

    generated = []
    for _ in range(int(max_new_tokens)):
        prefix = example.query_ids + tuple(generated)
        prefix_ids = torch.tensor(prefix, dtype=torch.long, device=device).unsqueeze(0)
        generated_output = decoder(
            prefix_ids,
            torch.ones_like(prefix_ids, dtype=torch.bool),
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
    return prediction, float(answer_nll.cpu()), first_byte_correct, first_logits


@torch.no_grad()
def evaluate_replacement_qa(
    decoder: SegmentedContinuousDecoder,
    controller: SlotReplacementController,
    examples: Sequence[ReplacementQAExample],
    *,
    device: torch.device | str,
    max_new_tokens: int,
) -> ReplacementQAEvaluation:
    """Compare learned replacement with controlled slots and memory ablations."""
    if not isinstance(decoder, SegmentedContinuousDecoder):
        raise TypeError("decoder must be a SegmentedContinuousDecoder")
    if not isinstance(controller, SlotReplacementController):
        raise TypeError("controller must be a SlotReplacementController")
    if not examples or not all(
        isinstance(example, ReplacementQAExample) for example in examples
    ):
        raise ValueError("examples must contain ReplacementQAExample values")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, Integral):
        raise TypeError("max_new_tokens must be an integer")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if len(examples) < 2:
        raise ValueError("evaluation requires at least two examples for shuffling")
    if any(
        len(example.query_ids) + int(max_new_tokens) > decoder.segment_length
        for example in examples
    ):
        raise ValueError("max_new_tokens would exceed the query segment")
    correction_count = sum(example.query_requires_correction for example in examples)
    if correction_count in (0, len(examples)):
        raise ValueError("evaluation requires corrected and unchanged queries")

    was_decoder_training = decoder.training
    was_controller_training = controller.training
    decoder.eval()
    controller.eval()
    try:
        built = build_replacement_memory(
            decoder,
            controller,
            examples,
            pad_id=ByteTokenizer.special_tokens["<pad>"],
            device=device,
        )
        target_slots = torch.tensor(
            [example.correction_slot for example in examples],
            dtype=torch.long,
            device=device,
        )
        fifo_slots = torch.zeros_like(target_slots)
        wrong_slots = (target_slots + 1) % decoder.bank.capacity
        shuffled_indices = torch.tensor(
            mismatched_history_indices(examples), device=device, dtype=torch.long,
        )
        conditions = {
            "normal": built.corrected_memory,
            "oracle_slot": _forced_memory(
                built.initial_memory,
                built.correction_summary,
                built.correction_valid,
                built.correction_positions,
                target_slots,
            ),
            "fifo_slot": _forced_memory(
                built.initial_memory,
                built.correction_summary,
                built.correction_valid,
                built.correction_positions,
                fifo_slots,
            ),
            "wrong_slot": _forced_memory(
                built.initial_memory,
                built.correction_summary,
                built.correction_valid,
                built.correction_positions,
                wrong_slots,
            ),
            "frozen": built.initial_memory,
            "drop": drop_memory(built.corrected_memory),
            "zero": zero_memory(built.corrected_memory),
            "shuffle": AttentionMemory(
                values=built.corrected_memory.values.index_select(0, shuffled_indices),
                valid=built.corrected_memory.valid.index_select(0, shuffled_indices),
                positions=built.corrected_memory.positions.index_select(0, shuffled_indices),
            ),
        }
        raw: dict[
            str,
            tuple[
                tuple[ReplacementQAPrediction, ...],
                float,
                tuple[torch.Tensor, ...],
            ],
        ] = {}
        tokenizer = ByteTokenizer()
        for condition in REPLACEMENT_CONDITIONS:
            predictions = []
            total_nll = 0.0
            first_logits = []
            for index, example in enumerate(examples):
                prediction, nll, first_correct, logits = _score_and_generate(
                    decoder,
                    example,
                    _row_memory(conditions[condition], index),
                    device=device,
                    max_new_tokens=int(max_new_tokens),
                )
                reference = tokenizer.decode(example.answer_ids)
                predictions.append(
                    ReplacementQAPrediction(
                        source_example_id=example.source_example_id,
                        prediction=prediction,
                        reference=reference,
                        exact_match=prediction.casefold() == reference.casefold(),
                        first_byte_correct=first_correct,
                        query_requires_correction=example.query_requires_correction,
                    )
                )
                total_nll += nll
                first_logits.append(logits)
            raw[condition] = (tuple(predictions), total_nll, tuple(first_logits))

        normal_logits = raw["normal"][2]
        results = {}
        answer_byte_count = sum(len(example.answer_ids) + 1 for example in examples)
        for condition, (predictions, total_nll, first_logits) in raw.items():
            corrected = tuple(
                prediction
                for prediction in predictions
                if prediction.query_requires_correction
            )
            unchanged = tuple(
                prediction
                for prediction in predictions
                if not prediction.query_requires_correction
            )
            results[condition] = ReplacementQAConditionResult(
                condition=condition,
                count=len(predictions),
                exact_accuracy=sum(item.exact_match for item in predictions)
                / len(predictions),
                corrected_query_accuracy=sum(item.exact_match for item in corrected)
                / len(corrected),
                unchanged_query_accuracy=sum(item.exact_match for item in unchanged)
                / len(unchanged),
                first_byte_accuracy=sum(
                    item.first_byte_correct for item in predictions
                )
                / len(predictions),
                answer_byte_nll=total_nll / answer_byte_count,
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
        predicted_slots = built.controller_output.slots.cpu()
        target_slots_cpu = target_slots.cpu()
        return ReplacementQAEvaluation(
            replacement_slot_accuracy=float(
                (predicted_slots == target_slots_cpu).to(torch.float32).mean()
            ),
            predicted_slots=tuple(int(slot) for slot in predicted_slots),
            target_slots=tuple(int(slot) for slot in target_slots_cpu),
            conditions=results,
        )
    finally:
        decoder.train(was_decoder_training)
        controller.train(was_controller_training)
