"""Generated-answer evaluation for byte-level controlled conversations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import torch

from tinymem.evaluation.longmemeval import (
    answer_token_f1,
    generate_longmemeval_from_state,
    normalized_answer,
    stream_longmemeval_prompt,
)
from tinymem.evaluation.longmemeval_diagnostics import score_candidate_answer
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.tokenization.byte_tokenizer import ByteTokenizer
from tinymem.training.conversational_qa import ByteQAExample


@dataclass(frozen=True)
class ConversationalQAPrediction:
    """Store one controlled generated answer and causal answer loss."""

    source_example_id: str
    dataset: str
    task_id: str
    prediction: str
    reference: str
    exact_match: bool
    token_f1: float
    answer_total_nll: float
    answer_byte_count: int
    first_byte_correct: bool
    prompt_bytes: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ConversationalQACategoryResult:
    """Aggregate generated accuracy and token-weighted answer loss."""

    count: int
    exact_accuracy: float
    mean_token_f1: float
    answer_byte_nll: float
    first_byte_accuracy: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class ConversationalQAEvaluation:
    """Hold overall, per-task, and per-example bridge metrics."""

    overall: ConversationalQACategoryResult
    by_task: dict[str, ConversationalQACategoryResult]
    predictions: tuple[ConversationalQAPrediction, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "overall": self.overall.to_dict(),
            "by_task": {
                name: result.to_dict() for name, result in self.by_task.items()
            },
            "predictions": [prediction.to_dict() for prediction in self.predictions],
        }


def _aggregate(
    predictions: Sequence[ConversationalQAPrediction],
) -> ConversationalQACategoryResult:
    if not predictions:
        raise ValueError("predictions must be nonempty")
    count = len(predictions)
    total_bytes = sum(prediction.answer_byte_count for prediction in predictions)
    return ConversationalQACategoryResult(
        count=count,
        exact_accuracy=sum(prediction.exact_match for prediction in predictions)
        / count,
        mean_token_f1=sum(prediction.token_f1 for prediction in predictions) / count,
        answer_byte_nll=sum(
            prediction.answer_total_nll for prediction in predictions
        )
        / total_bytes,
        first_byte_accuracy=sum(
            prediction.first_byte_correct for prediction in predictions
        )
        / count,
    )


def evaluate_conversational_qa(
    decoder: SegmentedContinuousDecoder,
    examples: Sequence[ByteQAExample],
    *,
    device: torch.device | str,
    max_new_tokens: int,
    chunk_tokens: int,
) -> ConversationalQAEvaluation:
    """Generate controlled answers through the external prompt boundary."""
    if not isinstance(decoder, SegmentedContinuousDecoder):
        raise TypeError("decoder must be a SegmentedContinuousDecoder")
    if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)):
        raise TypeError("examples must be a sequence")
    if not examples or not all(
        isinstance(example, ByteQAExample) for example in examples
    ):
        raise ValueError("examples must contain ByteQAExample values")

    tokenizer = ByteTokenizer()
    was_training = decoder.training
    decoder.eval()
    predictions = []
    try:
        for example in examples:
            prompt = tokenizer.decode(example.prompt_ids)
            reference = tokenizer.decode(example.answer_ids)
            state = stream_longmemeval_prompt(
                decoder,
                tokenizer,
                prompt,
                device=device,
                chunk_tokens=chunk_tokens,
            )
            prediction = generate_longmemeval_from_state(
                decoder,
                tokenizer,
                state,
                device=device,
                max_new_tokens=max_new_tokens,
            )
            answer_score = score_candidate_answer(
                decoder,
                tokenizer,
                state,
                reference,
                device=device,
            )
            predictions.append(
                ConversationalQAPrediction(
                    source_example_id=example.source_example_id,
                    dataset=example.dataset,
                    task_id=example.task_id,
                    prediction=prediction,
                    reference=reference,
                    exact_match=(
                        normalized_answer(prediction)
                        == normalized_answer(reference)
                    ),
                    token_f1=answer_token_f1(prediction, reference),
                    answer_total_nll=answer_score.total_nll,
                    answer_byte_count=answer_score.byte_count,
                    first_byte_correct=answer_score.first_byte_correct,
                    prompt_bytes=state.context_bytes,
                )
            )
    finally:
        decoder.train(was_training)

    grouped: dict[str, list[ConversationalQAPrediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[f"{prediction.dataset}:{prediction.task_id}"].append(prediction)
    return ConversationalQAEvaluation(
        overall=_aggregate(predictions),
        by_task={
            name: _aggregate(group) for name, group in sorted(grouped.items())
        },
        predictions=tuple(predictions),
    )
