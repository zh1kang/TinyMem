"""Oracle diagnostics for locating LongMemEval transfer failures."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from math import exp

import torch
from torch.nn import functional as F

from tinymem.data.longmemeval import LongMemEvalExample, LongMemSession
from tinymem.evaluation.longmemeval import (
    LongMemEvalPromptState,
    answer_token_f1,
    format_longmemeval_prompt,
    generate_longmemeval_from_state,
    normalized_answer,
    stream_longmemeval_prompt,
)
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.tokenization.byte_tokenizer import ByteTokenizer


DIAGNOSTIC_CONDITIONS = (
    "full_oracle",
    "answer_messages",
    "query_only",
    "answer_copy",
)


@dataclass(frozen=True)
class CandidateAnswerScore:
    """Store teacher-forced likelihood and prefix-copy behavior."""

    total_nll: float
    byte_count: int
    first_byte_correct: bool
    greedy_prefix_bytes: int

    @property
    def mean_nll(self) -> float:
        return self.total_nll / self.byte_count

    @property
    def byte_perplexity(self) -> float:
        return exp(self.mean_nll)

    @property
    def greedy_prefix_fraction(self) -> float:
        return self.greedy_prefix_bytes / self.byte_count

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            **asdict(self),
            "mean_nll": self.mean_nll,
            "byte_perplexity": self.byte_perplexity,
            "greedy_prefix_fraction": self.greedy_prefix_fraction,
        }


@dataclass(frozen=True)
class LongMemEvalDiagnosticPrediction:
    """Store all measurements for one example and diagnostic condition."""

    question_id: str
    question_type: str
    prediction: str
    reference: str
    counterfactual: str
    exact_match: bool
    token_f1: float
    prompt_bytes: int
    valid_memory_slots: int
    answer_score: CandidateAnswerScore
    counterfactual_score: CandidateAnswerScore

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["answer_score"] = self.answer_score.to_dict()
        result["counterfactual_score"] = self.counterfactual_score.to_dict()
        return result


@dataclass(frozen=True)
class LongMemEvalDiagnosticResult:
    """Aggregate one oracle diagnostic condition."""

    condition: str
    count: int
    generated_exact_accuracy: float
    generated_mean_token_f1: float
    answer_byte_nll: float
    answer_byte_perplexity: float
    counterfactual_byte_nll: float
    mean_nll_margin: float
    answer_preference_rate: float
    first_byte_accuracy: float
    mean_greedy_prefix_fraction: float
    mean_prompt_bytes: float
    mean_valid_memory_slots: float
    uses_oracle_evidence: bool
    uses_answer_message_labels: bool
    reference_inserted: bool
    predictions: tuple[LongMemEvalDiagnosticPrediction, ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["predictions"] = [
            prediction.to_dict() for prediction in self.predictions
        ]
        return result


def format_longmemeval_diagnostic_prompt(
    example: LongMemEvalExample,
    condition: str,
) -> str:
    """Build one prompt with an explicit oracle-information boundary."""
    if not isinstance(example, LongMemEvalExample):
        raise TypeError("example must be a LongMemEvalExample")
    if not isinstance(condition, str):
        raise TypeError("condition must be a string")
    if condition not in DIAGNOSTIC_CONDITIONS:
        raise ValueError(f"unsupported diagnostic condition: {condition!r}")
    if condition == "full_oracle":
        return format_longmemeval_prompt(example)
    if condition == "query_only":
        return format_longmemeval_prompt(replace(example, sessions=()))
    if condition == "answer_copy":
        return (
            "[diagnostic copy task]\n"
            "User: Repeat the text exactly.\n"
            f"Text: {example.answer}\n"
            "Assistant:"
        )

    sessions = tuple(
        LongMemSession(
            session.session_id,
            session.date,
            tuple(message for message in session.messages if message.has_answer),
        )
        for session in example.sessions
        if any(message.has_answer for message in session.messages)
    )
    if not sessions:
        raise ValueError("answer_messages requires an answer-bearing message")
    return format_longmemeval_prompt(replace(example, sessions=sessions))


@torch.no_grad()
def score_candidate_answer(
    decoder: SegmentedContinuousDecoder,
    tokenizer: ByteTokenizer,
    state: LongMemEvalPromptState,
    candidate: str,
    *,
    device: torch.device | str,
) -> CandidateAnswerScore:
    """Score candidate bytes causally from a frozen prompt state."""
    if not isinstance(decoder, SegmentedContinuousDecoder):
        raise TypeError("decoder must be a SegmentedContinuousDecoder")
    if not isinstance(tokenizer, ByteTokenizer):
        raise TypeError("tokenizer must be a ByteTokenizer")
    if not isinstance(state, LongMemEvalPromptState):
        raise TypeError("state must be a LongMemEvalPromptState")
    if not isinstance(candidate, str):
        raise TypeError("candidate must be a string")
    candidate_ids = tokenizer.encode(candidate)
    if not candidate_ids:
        raise ValueError("candidate must contain at least one UTF-8 byte")

    targets = torch.tensor(
        candidate_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    logits = state.next_logits.unsqueeze(1)
    if targets.shape[1] > 1:
        prefix = targets[:, :-1]
        output = decoder(
            prefix,
            torch.ones_like(prefix, dtype=torch.bool),
            initial_memory=state.memory,
            position_offset=state.position,
            update_memory=False,
        )
        logits = torch.cat((logits, output.logits[:, : prefix.shape[1]]), dim=1)

    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    )
    matches = logits.argmax(dim=-1).eq(targets).squeeze(0)
    mismatch_indices = (~matches).nonzero(as_tuple=False)
    greedy_prefix_bytes = (
        int(mismatch_indices[0, 0])
        if mismatch_indices.numel()
        else targets.shape[1]
    )
    return CandidateAnswerScore(
        total_nll=float(losses.sum()),
        byte_count=targets.shape[1],
        first_byte_correct=bool(matches[0]),
        greedy_prefix_bytes=greedy_prefix_bytes,
    )


def _counterfactual_answers(
    examples: Sequence[LongMemEvalExample],
) -> tuple[str, ...]:
    answers = tuple(str(example.answer) for example in examples)
    counterfactuals = []
    for index, answer in enumerate(answers):
        normalized = normalized_answer(answer)
        for offset in range(1, len(answers)):
            candidate = answers[(index + offset) % len(answers)]
            if normalized_answer(candidate) != normalized:
                counterfactuals.append(candidate)
                break
        else:
            raise ValueError("diagnostics require at least two distinct answers")
    return tuple(counterfactuals)


def _aggregate_diagnostic(
    condition: str,
    predictions: Sequence[LongMemEvalDiagnosticPrediction],
) -> LongMemEvalDiagnosticResult:
    answer_total_nll = sum(
        prediction.answer_score.total_nll for prediction in predictions
    )
    answer_byte_count = sum(
        prediction.answer_score.byte_count for prediction in predictions
    )
    counterfactual_total_nll = sum(
        prediction.counterfactual_score.total_nll for prediction in predictions
    )
    counterfactual_byte_count = sum(
        prediction.counterfactual_score.byte_count for prediction in predictions
    )
    answer_byte_nll = answer_total_nll / answer_byte_count
    count = len(predictions)
    return LongMemEvalDiagnosticResult(
        condition=condition,
        count=count,
        generated_exact_accuracy=sum(
            prediction.exact_match for prediction in predictions
        )
        / count,
        generated_mean_token_f1=sum(
            prediction.token_f1 for prediction in predictions
        )
        / count,
        answer_byte_nll=answer_byte_nll,
        answer_byte_perplexity=exp(answer_byte_nll),
        counterfactual_byte_nll=(
            counterfactual_total_nll / counterfactual_byte_count
        ),
        mean_nll_margin=sum(
            prediction.counterfactual_score.mean_nll
            - prediction.answer_score.mean_nll
            for prediction in predictions
        )
        / count,
        answer_preference_rate=sum(
            prediction.answer_score.mean_nll
            < prediction.counterfactual_score.mean_nll
            for prediction in predictions
        )
        / count,
        first_byte_accuracy=sum(
            prediction.answer_score.first_byte_correct
            for prediction in predictions
        )
        / count,
        mean_greedy_prefix_fraction=sum(
            prediction.answer_score.greedy_prefix_fraction
            for prediction in predictions
        )
        / count,
        mean_prompt_bytes=sum(
            prediction.prompt_bytes for prediction in predictions
        )
        / count,
        mean_valid_memory_slots=sum(
            prediction.valid_memory_slots for prediction in predictions
        )
        / count,
        uses_oracle_evidence=condition in {"full_oracle", "answer_messages"},
        uses_answer_message_labels=condition == "answer_messages",
        reference_inserted=condition == "answer_copy",
        predictions=tuple(predictions),
    )


@torch.no_grad()
def evaluate_longmemeval_diagnostics(
    decoder: SegmentedContinuousDecoder,
    examples: Sequence[LongMemEvalExample],
    *,
    device: torch.device | str,
    max_new_tokens: int,
    chunk_tokens: int,
    conditions: Sequence[str] = DIAGNOSTIC_CONDITIONS,
) -> tuple[LongMemEvalDiagnosticResult, ...]:
    """Run oracle, no-context, and copy controls without training the model."""
    if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)):
        raise TypeError("examples must be a sequence")
    if len(examples) < 2:
        raise ValueError("diagnostics require at least two examples")
    if not all(isinstance(example, LongMemEvalExample) for example in examples):
        raise TypeError("examples must contain LongMemEvalExample values")
    if not isinstance(conditions, Sequence) or isinstance(conditions, (str, bytes)):
        raise TypeError("conditions must be a sequence")
    if not conditions:
        raise ValueError("conditions must be nonempty")
    ordered_conditions = tuple(dict.fromkeys(conditions))
    for condition in ordered_conditions:
        if condition not in DIAGNOSTIC_CONDITIONS:
            raise ValueError(f"unsupported diagnostic condition: {condition!r}")

    tokenizer = ByteTokenizer()
    counterfactuals = _counterfactual_answers(examples)
    was_training = decoder.training
    decoder.eval()
    results = []
    try:
        for condition in ordered_conditions:
            predictions = []
            for example, counterfactual in zip(
                examples,
                counterfactuals,
                strict=True,
            ):
                prompt = format_longmemeval_diagnostic_prompt(example, condition)
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
                reference = str(example.answer)
                predictions.append(
                    LongMemEvalDiagnosticPrediction(
                        question_id=example.question_id,
                        question_type=example.question_type,
                        prediction=prediction,
                        reference=reference,
                        counterfactual=counterfactual,
                        exact_match=(
                            normalized_answer(prediction)
                            == normalized_answer(reference)
                        ),
                        token_f1=answer_token_f1(prediction, reference),
                        prompt_bytes=state.context_bytes,
                        valid_memory_slots=int(state.memory.valid.sum()),
                        answer_score=score_candidate_answer(
                            decoder,
                            tokenizer,
                            state,
                            reference,
                            device=device,
                        ),
                        counterfactual_score=score_candidate_answer(
                            decoder,
                            tokenizer,
                            state,
                            counterfactual,
                            device=device,
                        ),
                    )
                )
            results.append(_aggregate_diagnostic(condition, predictions))
    finally:
        decoder.train(was_training)
    return tuple(results)
