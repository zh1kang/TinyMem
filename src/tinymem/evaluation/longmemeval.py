"""Frozen streaming evaluation for LongMemEval."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from numbers import Integral

import torch

from tinymem.data.longmemeval import LongMemEvalExample
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.memory_input import AttentionMemory
from tinymem.tokenization.byte_tokenizer import ByteTokenizer


MemoryIntervention = Callable[[AttentionMemory], AttentionMemory]
_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
_ABSTENTION_PHRASES = (
    "unknown",
    "cannot answer",
    "can t answer",
    "not enough information",
    "do not know",
    "don t know",
)


@dataclass(frozen=True)
class LongMemEvalPrediction:
    """Store one generated answer and its deterministic metrics."""

    question_id: str
    question_type: str
    prediction: str
    reference: str
    exact_match: bool
    token_f1: float
    abstained: bool
    context_bytes: int
    answer_session_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LongMemEvalCategoryResult:
    """Aggregate exact match, token F1, and selective coverage."""

    count: int
    exact_accuracy: float
    mean_token_f1: float
    coverage: float
    selective_accuracy: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class LongMemEvalResult:
    """Hold one frozen evaluation condition."""

    condition: str
    overall: LongMemEvalCategoryResult
    by_question_type: dict[str, LongMemEvalCategoryResult]
    predictions: tuple[LongMemEvalPrediction, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "condition": self.condition,
            "overall": self.overall.to_dict(),
            "by_question_type": {
                name: result.to_dict()
                for name, result in self.by_question_type.items()
            },
            "predictions": [prediction.to_dict() for prediction in self.predictions],
        }


@dataclass(frozen=True)
class LongMemEvalPromptState:
    """Hold the frozen state produced by one complete evaluation prompt."""

    memory: AttentionMemory
    position: int
    next_logits: torch.Tensor
    context_bytes: int
    continuation_memory: AttentionMemory
    continuation_position: int
    continuation_ids: tuple[int, ...]


def format_longmemeval_prompt(example: LongMemEvalExample) -> str:
    """Format sessions in source order without exposing the reference answer."""
    if not isinstance(example, LongMemEvalExample):
        raise TypeError("example must be a LongMemEvalExample")
    parts = []
    for session in example.sessions:
        parts.append(f"[session {session.session_id} | {session.date}]\n")
        for message in session.messages:
            role = "User" if message.role == "user" else "Assistant"
            parts.append(f"{role}: {message.content}\n")
    parts.append(f"[question | {example.question_date}]\n")
    parts.append(f"User: {example.question}\nAssistant:")
    return "".join(parts)


def normalized_answer(text: str) -> str:
    """Return a stable lowercase alphanumeric answer form."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return " ".join(_TOKEN_PATTERN.findall(text.casefold()))


def answer_token_f1(prediction: str, reference: str) -> float:
    """Compute bag-of-token F1 after deterministic normalization."""
    prediction_tokens = normalized_answer(prediction).split()
    reference_tokens = normalized_answer(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def is_abstention(prediction: str) -> bool:
    """Detect empty output and explicit inability-to-answer phrases."""
    normalized = normalized_answer(prediction)
    return not normalized or any(phrase in normalized for phrase in _ABSTENTION_PHRASES)


def _memory_from_output(
    values: torch.Tensor,
    valid: torch.Tensor,
    positions: torch.Tensor,
) -> AttentionMemory:
    return AttentionMemory(values=values, valid=valid, positions=positions)


@torch.no_grad()
def stream_longmemeval_prompt(
    decoder: SegmentedContinuousDecoder,
    tokenizer: ByteTokenizer,
    prompt: str,
    *,
    device: torch.device | str,
    chunk_tokens: int,
    query_memory_intervention: MemoryIntervention | None = None,
    update_memory: bool = True,
) -> LongMemEvalPromptState:
    """Stream one prompt and return its frozen memory and next-token state."""
    if not isinstance(decoder, SegmentedContinuousDecoder):
        raise TypeError("decoder must be a SegmentedContinuousDecoder")
    if not isinstance(tokenizer, ByteTokenizer):
        raise TypeError("tokenizer must be a ByteTokenizer")
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if not prompt:
        raise ValueError("prompt must be nonempty")
    if isinstance(chunk_tokens, bool) or not isinstance(chunk_tokens, Integral):
        raise TypeError("chunk_tokens must be an integer")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if chunk_tokens % decoder.segment_length != 0:
        raise ValueError("chunk_tokens must be a multiple of segment_length")
    if query_memory_intervention is not None and not callable(
        query_memory_intervention
    ):
        raise TypeError("query_memory_intervention must be callable or None")
    if not isinstance(update_memory, bool):
        raise TypeError("update_memory must be a boolean")

    prompt_ids = tokenizer.encode(prompt)
    memory = None
    position = 0
    next_logits = None
    final_segment_start = (
        (len(prompt_ids) - 1) // decoder.segment_length
    ) * decoder.segment_length
    for start in range(0, final_segment_start, int(chunk_tokens)):
        chunk = torch.tensor(
            prompt_ids[
                start : min(start + int(chunk_tokens), final_segment_start)
            ],
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        output = decoder(
            chunk,
            torch.ones_like(chunk, dtype=torch.bool),
            initial_memory=memory,
            position_offset=position,
            update_memory=update_memory,
        )
        memory = _memory_from_output(
            output.memory,
            output.memory_valid,
            output.memory_positions,
        )
        position += chunk.shape[1]
        next_logits = output.logits[:, -1]
    if memory is None:
        memory = AttentionMemory(
            values=torch.zeros(
                1,
                decoder.bank.capacity,
                decoder.model.config.d_model,
                dtype=decoder.model.token_embedding.weight.dtype,
                device=device,
            ),
            valid=torch.zeros(
                1,
                decoder.bank.capacity,
                dtype=torch.bool,
                device=device,
            ),
            positions=torch.full(
                (1, decoder.bank.capacity),
                -1,
                dtype=torch.long,
                device=device,
            ),
        )
    if query_memory_intervention is not None:
        memory = query_memory_intervention(memory)
        if not isinstance(memory, AttentionMemory):
            raise TypeError(
                "query_memory_intervention must return AttentionMemory"
            )
    continuation_memory = memory
    continuation_position = position
    continuation_ids = tuple(prompt_ids[final_segment_start:])
    final_chunk = torch.tensor(
        continuation_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    output = decoder(
        final_chunk,
        torch.ones_like(final_chunk, dtype=torch.bool),
        initial_memory=memory,
        position_offset=position,
        update_memory=update_memory,
    )
    memory = _memory_from_output(
        output.memory,
        output.memory_valid,
        output.memory_positions,
    )
    position += final_chunk.shape[1]
    next_logits = output.logits[:, -1]

    return LongMemEvalPromptState(
        memory=memory,
        position=position,
        next_logits=next_logits,
        context_bytes=len(prompt_ids),
        continuation_memory=continuation_memory,
        continuation_position=continuation_position,
        continuation_ids=continuation_ids,
    )


@torch.no_grad()
def generate_longmemeval_from_state(
    decoder: SegmentedContinuousDecoder,
    tokenizer: ByteTokenizer,
    state: LongMemEvalPromptState,
    *,
    device: torch.device | str,
    max_new_tokens: int,
) -> str:
    """Greedily generate an answer from a frozen prompt state."""
    if not isinstance(decoder, SegmentedContinuousDecoder):
        raise TypeError("decoder must be a SegmentedContinuousDecoder")
    if not isinstance(tokenizer, ByteTokenizer):
        raise TypeError("tokenizer must be a ByteTokenizer")
    if not isinstance(state, LongMemEvalPromptState):
        raise TypeError("state must be a LongMemEvalPromptState")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, Integral):
        raise TypeError("max_new_tokens must be an integer")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    next_logits = state.next_logits
    generated: list[int] = []
    for _ in range(int(max_new_tokens)):
        next_id = int(next_logits.argmax(dim=-1))
        if next_id in (10, tokenizer.special_tokens["<eos>"]):
            break
        if not 0 <= next_id <= 255:
            break
        generated.append(next_id)
        continuation = torch.tensor(
            state.continuation_ids + tuple(generated),
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        output = decoder(
            continuation,
            torch.ones_like(continuation, dtype=torch.bool),
            initial_memory=state.continuation_memory,
            position_offset=state.continuation_position,
        )
        next_logits = output.logits[:, -1]
    return bytes(generated).decode("utf-8", errors="replace").strip()


@torch.no_grad()
def generate_longmemeval_answer(
    decoder: SegmentedContinuousDecoder,
    tokenizer: ByteTokenizer,
    example: LongMemEvalExample,
    *,
    device: torch.device | str,
    max_new_tokens: int,
    chunk_tokens: int,
    query_memory_intervention: MemoryIntervention | None = None,
    update_memory: bool = True,
) -> tuple[str, int]:
    """Stream one ordered prompt and greedily generate a bounded answer."""
    state = stream_longmemeval_prompt(
        decoder,
        tokenizer,
        format_longmemeval_prompt(example),
        device=device,
        chunk_tokens=chunk_tokens,
        query_memory_intervention=query_memory_intervention,
        update_memory=update_memory,
    )
    prediction = generate_longmemeval_from_state(
        decoder,
        tokenizer,
        state,
        device=device,
        max_new_tokens=max_new_tokens,
    )
    return prediction, state.context_bytes


def _aggregate(
    predictions: Sequence[LongMemEvalPrediction],
) -> LongMemEvalCategoryResult:
    if not predictions:
        raise ValueError("predictions must be nonempty")
    covered = [prediction for prediction in predictions if not prediction.abstained]
    return LongMemEvalCategoryResult(
        count=len(predictions),
        exact_accuracy=sum(prediction.exact_match for prediction in predictions)
        / len(predictions),
        mean_token_f1=sum(prediction.token_f1 for prediction in predictions)
        / len(predictions),
        coverage=len(covered) / len(predictions),
        selective_accuracy=(
            sum(prediction.exact_match for prediction in covered) / len(covered)
            if covered
            else 0.0
        ),
    )


@torch.no_grad()
def evaluate_longmemeval(
    decoder: SegmentedContinuousDecoder,
    examples: Sequence[LongMemEvalExample],
    *,
    device: torch.device | str,
    max_new_tokens: int,
    chunk_tokens: int,
    condition: str = "normal",
    query_memory_intervention: MemoryIntervention | None = None,
    update_memory: bool = True,
) -> LongMemEvalResult:
    """Evaluate a frozen decoder without using answers in its prompts."""
    if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)):
        raise TypeError("examples must be a sequence")
    if not examples:
        raise ValueError("examples must be nonempty")
    if not all(isinstance(example, LongMemEvalExample) for example in examples):
        raise TypeError("examples must contain LongMemEvalExample values")
    if not isinstance(condition, str) or not condition:
        raise ValueError("condition must be a nonempty string")

    tokenizer = ByteTokenizer()
    was_training = decoder.training
    decoder.eval()
    predictions = []
    try:
        for example in examples:
            prediction, context_bytes = generate_longmemeval_answer(
                decoder,
                tokenizer,
                example,
                device=device,
                max_new_tokens=max_new_tokens,
                chunk_tokens=chunk_tokens,
                query_memory_intervention=query_memory_intervention,
                update_memory=update_memory,
            )
            reference = str(example.answer)
            predictions.append(
                LongMemEvalPrediction(
                    question_id=example.question_id,
                    question_type=example.question_type,
                    prediction=prediction,
                    reference=reference,
                    exact_match=(
                        normalized_answer(prediction)
                        == normalized_answer(reference)
                    ),
                    token_f1=answer_token_f1(prediction, reference),
                    abstained=is_abstention(prediction),
                    context_bytes=context_bytes,
                    answer_session_count=len(example.answer_session_ids),
                )
            )
    finally:
        decoder.train(was_training)

    grouped: dict[str, list[LongMemEvalPrediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction.question_type].append(prediction)
    return LongMemEvalResult(
        condition=condition,
        overall=_aggregate(predictions),
        by_question_type={
            name: _aggregate(group)
            for name, group in sorted(grouped.items())
        },
        predictions=tuple(predictions),
    )
