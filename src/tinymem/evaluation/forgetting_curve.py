"""Evidence-delay evaluation for the local-only controlled model."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import torch

from tinymem.data.schema import EvidenceFact, ReasoningExample
from tinymem.data.vocabulary import ControlledVocabulary
from tinymem.model.streaming import StreamingDecoder
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.training.controlled_qa import format_qa_prompt


DEFAULT_DELAY_LIMITS = (32, 64, 128, 256, 512)
_QA1_QUESTION = re.compile(r"Where is (?P<person>[A-Za-z]+)\?\s*")
_DESTINATION = re.compile(r"the (?P<destination>[A-Za-z]+)\.$")


@dataclass(frozen=True)
class DelayResult:
    """Store exact accuracy counts for one evidence-delay bucket."""

    label: str
    minimum_delay: int
    maximum_delay: int | None
    correct: int
    count: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.count if self.count else 0.0

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "accuracy": self.accuracy}


def qa1_answer_evidence(example: ReasoningExample) -> EvidenceFact:
    """Return the final movement fact for the person in a qa1 question."""
    if not isinstance(example, ReasoningExample):
        raise TypeError("example must be a ReasoningExample")
    if example.task_id != "qa1":
        raise ValueError("qa1 evidence lookup requires task_id 'qa1'")
    if example.evidence_facts is None:
        raise ValueError("qa1 evidence lookup requires exact evidence facts")
    question_match = _QA1_QUESTION.fullmatch(example.question)
    if question_match is None:
        raise ValueError("qa1 question must have the form 'Where is PERSON?'")
    person = question_match.group("person")
    candidates = [
        fact
        for fact in example.evidence_facts
        if fact.text.startswith(f"{person} ")
    ]
    if not candidates:
        raise ValueError("qa1 example has no movement fact for the queried person")
    evidence = candidates[-1]
    destination_match = _DESTINATION.search(evidence.text)
    if destination_match is None:
        raise ValueError("qa1 evidence fact has no destination")
    if destination_match.group("destination") != example.answer:
        raise ValueError("qa1 answer does not match the final movement fact")
    return evidence


def qa1_evidence_delay_tokens(
    example: ReasoningExample,
    vocabulary: ControlledVocabulary,
) -> int:
    """Count prompt tokens after the answer-bearing evidence fact."""
    if not isinstance(vocabulary, ControlledVocabulary):
        raise TypeError("vocabulary must be a ControlledVocabulary")
    evidence = qa1_answer_evidence(example)
    trailing_prompt = (
        example.context[evidence.end_char :] + f"\n{example.question} "
    )
    return len(vocabulary.encode(trailing_prompt))


def delay_label(delay: int, limits: tuple[int, ...] = DEFAULT_DELAY_LIMITS) -> str:
    """Return the inclusive upper-bound label for one token delay."""
    if isinstance(delay, bool) or not isinstance(delay, int):
        raise TypeError("delay must be an integer")
    if delay < 0:
        raise ValueError("delay must be nonnegative")
    if not limits or any(limit <= 0 for limit in limits):
        raise ValueError("limits must contain positive values")
    if tuple(sorted(set(limits))) != limits:
        raise ValueError("limits must be unique and strictly increasing")
    lower = 0
    for limit in limits:
        if delay <= limit:
            return f"{lower}-{limit}"
        lower = limit + 1
    return f">{limits[-1]}"


@torch.no_grad()
def evaluate_local_forgetting_curve(
    model: DecoderOnlyTransformer,
    vocabulary: ControlledVocabulary,
    examples: Sequence[ReasoningExample],
    *,
    batch_size: int,
    device: torch.device | str,
    limits: tuple[int, ...] = DEFAULT_DELAY_LIMITS,
) -> list[DelayResult]:
    """Evaluate qa1 answers using only the final local attention window."""
    if not isinstance(model, DecoderOnlyTransformer):
        raise TypeError("model must be a DecoderOnlyTransformer")
    if not isinstance(vocabulary, ControlledVocabulary):
        raise TypeError("vocabulary must be a ControlledVocabulary")
    if not examples:
        raise ValueError("examples must be nonempty")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    prepared: list[tuple[tuple[int, ...], int, int]] = []
    for example in examples:
        prompt_ids = vocabulary.encode(format_qa_prompt(example), add_bos=True)
        answer_ids = vocabulary.encode(example.answer)
        if len(answer_ids) != 1:
            raise ValueError("controlled answer must encode to exactly one token")
        delay = qa1_evidence_delay_tokens(example, vocabulary)
        prepared.append((tuple(prompt_ids), answer_ids[0], delay))
    prepared.sort(key=lambda item: len(item[0]))

    counts: dict[str, list[int]] = {}
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start : start + batch_size]
            max_length = max(len(prompt) for prompt, _, _ in batch)
            input_ids = torch.full(
                (len(batch), max_length),
                vocabulary.token_to_id["<pad>"],
                dtype=torch.long,
                device=device,
            )
            prompt_lengths = []
            for row, (prompt, _, _) in enumerate(batch):
                input_ids[row, : len(prompt)] = torch.tensor(
                    prompt,
                    dtype=torch.long,
                    device=device,
                )
                prompt_lengths.append(len(prompt))
            predictions = [-1] * len(batch)
            stream = StreamingDecoder(model, segment_length=1)
            for position in range(max_length):
                logits = stream.process_segment(
                    input_ids[:, position : position + 1]
                )
                for row, prompt_length in enumerate(prompt_lengths):
                    if position + 1 == prompt_length:
                        predictions[row] = int(logits[row, 0].argmax().cpu())
            if any(prediction < 0 for prediction in predictions):
                raise RuntimeError("failed to capture every streamed answer prediction")
            for prediction, (_, answer_id, delay) in zip(
                predictions,
                batch,
                strict=True,
            ):
                label = delay_label(delay, limits)
                totals = counts.setdefault(label, [0, 0])
                totals[0] += int(prediction == answer_id)
                totals[1] += 1
    finally:
        model.train(was_training)

    results: list[DelayResult] = []
    lower = 0
    for limit in limits:
        label = f"{lower}-{limit}"
        correct, count = counts.get(label, [0, 0])
        results.append(DelayResult(label, lower, limit, correct, count))
        lower = limit + 1
    label = f">{limits[-1]}"
    correct, count = counts.get(label, [0, 0])
    results.append(DelayResult(label, lower, None, correct, count))
    return results
