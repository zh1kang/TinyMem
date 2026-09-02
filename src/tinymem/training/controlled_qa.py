"""Answer-supervised training helpers for controlled reasoning tasks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import torch
from torch.optim import Optimizer

from tinymem.data.schema import ReasoningExample
from tinymem.data.vocabulary import ControlledVocabulary
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.training.losses import next_token_cross_entropy


@dataclass(frozen=True)
class EncodedQAExample:
    """Store one prompt followed by its single-token answer."""

    input_ids: tuple[int, ...]
    answer_id: int
    source_example_id: str
    segment_write_targets: tuple[bool, ...] | None = None
    segment_event_types: tuple[tuple[str, ...], ...] | None = None


def format_qa_prompt(example: ReasoningExample) -> str:
    """Format context and question so the next token is the answer."""
    if not isinstance(example, ReasoningExample):
        raise TypeError("example must be a ReasoningExample")
    return f"{example.context}\n{example.question} "


def build_qa_vocabulary(
    examples: Sequence[ReasoningExample],
) -> ControlledVocabulary:
    """Build a training-only vocabulary from controlled examples."""
    if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)):
        raise TypeError("examples must be a sequence of ReasoningExample values")
    if not examples:
        raise ValueError("examples must be nonempty")
    texts: list[str] = []
    for example in examples:
        if not isinstance(example, ReasoningExample):
            raise TypeError("examples must contain ReasoningExample values")
        texts.extend((example.context, example.question, example.answer))
    return ControlledVocabulary.from_texts(texts)


def encode_qa_example(
    example: ReasoningExample,
    vocabulary: ControlledVocabulary,
) -> EncodedQAExample:
    """Encode one prompt and require a single-token controlled answer."""
    if not isinstance(example, ReasoningExample):
        raise TypeError("example must be a ReasoningExample")
    if not isinstance(vocabulary, ControlledVocabulary):
        raise TypeError("vocabulary must be a ControlledVocabulary")
    prompt_ids = vocabulary.encode(format_qa_prompt(example), add_bos=True)
    answer_ids = vocabulary.encode(example.answer)
    if len(answer_ids) != 1:
        raise ValueError("controlled answer must encode to exactly one token")
    answer_id = answer_ids[0]
    return EncodedQAExample(
        input_ids=tuple((*prompt_ids, answer_id)),
        answer_id=answer_id,
        source_example_id=example.source_example_id,
    )


def encode_qa_examples(
    examples: Sequence[ReasoningExample],
    vocabulary: ControlledVocabulary,
    *,
    max_tokens: int,
) -> tuple[list[EncodedQAExample], int]:
    """Encode examples and count sequences that exceed the local window."""
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise TypeError("max_tokens must be an integer")
    if max_tokens <= 1:
        raise ValueError("max_tokens must be greater than one")
    encoded: list[EncodedQAExample] = []
    skipped = 0
    for example in examples:
        item = encode_qa_example(example, vocabulary)
        if len(item.input_ids) > max_tokens:
            skipped += 1
        else:
            encoded.append(item)
    if not encoded:
        raise ValueError("no encoded examples fit within max_tokens")
    return encoded, skipped


def collate_answer_supervision(
    examples: Sequence[EncodedQAExample],
    *,
    pad_id: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad examples and supervise only each answer token."""
    if not examples:
        raise ValueError("examples must be nonempty")
    if not all(isinstance(example, EncodedQAExample) for example in examples):
        raise TypeError("examples must contain EncodedQAExample values")
    if isinstance(pad_id, bool) or not isinstance(pad_id, int):
        raise TypeError("pad_id must be an integer")
    max_length = max(len(example.input_ids) for example in examples)
    input_ids = torch.full(
        (len(examples), max_length),
        pad_id,
        dtype=torch.long,
        device=device,
    )
    target_ids = torch.full_like(input_ids, -100)
    for row, example in enumerate(examples):
        length = len(example.input_ids)
        if length < 2:
            raise ValueError("encoded examples must contain a prompt and answer")
        input_ids[row, :length] = torch.tensor(
            example.input_ids,
            dtype=torch.long,
            device=device,
        )
        target_ids[row, length - 1] = example.answer_id
    return input_ids, target_ids


def train_answer_supervision(
    model: DecoderOnlyTransformer,
    optimizer: Optimizer,
    examples: Sequence[EncodedQAExample],
    *,
    steps: int,
    batch_size: int,
    gradient_clip_norm: float,
    pad_id: int,
    device: torch.device | str,
    seed: int,
) -> list[float]:
    """Train a controlled model and return one loss value per step."""
    if not isinstance(model, DecoderOnlyTransformer):
        raise TypeError("model must be a DecoderOnlyTransformer")
    if not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be a torch Optimizer")
    for name, value in (("steps", steps), ("batch_size", batch_size), ("seed", seed)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if steps <= 0 or batch_size <= 0 or seed < 0:
        raise ValueError("steps and batch_size must be positive and seed nonnegative")
    if isinstance(gradient_clip_norm, bool) or not isinstance(
        gradient_clip_norm,
        Real,
    ):
        raise TypeError("gradient_clip_norm must be a real number")
    if gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive")
    if not examples:
        raise ValueError("examples must be nonempty")

    generator = torch.Generator().manual_seed(seed)
    losses: list[float] = []
    model.train()
    for _ in range(steps):
        indices = torch.randint(
            len(examples),
            (batch_size,),
            generator=generator,
        ).tolist()
        batch = [examples[index] for index in indices]
        input_ids, target_ids = collate_answer_supervision(
            batch,
            pad_id=pad_id,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = next_token_cross_entropy(logits, target_ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


@torch.no_grad()
def answer_accuracy(
    model: DecoderOnlyTransformer,
    examples: Sequence[EncodedQAExample],
    *,
    batch_size: int,
    pad_id: int,
    device: torch.device | str,
) -> float:
    """Measure exact single-token answer accuracy."""
    if not examples:
        raise ValueError("examples must be nonempty")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    was_training = model.training
    model.eval()
    correct = 0
    try:
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            prompts = [example.input_ids[:-1] for example in batch]
            max_length = max(len(prompt) for prompt in prompts)
            input_ids = torch.full(
                (len(batch), max_length),
                pad_id,
                dtype=torch.long,
                device=device,
            )
            final_prompt_indices = []
            for row, prompt in enumerate(prompts):
                input_ids[row, : len(prompt)] = torch.tensor(
                    prompt,
                    dtype=torch.long,
                    device=device,
                )
                final_prompt_indices.append(len(prompt) - 1)
            logits = model(input_ids)
            rows = torch.arange(len(batch), device=device)
            positions = torch.tensor(final_prompt_indices, device=device)
            predictions = logits[rows, positions].argmax(dim=-1).cpu()
            answers = torch.tensor([example.answer_id for example in batch])
            correct += int((predictions == answers).sum())
    finally:
        model.train(was_training)
    return correct / len(examples)
