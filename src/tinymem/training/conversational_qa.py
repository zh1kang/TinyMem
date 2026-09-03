"""Byte-level conversational QA supervision from controlled examples."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from numbers import Integral, Real

import torch
from torch.optim import Optimizer

from tinymem.data.schema import ReasoningExample
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.tokenization.byte_tokenizer import ByteTokenizer
from tinymem.training.language_model import sample_language_model_batch
from tinymem.training.losses import next_token_cross_entropy


@dataclass(frozen=True)
class ByteQAExample:
    """Hold one byte-tokenized prompt and its multi-byte answer."""

    prompt_ids: tuple[int, ...]
    answer_ids: tuple[int, ...]
    source_example_id: str
    dataset: str
    task_id: str
    split: str

    def __post_init__(self) -> None:
        for name in ("prompt_ids", "answer_ids"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not values:
                raise ValueError(f"{name} must be a nonempty tuple")
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < ByteTokenizer.vocab_size
                for value in values
            ):
                raise ValueError(f"{name} must contain valid byte-token IDs")
        for name in ("source_example_id", "dataset", "task_id", "split"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")


@dataclass(frozen=True)
class ConversationalQATrainingHistory:
    """Store total, answer, and optional language-model losses."""

    total_losses: tuple[float, ...]
    answer_losses: tuple[float, ...]
    language_model_losses: tuple[float, ...] | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def format_conversational_qa_prompt(example: ReasoningExample) -> str:
    """Map a controlled example into the external conversational boundary."""
    if not isinstance(example, ReasoningExample):
        raise TypeError("example must be a ReasoningExample")
    parts = [f"[session {example.dataset}:{example.task_id} | undated]\n"]
    parts.extend(f"User: {line}\n" for line in example.context.splitlines())
    parts.append("[question | undated]\n")
    parts.append(f"User: {example.question}\nAssistant:")
    return "".join(parts)


def encode_conversational_qa_example(
    example: ReasoningExample,
    tokenizer: ByteTokenizer,
) -> ByteQAExample:
    """Encode a controlled QA pair without placing its answer in the prompt."""
    if not isinstance(tokenizer, ByteTokenizer):
        raise TypeError("tokenizer must be a ByteTokenizer")
    return ByteQAExample(
        prompt_ids=tuple(tokenizer.encode(format_conversational_qa_prompt(example))),
        answer_ids=tuple(tokenizer.encode(example.answer)),
        source_example_id=example.source_example_id,
        dataset=example.dataset,
        task_id=example.task_id,
        split=example.split,
    )


def collate_conversational_qa(
    examples: Sequence[ByteQAExample],
    *,
    pad_id: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad examples and supervise only answer bytes plus the final newline."""
    if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)):
        raise TypeError("examples must be a sequence")
    if not examples:
        raise ValueError("examples must be nonempty")
    if not all(isinstance(example, ByteQAExample) for example in examples):
        raise TypeError("examples must contain ByteQAExample values")
    if isinstance(pad_id, bool) or not isinstance(pad_id, Integral):
        raise TypeError("pad_id must be an integer")
    if not 0 <= pad_id < ByteTokenizer.vocab_size:
        raise ValueError("pad_id must be in the byte-token vocabulary")

    newline_id = ord("\n")
    sequences = [
        example.prompt_ids + example.answer_ids + (newline_id,)
        for example in examples
    ]
    max_length = max(len(sequence) for sequence in sequences)
    input_ids = torch.full(
        (len(examples), max_length),
        int(pad_id),
        dtype=torch.long,
        device=device,
    )
    target_ids = torch.full_like(input_ids, -100)
    token_valid = torch.zeros_like(input_ids, dtype=torch.bool)
    for row, (example, sequence) in enumerate(zip(examples, sequences, strict=True)):
        length = len(sequence)
        input_ids[row, :length] = torch.tensor(
            sequence,
            dtype=torch.long,
            device=device,
        )
        token_valid[row, :length] = True
        answer_start = len(example.prompt_ids)
        target_ids[row, answer_start:length] = input_ids[row, answer_start:length]
    return input_ids, target_ids, token_valid


def train_conversational_qa(
    decoder: SegmentedContinuousDecoder,
    optimizer: Optimizer,
    examples: Sequence[ByteQAExample],
    *,
    steps: int,
    batch_size: int,
    gradient_clip_norm: float,
    pad_id: int,
    device: torch.device | str,
    seed: int,
    language_model_token_ids: torch.Tensor | None = None,
    language_model_loss_weight: float = 0.0,
    language_model_sequence_length: int = 256,
) -> ConversationalQATrainingHistory:
    """Fine-tune answer generation with an optional ordinary-LM anchor."""
    if not isinstance(decoder, SegmentedContinuousDecoder):
        raise TypeError("decoder must be a SegmentedContinuousDecoder")
    if not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be a torch Optimizer")
    if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)):
        raise TypeError("examples must be a sequence")
    if not examples or not all(
        isinstance(example, ByteQAExample) for example in examples
    ):
        raise ValueError("examples must contain ByteQAExample values")
    for name, value in (
        ("steps", steps),
        ("batch_size", batch_size),
        ("seed", seed),
        ("language_model_sequence_length", language_model_sequence_length),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    if steps <= 0 or batch_size <= 0 or seed < 0:
        raise ValueError("training counts must be positive and seed nonnegative")
    if language_model_sequence_length < 2:
        raise ValueError("language_model_sequence_length must exceed one")
    for name, value in (
        ("gradient_clip_norm", gradient_clip_norm),
        ("language_model_loss_weight", language_model_loss_weight),
    ):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number")
    if gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive")
    if language_model_loss_weight < 0:
        raise ValueError("language_model_loss_weight must be nonnegative")
    if language_model_loss_weight > 0 and language_model_token_ids is None:
        raise ValueError("positive language-model weight requires a token stream")

    generator = torch.Generator().manual_seed(int(seed))
    total_losses = []
    answer_losses = []
    language_model_losses = [] if language_model_loss_weight > 0 else None
    decoder.train()
    for _ in range(int(steps)):
        indices = torch.randint(
            len(examples),
            (int(batch_size),),
            generator=generator,
        ).tolist()
        batch = [examples[index] for index in indices]
        input_ids, target_ids, token_valid = collate_conversational_qa(
            batch,
            pad_id=pad_id,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        output = decoder(input_ids, token_valid)
        answer_loss = next_token_cross_entropy(output.logits, target_ids)
        loss = answer_loss

        language_model_loss = None
        if language_model_loss_weight > 0:
            assert language_model_token_ids is not None
            language_model_input = sample_language_model_batch(
                language_model_token_ids,
                sequence_length=int(language_model_sequence_length),
                batch_size=int(batch_size),
                generator=generator,
                device=device,
                vocab_size=decoder.model.config.vocab_size,
            )
            language_model_output = decoder(
                language_model_input,
                torch.ones_like(language_model_input, dtype=torch.bool),
            )
            language_model_loss = next_token_cross_entropy(
                language_model_output.logits,
                language_model_input,
            )
            loss = loss + language_model_loss_weight * language_model_loss

        if not torch.isfinite(loss):
            raise RuntimeError("conversational QA training produced nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), gradient_clip_norm)
        optimizer.step()
        total_losses.append(float(loss.detach().cpu()))
        answer_losses.append(float(answer_loss.detach().cpu()))
        if language_model_losses is not None:
            assert language_model_loss is not None
            language_model_losses.append(float(language_model_loss.detach().cpu()))

    return ConversationalQATrainingHistory(
        total_losses=tuple(total_losses),
        answer_losses=tuple(answer_losses),
        language_model_losses=(
            tuple(language_model_losses)
            if language_model_losses is not None
            else None
        ),
    )
