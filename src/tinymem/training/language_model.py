"""Streaming language-model training and evaluation."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from numbers import Integral, Real

import torch
from torch.nn import functional as F
from torch.optim import Optimizer

from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.memory_input import AttentionMemory
from tinymem.training.losses import next_token_cross_entropy


MemoryIntervention = Callable[[AttentionMemory], AttentionMemory]


@dataclass(frozen=True)
class LanguageModelEvaluation:
    """Hold token-weighted loss and perplexity."""

    loss: float
    perplexity: float
    predicted_tokens: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _validate_token_stream(
    token_ids: torch.Tensor,
    *,
    vocab_size: int,
) -> None:
    if not isinstance(token_ids, torch.Tensor):
        raise TypeError("token_ids must be a torch.Tensor")
    if token_ids.ndim != 1:
        raise ValueError("token_ids must have shape [tokens]")
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("token_ids must be an integer tensor")
    if token_ids.numel() < 2:
        raise ValueError("token_ids must contain at least two tokens")
    if ((token_ids < 0) | (token_ids >= vocab_size)).any():
        raise ValueError("token_ids contain values outside the vocabulary")


def sample_language_model_batch(
    token_ids: torch.Tensor,
    *,
    sequence_length: int,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device | str,
    vocab_size: int,
) -> torch.Tensor:
    """Sample fixed-length contiguous sequences from one training stream."""
    _validate_token_stream(token_ids, vocab_size=vocab_size)
    for name, value in (
        ("sequence_length", sequence_length),
        ("batch_size", batch_size),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    if sequence_length < 2 or batch_size <= 0:
        raise ValueError("sequence_length must exceed one and batch_size be positive")
    if sequence_length > token_ids.numel():
        raise ValueError("sequence_length exceeds the token stream")
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")

    starts = torch.randint(
        token_ids.numel() - int(sequence_length) + 1,
        (int(batch_size),),
        generator=generator,
    )
    batch = torch.stack(
        [
            token_ids[start : start + int(sequence_length)]
            for start in starts.tolist()
        ]
    )
    return batch.to(device=device)


def train_segmented_language_model(
    decoder: SegmentedContinuousDecoder,
    optimizer: Optimizer,
    token_ids: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    sequence_length: int,
    gradient_clip_norm: float,
    device: torch.device | str,
    seed: int,
) -> list[float]:
    """Train a segmented decoder on randomly sampled contiguous text."""
    if not isinstance(decoder, SegmentedContinuousDecoder):
        raise TypeError("decoder must be a SegmentedContinuousDecoder")
    if not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be a torch Optimizer")
    _validate_token_stream(token_ids, vocab_size=decoder.model.config.vocab_size)
    for name, value in (
        ("steps", steps),
        ("batch_size", batch_size),
        ("sequence_length", sequence_length),
        ("seed", seed),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    if steps <= 0 or batch_size <= 0 or sequence_length < 2 or seed < 0:
        raise ValueError("training counts must be positive and seed nonnegative")
    if isinstance(gradient_clip_norm, bool) or not isinstance(
        gradient_clip_norm,
        Real,
    ):
        raise TypeError("gradient_clip_norm must be a real number")
    if gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive")

    generator = torch.Generator().manual_seed(int(seed))
    losses = []
    decoder.train()
    for _ in range(int(steps)):
        input_ids = sample_language_model_batch(
            token_ids,
            sequence_length=int(sequence_length),
            batch_size=int(batch_size),
            generator=generator,
            device=device,
            vocab_size=decoder.model.config.vocab_size,
        )
        token_valid = torch.ones_like(input_ids, dtype=torch.bool)
        optimizer.zero_grad(set_to_none=True)
        output = decoder(input_ids, token_valid)
        loss = next_token_cross_entropy(output.logits, input_ids)
        if not torch.isfinite(loss):
            raise RuntimeError("language-model training produced nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), gradient_clip_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


@torch.no_grad()
def evaluate_segmented_language_model(
    decoder: SegmentedContinuousDecoder,
    token_ids: torch.Tensor,
    *,
    sequence_length: int,
    batch_size: int,
    pad_id: int,
    device: torch.device | str,
    max_tokens: int | None = None,
    final_segment_only: bool = False,
    memory_intervention: MemoryIntervention | None = None,
) -> LanguageModelEvaluation:
    """Evaluate non-overlapping stream transitions with token-weighted loss."""
    if not isinstance(decoder, SegmentedContinuousDecoder):
        raise TypeError("decoder must be a SegmentedContinuousDecoder")
    _validate_token_stream(token_ids, vocab_size=decoder.model.config.vocab_size)
    for name, value in (
        ("sequence_length", sequence_length),
        ("batch_size", batch_size),
        ("pad_id", pad_id),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    if sequence_length < 2 or batch_size <= 0:
        raise ValueError("sequence_length must exceed one and batch_size be positive")
    if not 0 <= pad_id < decoder.model.config.vocab_size:
        raise ValueError("pad_id must be in the vocabulary")
    if max_tokens is not None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, Integral):
            raise TypeError("max_tokens must be an integer or None")
        if max_tokens < 2:
            raise ValueError("max_tokens must exceed one")
        token_ids = token_ids[: int(max_tokens)]
    if not isinstance(final_segment_only, bool):
        raise TypeError("final_segment_only must be a boolean")
    if memory_intervention is not None and not callable(memory_intervention):
        raise TypeError("memory_intervention must be callable or None")

    stride = int(sequence_length) - 1
    chunks = [
        token_ids[start : start + int(sequence_length)]
        for start in range(0, token_ids.numel() - 1, stride)
        if token_ids[start : start + int(sequence_length)].numel() >= 2
    ]
    was_training = decoder.training
    decoder.eval()
    total_loss = 0.0
    predicted_tokens = 0
    try:
        for start in range(0, len(chunks), int(batch_size)):
            batch = chunks[start : start + int(batch_size)]
            max_length = max(chunk.numel() for chunk in batch)
            input_ids = torch.full(
                (len(batch), max_length),
                int(pad_id),
                dtype=torch.long,
                device=device,
            )
            token_valid = torch.zeros_like(input_ids, dtype=torch.bool)
            targets = torch.full_like(input_ids, -100)
            for row, chunk in enumerate(batch):
                length = chunk.numel()
                input_ids[row, :length] = chunk.to(device=device)
                token_valid[row, :length] = True
                targets[row, :length] = chunk.to(device=device)
                if final_segment_only:
                    target_start = max(1, length - decoder.segment_length)
                    targets[row, 1:target_start] = -100
            output = decoder(
                input_ids,
                token_valid,
                memory_intervention=memory_intervention,
            )
            shifted_targets = targets[:, 1:]
            count = int((shifted_targets != -100).sum())
            if count == 0:
                continue
            loss = F.cross_entropy(
                output.logits[:, :-1].reshape(
                    -1,
                    decoder.model.config.vocab_size,
                ),
                shifted_targets.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_loss += float(loss.cpu())
            predicted_tokens += count
    finally:
        decoder.train(was_training)

    if predicted_tokens == 0:
        raise ValueError("evaluation produced no target tokens")
    mean_loss = total_loss / predicted_tokens
    return LanguageModelEvaluation(
        loss=mean_loss,
        perplexity=math.exp(mean_loss),
        predicted_tokens=predicted_tokens,
    )


@torch.no_grad()
def evaluate_streaming_language_model(
    decoder: SegmentedContinuousDecoder,
    token_ids: torch.Tensor,
    *,
    chunk_tokens: int,
    device: torch.device | str,
    max_tokens: int | None = None,
    memory_intervention: MemoryIntervention | None = None,
) -> LanguageModelEvaluation:
    """Score one ordered stream while carrying memory across chunks."""
    if not isinstance(decoder, SegmentedContinuousDecoder):
        raise TypeError("decoder must be a SegmentedContinuousDecoder")
    _validate_token_stream(token_ids, vocab_size=decoder.model.config.vocab_size)
    if isinstance(chunk_tokens, bool) or not isinstance(chunk_tokens, Integral):
        raise TypeError("chunk_tokens must be an integer")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if chunk_tokens % decoder.segment_length != 0:
        raise ValueError("chunk_tokens must be a multiple of segment_length")
    if max_tokens is not None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, Integral):
            raise TypeError("max_tokens must be an integer or None")
        if max_tokens < 2:
            raise ValueError("max_tokens must exceed one")
        token_ids = token_ids[: int(max_tokens)]
    if memory_intervention is not None and not callable(memory_intervention):
        raise TypeError("memory_intervention must be callable or None")

    was_training = decoder.training
    decoder.eval()
    memory = None
    previous_logits = None
    position = 0
    total_loss = 0.0
    predicted_tokens = 0
    try:
        for start in range(0, token_ids.numel(), int(chunk_tokens)):
            chunk = token_ids[start : start + int(chunk_tokens)].to(
                device=device
            ).unsqueeze(0)
            output = decoder(
                chunk,
                torch.ones_like(chunk, dtype=torch.bool),
                initial_memory=memory,
                position_offset=position,
                memory_intervention=memory_intervention,
            )
            if previous_logits is not None:
                boundary_loss = F.cross_entropy(
                    previous_logits,
                    chunk[:, 0],
                    reduction="sum",
                )
                total_loss += float(boundary_loss.cpu())
                predicted_tokens += 1
            if chunk.shape[1] > 1:
                interior_loss = F.cross_entropy(
                    output.logits[:, :-1].reshape(
                        -1,
                        decoder.model.config.vocab_size,
                    ),
                    chunk[:, 1:].reshape(-1),
                    reduction="sum",
                )
                total_loss += float(interior_loss.cpu())
                predicted_tokens += chunk.shape[1] - 1
            memory = AttentionMemory(
                values=output.memory,
                valid=output.memory_valid,
                positions=output.memory_positions,
            )
            previous_logits = output.logits[:, -1]
            position += chunk.shape[1]
    finally:
        decoder.train(was_training)

    if predicted_tokens != token_ids.numel() - 1:
        raise RuntimeError("streaming evaluation did not score every transition")
    mean_loss = total_loss / predicted_tokens
    return LanguageModelEvaluation(
        loss=mean_loss,
        perplexity=math.exp(mean_loss),
        predicted_tokens=predicted_tokens,
    )
