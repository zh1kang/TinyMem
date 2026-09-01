"""Answer-supervised training for segmented continuous memory."""

from collections.abc import Sequence
from numbers import Real

import torch
from torch.optim import Optimizer

from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.training.controlled_qa import (
    EncodedQAExample,
    collate_answer_supervision,
)
from tinymem.training.losses import next_token_cross_entropy


def collate_segmented_answer_supervision(
    examples: Sequence[EncodedQAExample],
    *,
    pad_id: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad examples and return IDs, targets, and a valid-token mask."""
    input_ids, target_ids = collate_answer_supervision(
        examples,
        pad_id=pad_id,
        device=device,
    )
    token_valid = torch.zeros_like(input_ids, dtype=torch.bool)
    for row, example in enumerate(examples):
        token_valid[row, : len(example.input_ids)] = True
    return input_ids, target_ids, token_valid


def train_continuous_answer_supervision(
    decoder: SegmentedContinuousDecoder,
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
    """Train a segmented decoder and return one finite loss per step."""
    if not isinstance(decoder, SegmentedContinuousDecoder):
        raise TypeError("decoder must be a SegmentedContinuousDecoder")
    if not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be a torch Optimizer")
    for name, value in (
        ("steps", steps),
        ("batch_size", batch_size),
        ("seed", seed),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if steps <= 0 or batch_size <= 0 or seed < 0:
        raise ValueError(
            "steps and batch_size must be positive and seed nonnegative"
        )
    if isinstance(gradient_clip_norm, bool) or not isinstance(
        gradient_clip_norm,
        Real,
    ):
        raise TypeError("gradient_clip_norm must be a real number")
    if gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive")
    if not examples:
        raise ValueError("examples must be nonempty")
    if not all(isinstance(example, EncodedQAExample) for example in examples):
        raise TypeError("examples must contain EncodedQAExample values")

    generator = torch.Generator().manual_seed(seed)
    losses = []
    decoder.train()
    for _ in range(steps):
        indices = torch.randint(
            len(examples),
            (batch_size,),
            generator=generator,
        ).tolist()
        batch = [examples[index] for index in indices]
        input_ids, target_ids, token_valid = collate_segmented_answer_supervision(
            batch,
            pad_id=pad_id,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        output = decoder(input_ids, token_valid)
        loss = next_token_cross_entropy(output.logits, target_ids)
        if not torch.isfinite(loss):
            raise RuntimeError("continuous-memory training produced nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            decoder.parameters(),
            gradient_clip_norm,
        )
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses
