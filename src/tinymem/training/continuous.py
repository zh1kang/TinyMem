"""Answer-supervised training for segmented continuous memory."""

from collections.abc import Sequence
from numbers import Real

import torch
from torch.nn import functional as F
from torch.optim import Optimizer

from tinymem.data.schema import ReasoningExample
from tinymem.data.vocabulary import ControlledVocabulary
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.training.controlled_qa import (
    EncodedQAExample,
    collate_answer_supervision,
    format_qa_prompt,
)
from tinymem.training.losses import next_token_cross_entropy


def qa1_requires_cross_segment_memory(
    example: ReasoningExample,
    vocabulary: ControlledVocabulary,
    *,
    segment_length: int,
) -> bool:
    """Return whether the qa1 evidence ends before the query segment."""
    if not isinstance(example, ReasoningExample):
        raise TypeError("example must be a ReasoningExample")
    if not isinstance(vocabulary, ControlledVocabulary):
        raise TypeError("vocabulary must be a ControlledVocabulary")
    if isinstance(segment_length, bool) or not isinstance(segment_length, int):
        raise TypeError("segment_length must be an integer")
    if segment_length <= 0:
        raise ValueError("segment_length must be positive")
    if example.task_id != "qa1":
        raise ValueError("cross-segment memory selection requires qa1")
    if example.supporting_fact_ids is None or len(example.supporting_fact_ids) != 1:
        raise ValueError("qa1 examples must contain one supporting fact ID")
    if example.context_fact_ids is None:
        raise ValueError("qa1 examples must contain context fact IDs")

    support_id = example.supporting_fact_ids[0]
    try:
        fact_index = example.context_fact_ids.index(support_id)
    except ValueError as error:
        raise ValueError("supporting fact ID is absent from the context") from error
    context_lines = example.context.splitlines(keepends=True)
    if len(context_lines) != len(example.context_fact_ids):
        raise ValueError("context lines and fact IDs must remain aligned")

    evidence_start = sum(len(line) for line in context_lines[:fact_index])
    evidence_end = evidence_start + len(
        context_lines[fact_index].rstrip("\r\n")
    )
    evidence_prefix_ids = vocabulary.encode(
        example.context[:evidence_end],
        add_bos=True,
    )
    if not evidence_prefix_ids:
        raise ValueError("supporting fact must encode to at least one token")
    evidence_last_position = len(evidence_prefix_ids) - 1
    prompt_ids = vocabulary.encode(format_qa_prompt(example), add_bos=True)
    query_position = len(prompt_ids) - 1
    return (
        evidence_last_position // segment_length
        < query_position // segment_length
    )


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


def encode_qa_with_token_distractor(
    example: ReasoningExample,
    vocabulary: ControlledVocabulary,
    distractor_ids: Sequence[int],
) -> EncodedQAExample:
    """Insert encoded distractor tokens between a context and its question."""
    if not isinstance(example, ReasoningExample):
        raise TypeError("example must be a ReasoningExample")
    if not isinstance(vocabulary, ControlledVocabulary):
        raise TypeError("vocabulary must be a ControlledVocabulary")
    if not isinstance(distractor_ids, Sequence) or isinstance(
        distractor_ids,
        (str, bytes),
    ):
        raise TypeError("distractor_ids must be a sequence of integers")
    if any(
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or not 0 <= token_id < len(vocabulary)
        for token_id in distractor_ids
    ):
        raise ValueError("distractor token IDs must be valid vocabulary IDs")

    prefix_ids = vocabulary.encode(
        f"{example.context}\n",
        add_bos=True,
    )
    question_ids = vocabulary.encode(f"\n{example.question} ")
    answer_ids = vocabulary.encode(example.answer)
    if len(answer_ids) != 1:
        raise ValueError("controlled answer must encode to exactly one token")
    answer_id = answer_ids[0]
    return EncodedQAExample(
        input_ids=tuple(
            (*prefix_ids, *distractor_ids, *question_ids, answer_id)
        ),
        answer_id=answer_id,
        source_example_id=example.source_example_id,
    )


def encode_qa_with_token_distractors(
    example: ReasoningExample,
    vocabulary: ControlledVocabulary,
    *,
    prefix_distractor_ids: Sequence[int],
    suffix_distractor_ids: Sequence[int],
    segment_length: int,
) -> EncodedQAExample:
    """Place a controlled context inside filler and label its write segments."""
    if not isinstance(example, ReasoningExample):
        raise TypeError("example must be a ReasoningExample")
    if not isinstance(vocabulary, ControlledVocabulary):
        raise TypeError("vocabulary must be a ControlledVocabulary")
    if isinstance(segment_length, bool) or not isinstance(segment_length, int):
        raise TypeError("segment_length must be an integer")
    if segment_length <= 0:
        raise ValueError("segment_length must be positive")
    for name, token_ids in (
        ("prefix_distractor_ids", prefix_distractor_ids),
        ("suffix_distractor_ids", suffix_distractor_ids),
    ):
        if not isinstance(token_ids, Sequence) or isinstance(
            token_ids,
            (str, bytes),
        ):
            raise TypeError(f"{name} must be a sequence of integers")
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id < len(vocabulary)
            for token_id in token_ids
        ):
            raise ValueError(f"{name} must contain valid vocabulary IDs")

    separator_ids = vocabulary.encode("\n")
    context_ids = vocabulary.encode(example.context)
    question_ids = vocabulary.encode(f"\n{example.question} ")
    answer_ids = vocabulary.encode(example.answer)
    if not context_ids:
        raise ValueError("controlled context must encode to at least one token")
    if len(answer_ids) != 1:
        raise ValueError("controlled answer must encode to exactly one token")

    input_prefix = (
        vocabulary.token_to_id["<bos>"],
        *prefix_distractor_ids,
        *separator_ids,
    )
    context_start = len(input_prefix)
    context_end = context_start + len(context_ids)
    input_ids = (
        *input_prefix,
        *context_ids,
        *separator_ids,
        *suffix_distractor_ids,
        *question_ids,
        answer_ids[0],
    )
    segment_count = (len(input_ids) + segment_length - 1) // segment_length
    write_targets = tuple(
        segment_start < context_end
        and segment_start + segment_length > context_start
        for segment_start in range(0, segment_count * segment_length, segment_length)
    )
    return EncodedQAExample(
        input_ids=tuple(input_ids),
        answer_id=answer_ids[0],
        source_example_id=example.source_example_id,
        segment_write_targets=write_targets,
    )


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
    write_loss_weight: float = 0.0,
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
    if isinstance(write_loss_weight, bool) or not isinstance(
        write_loss_weight,
        Real,
    ):
        raise TypeError("write_loss_weight must be a real number")
    if write_loss_weight < 0:
        raise ValueError("write_loss_weight must be nonnegative")
    if not examples:
        raise ValueError("examples must be nonempty")
    if not all(isinstance(example, EncodedQAExample) for example in examples):
        raise TypeError("examples must contain EncodedQAExample values")
    if write_loss_weight > 0 and not all(
        example.segment_write_targets is not None for example in examples
    ):
        raise ValueError(
            "positive write loss requires segment write targets for every example"
        )

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
        if write_loss_weight > 0:
            if output.write_logits is None:
                raise ValueError("positive write loss requires a gated memory bank")
            write_targets = torch.zeros_like(
                output.write_logits,
                dtype=torch.bool,
            )
            write_valid = torch.zeros_like(write_targets)
            for row, example in enumerate(batch):
                targets = example.segment_write_targets
                assert targets is not None
                expected_count = (
                    len(example.input_ids) + decoder.segment_length - 1
                ) // decoder.segment_length
                if len(targets) != expected_count:
                    raise ValueError(
                        "segment write targets must match the encoded sequence"
                    )
                write_targets[row, :expected_count] = torch.tensor(
                    targets,
                    dtype=torch.bool,
                    device=device,
                )
                write_valid[row, :expected_count] = True
            positive_mask = write_valid & write_targets
            negative_mask = write_valid & ~write_targets
            if not positive_mask.any() or not negative_mask.any():
                raise ValueError(
                    "write supervision requires positive and negative segments"
                )
            positive_weights = positive_mask.to(
                dtype=output.write_logits.dtype
            )
            negative_weights = negative_mask.to(
                dtype=output.write_logits.dtype
            )
            positive_loss = (
                F.softplus(-output.write_logits) * positive_weights
            ).sum() / positive_weights.sum()
            negative_loss = (
                F.softplus(output.write_logits) * negative_weights
            ).sum() / negative_weights.sum()
            write_loss = 0.5 * (
                positive_loss + negative_loss
            )
            loss = loss + write_loss_weight * write_loss
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
