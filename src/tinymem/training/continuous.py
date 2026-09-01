"""Answer-supervised training for segmented continuous memory."""

from collections.abc import Sequence
from numbers import Real

import torch
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
