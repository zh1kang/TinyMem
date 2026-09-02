"""Answer-supervised training for segmented continuous memory."""

from collections.abc import Sequence
from numbers import Real

import torch
from torch.nn import functional as F
from torch.optim import Optimizer

from tinymem.data.schema import ReasoningExample
from tinymem.data.symbolic_world import parse_qa1_movement
from tinymem.data.vocabulary import ControlledVocabulary
from tinymem.memory.controller import AdaptiveWriteController
from tinymem.memory.discrete_compressor import DiscreteMemoryCompressor
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.training.controlled_qa import (
    EncodedQAExample,
    collate_answer_supervision,
    format_qa_prompt,
)
from tinymem.training.controller import controller_write_cost
from tinymem.training.discrete import (
    GumbelTemperatureSchedule,
    codebook_usage_loss,
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


def evidence_segment_write_targets(
    example: ReasoningExample,
    vocabulary: ControlledVocabulary,
    *,
    segment_length: int,
    prompt_ids: Sequence[int],
) -> tuple[bool, ...]:
    """Label prompt segments that overlap exact controlled evidence."""
    if not isinstance(example, ReasoningExample):
        raise TypeError("example must be a ReasoningExample")
    if not isinstance(vocabulary, ControlledVocabulary):
        raise TypeError("vocabulary must be a ControlledVocabulary")
    if isinstance(segment_length, bool) or not isinstance(segment_length, int):
        raise TypeError("segment_length must be an integer")
    if segment_length <= 0:
        raise ValueError("segment_length must be positive")
    if not isinstance(prompt_ids, Sequence) or isinstance(prompt_ids, (str, bytes)):
        raise TypeError("prompt_ids must be a sequence of integers")
    if not prompt_ids:
        raise ValueError("prompt_ids must be nonempty")
    if example.evidence_facts is None:
        raise ValueError("write supervision requires exact evidence facts")

    segment_count = (len(prompt_ids) + segment_length - 1) // segment_length
    targets = [False] * segment_count
    for fact in example.evidence_facts:
        fact_ids = vocabulary.encode(fact.text)
        start = 1 + len(vocabulary.encode(example.context[: fact.start_char]))
        end = start + len(fact_ids)
        if list(prompt_ids[start:end]) != fact_ids:
            raise ValueError("evidence span does not align with prompt tokens")
        for segment in range(
            start // segment_length,
            (end - 1) // segment_length + 1,
        ):
            targets[segment] = True
    return tuple(targets)


def encode_qa_with_evidence_write_targets(
    example: ReasoningExample,
    vocabulary: ControlledVocabulary,
    *,
    segment_length: int,
) -> EncodedQAExample:
    """Encode one controlled example with exact evidence-write labels."""
    prompt_ids = tuple(
        vocabulary.encode(format_qa_prompt(example), add_bos=True)
    )
    answer_ids = vocabulary.encode(example.answer)
    if len(answer_ids) != 1:
        raise ValueError("controlled answer must encode to exactly one token")
    targets = list(
        evidence_segment_write_targets(
            example,
            vocabulary,
            segment_length=segment_length,
            prompt_ids=prompt_ids,
        )
    )
    input_ids = (*prompt_ids, answer_ids[0])
    input_segment_count = (
        len(input_ids) + segment_length - 1
    ) // segment_length
    targets.extend(False for _ in range(input_segment_count - len(targets)))
    return EncodedQAExample(
        input_ids=input_ids,
        answer_id=answer_ids[0],
        source_example_id=example.source_example_id,
        segment_write_targets=tuple(targets),
    )


def encode_update_with_write_targets(
    example: ReasoningExample,
    vocabulary: ControlledVocabulary,
    *,
    segment_length: int,
) -> EncodedQAExample:
    """Encode one update episode with operation-level write labels."""
    if not isinstance(example, ReasoningExample):
        raise TypeError("example must be a ReasoningExample")
    if example.task_id != "correction_deletion":
        raise ValueError("update encoding requires correction_deletion data")
    if not isinstance(vocabulary, ControlledVocabulary):
        raise TypeError("vocabulary must be a ControlledVocabulary")
    if isinstance(segment_length, bool) or not isinstance(segment_length, int):
        raise TypeError("segment_length must be an integer")
    if segment_length <= 0:
        raise ValueError("segment_length must be positive")

    if not example.question.startswith("QUERY ") or not example.question.endswith("?"):
        raise ValueError("invalid update question")
    query = example.question.removeprefix("QUERY ").removesuffix("?").split()
    if len(query) != 2:
        raise ValueError("update question must identify one entity and attribute")
    query_key = tuple(query)
    prompt_ids = tuple(vocabulary.encode(format_qa_prompt(example), add_bos=True))
    answer_ids = vocabulary.encode(example.answer)
    if len(answer_ids) != 1:
        raise ValueError("controlled answer must encode to exactly one token")
    input_ids = (*prompt_ids, answer_ids[0])
    segment_count = (len(input_ids) + segment_length - 1) // segment_length
    targets = [False] * segment_count
    event_types: list[set[str]] = [set() for _ in range(segment_count)]

    offset = 0
    for line in example.context.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        parts = content.removesuffix(".").split()
        if not parts:
            raise ValueError("update context must contain nonempty operations")
        operation = parts[0].lower()
        start = 1 + len(vocabulary.encode(example.context[:offset]))
        end = start + len(vocabulary.encode(content))
        if end <= start or list(prompt_ids[start:end]) != vocabulary.encode(content):
            raise ValueError("update operation does not align with prompt tokens")
        is_relevant = (
            operation in {"set", "correct", "delete"}
            and len(parts) >= 3
            and tuple(parts[1:3]) == query_key
        )
        label = (
            "correction"
            if operation == "correct"
            else operation
            if is_relevant
            else "background"
        )
        for segment in range(
            start // segment_length,
            (end - 1) // segment_length + 1,
        ):
            event_types[segment].add(label)
            targets[segment] = targets[segment] or is_relevant
        offset += len(line)

    return EncodedQAExample(
        input_ids=input_ids,
        answer_id=answer_ids[0],
        source_example_id=example.source_example_id,
        segment_write_targets=tuple(targets),
        segment_event_types=tuple(
            tuple(sorted(labels)) if labels else ("background",)
            for labels in event_types
        ),
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


def encode_qa_with_distributed_facts(
    example: ReasoningExample,
    vocabulary: ControlledVocabulary,
    *,
    distractor_ids: Sequence[int],
    segment_length: int,
    gap_rotation: int = 0,
) -> EncodedQAExample:
    """Distribute controlled facts through filler and label fact segments."""
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
        raise ValueError("distractor_ids must contain valid vocabulary IDs")
    for name, value in (
        ("segment_length", segment_length),
        ("gap_rotation", gap_rotation),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if segment_length <= 0:
        raise ValueError("segment_length must be positive")
    if gap_rotation < 0:
        raise ValueError("gap_rotation must be nonnegative")

    facts = example.context.splitlines()
    if not facts or any(not fact for fact in facts):
        raise ValueError("controlled context must contain nonempty fact lines")
    gap_count = len(facts) + 1
    base_length, remainder = divmod(len(distractor_ids), gap_count)
    long_gaps = {
        (gap_rotation + offset) % gap_count
        for offset in range(remainder)
    }
    gap_lengths = [
        base_length + int(index in long_gaps)
        for index in range(gap_count)
    ]
    gaps = []
    offset = 0
    for length in gap_lengths:
        gaps.append(distractor_ids[offset : offset + length])
        offset += length

    separator_ids = vocabulary.encode("\n")
    input_ids = [vocabulary.token_to_id["<bos>"]]
    fact_spans = []
    seen_people: set[str] = set()
    for gap, fact in zip(gaps[:-1], facts, strict=True):
        input_ids.extend(gap)
        input_ids.extend(separator_ids)
        start = len(input_ids)
        input_ids.extend(vocabulary.encode(fact))
        event_type = "relevant_fact"
        if example.task_id == "qa1":
            person, _ = parse_qa1_movement(fact, len(fact_spans) + 1)
            event_type = "correction" if person in seen_people else "set"
            seen_people.add(person)
        fact_spans.append((start, len(input_ids), event_type))
    input_ids.extend(separator_ids)
    input_ids.extend(gaps[-1])
    input_ids.extend(vocabulary.encode(f"\n{example.question} "))
    answer_ids = vocabulary.encode(example.answer)
    if len(answer_ids) != 1:
        raise ValueError("controlled answer must encode to exactly one token")
    input_ids.append(answer_ids[0])

    segment_count = (len(input_ids) + segment_length - 1) // segment_length
    write_targets = tuple(
        any(
            segment_start < fact_end
            and segment_start + segment_length > fact_start
            for fact_start, fact_end, _ in fact_spans
        )
        for segment_start in range(
            0,
            segment_count * segment_length,
            segment_length,
        )
    )
    event_types = []
    for segment_start in range(
        0,
        segment_count * segment_length,
        segment_length,
    ):
        labels = {
            event_type
            for fact_start, fact_end, event_type in fact_spans
            if segment_start < fact_end
            and segment_start + segment_length > fact_start
        }
        event_types.append(tuple(sorted(labels)) if labels else ("background",))
    return EncodedQAExample(
        input_ids=tuple(input_ids),
        answer_id=answer_ids[0],
        source_example_id=example.source_example_id,
        segment_write_targets=write_targets,
        segment_event_types=tuple(event_types),
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
    codebook_usage_loss_weight: float = 0.0,
    temperature_schedule: GumbelTemperatureSchedule | None = None,
    temperature_step_offset: int = 0,
    write_cost_weight: float = 0.0,
    controller_temperature_schedule: GumbelTemperatureSchedule | None = None,
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
    if isinstance(codebook_usage_loss_weight, bool) or not isinstance(
        codebook_usage_loss_weight,
        Real,
    ):
        raise TypeError("codebook_usage_loss_weight must be a real number")
    if codebook_usage_loss_weight < 0:
        raise ValueError("codebook_usage_loss_weight must be nonnegative")
    if isinstance(write_cost_weight, bool) or not isinstance(
        write_cost_weight,
        Real,
    ):
        raise TypeError("write_cost_weight must be a real number")
    if write_cost_weight < 0:
        raise ValueError("write_cost_weight must be nonnegative")
    if temperature_schedule is not None and not isinstance(
        temperature_schedule,
        GumbelTemperatureSchedule,
    ):
        raise TypeError(
            "temperature_schedule must be a GumbelTemperatureSchedule or None"
        )
    if controller_temperature_schedule is not None and not isinstance(
        controller_temperature_schedule,
        GumbelTemperatureSchedule,
    ):
        raise TypeError(
            "controller_temperature_schedule must be a "
            "GumbelTemperatureSchedule or None"
        )
    if isinstance(temperature_step_offset, bool) or not isinstance(
        temperature_step_offset,
        int,
    ):
        raise TypeError("temperature_step_offset must be an integer")
    if temperature_step_offset < 0:
        raise ValueError("temperature_step_offset must be nonnegative")
    discrete_compressor = (
        decoder.compressor
        if isinstance(decoder.compressor, DiscreteMemoryCompressor)
        else None
    )
    if codebook_usage_loss_weight > 0 and discrete_compressor is None:
        raise ValueError("codebook usage loss requires a discrete compressor")
    if temperature_schedule is not None and discrete_compressor is None:
        raise ValueError("temperature scheduling requires a discrete compressor")
    adaptive_controller = (
        decoder.write_controller
        if isinstance(decoder.write_controller, AdaptiveWriteController)
        else None
    )
    if write_cost_weight > 0 and adaptive_controller is None:
        raise ValueError("write cost requires an adaptive write controller")
    if controller_temperature_schedule is not None and adaptive_controller is None:
        raise ValueError(
            "controller temperature scheduling requires an adaptive controller"
        )
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
    for step in range(steps):
        if temperature_schedule is not None:
            assert discrete_compressor is not None
            discrete_compressor.set_temperature(
                temperature_schedule.value(temperature_step_offset + step)
            )
        if controller_temperature_schedule is not None:
            assert adaptive_controller is not None
            adaptive_controller.set_temperature(
                controller_temperature_schedule.value(
                    temperature_step_offset + step
                )
            )
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
        if codebook_usage_loss_weight > 0:
            if (
                output.code_assignments is None
                or output.proposed_code_valid is None
            ):
                raise ValueError(
                    "codebook usage loss requires discrete code traces"
                )
            loss = loss + codebook_usage_loss_weight * codebook_usage_loss(
                output.code_assignments,
                output.proposed_code_valid,
            )
        if write_cost_weight > 0:
            if (
                output.controller_probabilities is None
                or output.controller_valid is None
            ):
                raise ValueError(
                    "write cost requires adaptive controller probabilities"
                )
            loss = loss + write_cost_weight * controller_write_cost(
                output.controller_probabilities,
                output.controller_valid,
            )
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
            class_losses = []
            if positive_mask.any():
                positive_weights = positive_mask.to(
                    dtype=output.write_logits.dtype
                )
                class_losses.append(
                    (
                        F.softplus(-output.write_logits)
                        * positive_weights
                    ).sum()
                    / positive_weights.sum()
                )
            if negative_mask.any():
                negative_weights = negative_mask.to(
                    dtype=output.write_logits.dtype
                )
                class_losses.append(
                    (
                        F.softplus(output.write_logits)
                        * negative_weights
                    ).sum()
                    / negative_weights.sum()
                )
            write_loss = torch.stack(class_losses).mean()
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
