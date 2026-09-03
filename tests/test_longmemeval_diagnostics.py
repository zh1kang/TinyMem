import math

import pytest
import torch
from torch.nn import functional as F

from tinymem.data.longmemeval import (
    LongMemEvalExample,
    LongMemMessage,
    LongMemSession,
)
from tinymem.evaluation.longmemeval import (
    LongMemEvalPromptState,
    stream_longmemeval_prompt,
)
from tinymem.evaluation.longmemeval_diagnostics import (
    DIAGNOSTIC_CONDITIONS,
    evaluate_longmemeval_diagnostics,
    format_longmemeval_diagnostic_prompt,
    score_candidate_answer,
)
from tinymem.memory.continuous import MeanPoolMemoryCompressor
from tinymem.memory.recurrent_memory import RecurrentMemoryBank
from tinymem.model.config import ModelConfig
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.memory_input import AttentionMemory
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.tokenization.byte_tokenizer import ByteTokenizer


def make_decoder() -> SegmentedContinuousDecoder:
    config = ModelConfig(
        vocab_size=260,
        d_model=8,
        n_layers=1,
        n_heads=2,
        d_ff=16,
        max_local_tokens=16,
        dropout=0.0,
    )
    return SegmentedContinuousDecoder(
        DecoderOnlyTransformer(config),
        MeanPoolMemoryCompressor(8),
        RecurrentMemoryBank(capacity=2, model_width=8),
        segment_length=8,
    )


def make_example(
    question_id: str,
    answer: str,
    *,
    answer_message_marked: bool = True,
) -> LongMemEvalExample:
    return LongMemEvalExample(
        question_id=question_id,
        question_type="knowledge-update",
        question="Which color is current?",
        answer=answer,
        question_date="2024-01-03",
        sessions=(
            LongMemSession(
                "s1",
                "2024-01-01",
                (LongMemMessage("user", "It was red.", False),),
            ),
            LongMemSession(
                "s2",
                "2024-01-02",
                (
                    LongMemMessage(
                        "user",
                        f"Now it is {answer}.",
                        answer_message_marked,
                    ),
                ),
            ),
        ),
        answer_session_ids=("s2",),
    )


def empty_prompt_state(tokenizer: ByteTokenizer) -> LongMemEvalPromptState:
    next_logits = torch.full((1, tokenizer.vocab_size), -10.0)
    next_logits[0, ord("A")] = 10.0
    memory = AttentionMemory(
        values=torch.zeros(1, 2, 8),
        valid=torch.zeros(1, 2, dtype=torch.bool),
        positions=torch.full((1, 2), -1, dtype=torch.long),
    )
    return LongMemEvalPromptState(
        memory=memory,
        position=1,
        next_logits=next_logits,
        context_bytes=1,
        continuation_memory=memory,
        continuation_position=0,
        continuation_ids=(ord("?"),),
    )


def test_diagnostic_prompts_expose_only_the_named_information() -> None:
    example = make_example("q1", "blue")

    full = format_longmemeval_diagnostic_prompt(example, "full_oracle")
    evidence = format_longmemeval_diagnostic_prompt(example, "answer_messages")
    query_only = format_longmemeval_diagnostic_prompt(example, "query_only")
    copy = format_longmemeval_diagnostic_prompt(example, "answer_copy")

    assert "It was red" in full
    assert "Now it is blue" in full
    assert "It was red" not in evidence
    assert "Now it is blue" in evidence
    assert "It was red" not in query_only
    assert "Now it is blue" not in query_only
    assert "Text: blue" in copy


def test_candidate_scoring_uses_prompt_next_token_logits() -> None:
    tokenizer = ByteTokenizer()
    state = empty_prompt_state(tokenizer)

    correct = score_candidate_answer(
        make_decoder(), tokenizer, state, "A", device="cpu"
    )
    wrong = score_candidate_answer(
        make_decoder(), tokenizer, state, "B", device="cpu"
    )

    assert correct.mean_nll < wrong.mean_nll
    assert correct.first_byte_correct
    assert correct.greedy_prefix_bytes == 1
    assert math.isfinite(correct.byte_perplexity)


def test_candidate_scoring_aligns_multiple_bytes_across_segments() -> None:
    torch.manual_seed(11)
    decoder = make_decoder()
    tokenizer = ByteTokenizer()
    state = empty_prompt_state(tokenizer)
    candidate = "ABCDEFGHIJKLM"
    targets = torch.tensor(tokenizer.encode(candidate), dtype=torch.long)

    score = score_candidate_answer(
        decoder,
        tokenizer,
        state,
        candidate,
        device="cpu",
    )

    sequential_logits = [state.next_logits]
    for target_index in range(1, targets.numel()):
        continuation = torch.cat(
            (
                torch.tensor(state.continuation_ids, dtype=torch.long),
                targets[:target_index],
            )
        ).unsqueeze(0)
        output = decoder(
            continuation,
            torch.ones_like(continuation, dtype=torch.bool),
            initial_memory=state.continuation_memory,
            position_offset=state.continuation_position,
        )
        sequential_logits.append(output.logits[:, -1])
    expected_logits = torch.stack(sequential_logits, dim=1)
    expected_nll = F.cross_entropy(
        expected_logits.squeeze(0),
        targets,
        reduction="sum",
    )
    matches = expected_logits.argmax(dim=-1).squeeze(0).eq(targets)
    mismatches = (~matches).nonzero(as_tuple=False)
    expected_prefix = int(mismatches[0, 0]) if mismatches.numel() else len(candidate)

    assert score.total_nll == pytest.approx(float(expected_nll.detach()))
    assert score.first_byte_correct == bool(matches[0])
    assert score.greedy_prefix_bytes == expected_prefix


def test_candidate_scoring_preserves_the_final_prompt_segment() -> None:
    torch.manual_seed(13)
    decoder = make_decoder()
    tokenizer = ByteTokenizer()
    prompt = "prompt-crossing-a-segment:"
    candidate = "answer-crossing-a-segment"
    state = stream_longmemeval_prompt(
        decoder,
        tokenizer,
        prompt,
        device="cpu",
        chunk_tokens=16,
    )

    score = score_candidate_answer(
        decoder,
        tokenizer,
        state,
        candidate,
        device="cpu",
    )

    prompt_ids = tokenizer.encode(prompt)
    candidate_ids = tokenizer.encode(candidate)
    sequence = torch.tensor(
        prompt_ids + candidate_ids[:-1],
        dtype=torch.long,
    ).unsqueeze(0)
    output = decoder(sequence, torch.ones_like(sequence, dtype=torch.bool))
    expected_logits = output.logits[
        :,
        len(prompt_ids) - 1 : len(prompt_ids) + len(candidate_ids) - 1,
    ]
    expected_nll = F.cross_entropy(
        expected_logits.squeeze(0),
        torch.tensor(candidate_ids, dtype=torch.long),
        reduction="sum",
    )

    assert score.total_nll == pytest.approx(float(expected_nll.detach()))


def test_diagnostics_return_all_condition_metrics_deterministically() -> None:
    torch.manual_seed(7)
    decoder = make_decoder()
    examples = [make_example("q1", "blue"), make_example("q2", "green")]

    first = evaluate_longmemeval_diagnostics(
        decoder,
        examples,
        device="cpu",
        max_new_tokens=2,
        chunk_tokens=16,
    )
    second = evaluate_longmemeval_diagnostics(
        decoder,
        examples,
        device="cpu",
        max_new_tokens=2,
        chunk_tokens=16,
    )

    assert tuple(result.condition for result in first) == DIAGNOSTIC_CONDITIONS
    assert [result.to_dict() for result in first] == [
        result.to_dict() for result in second
    ]
    assert all(result.count == 2 for result in first)
    assert all(math.isfinite(result.answer_byte_nll) for result in first)
    assert first[-1].reference_inserted
    assert first[1].uses_answer_message_labels
    assert not first[2].uses_oracle_evidence


def test_answer_message_condition_reports_missing_label_coverage() -> None:
    examples = [
        make_example("q1", "blue"),
        make_example("q2", "green"),
        make_example("q3", "yellow", answer_message_marked=False),
    ]

    results = evaluate_longmemeval_diagnostics(
        make_decoder(),
        examples,
        device="cpu",
        max_new_tokens=2,
        chunk_tokens=16,
    )
    by_condition = {result.condition: result for result in results}

    assert by_condition["full_oracle"].count == 3
    assert by_condition["full_oracle"].skipped_missing_answer_message_labels == 0
    assert by_condition["answer_messages"].count == 2
    assert by_condition["answer_messages"].source_count == 3
    assert (
        by_condition["answer_messages"].skipped_missing_answer_message_labels
        == 1
    )


def test_diagnostics_require_distinct_answers() -> None:
    examples = [make_example("q1", "blue"), make_example("q2", "blue")]

    with pytest.raises(ValueError, match="distinct answers"):
        evaluate_longmemeval_diagnostics(
            make_decoder(),
            examples,
            device="cpu",
            max_new_tokens=2,
            chunk_tokens=16,
        )
