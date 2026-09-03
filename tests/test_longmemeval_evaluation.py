import torch

from tinymem.data.longmemeval import (
    LongMemEvalExample,
    LongMemMessage,
    LongMemSession,
)
from tinymem.evaluation.longmemeval import (
    answer_token_f1,
    evaluate_longmemeval,
    format_longmemeval_prompt,
    is_abstention,
    normalized_answer,
)
from tinymem.memory.continuous import MeanPoolMemoryCompressor
from tinymem.memory.recurrent_memory import RecurrentMemoryBank
from tinymem.model.config import ModelConfig
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.transformer import DecoderOnlyTransformer


def make_example() -> LongMemEvalExample:
    return LongMemEvalExample(
        question_id="q1",
        question_type="knowledge-update",
        question="Which color is current?",
        answer="blue",
        question_date="2024-01-03",
        sessions=(
            LongMemSession(
                "s1",
                "2024-01-01",
                (LongMemMessage("user", "It was red.", True),),
            ),
            LongMemSession(
                "s2",
                "2024-01-02",
                (LongMemMessage("user", "Now it is blue.", True),),
            ),
        ),
        answer_session_ids=("s2",),
    )


def make_decoder() -> SegmentedContinuousDecoder:
    config = ModelConfig(
        vocab_size=260,
        d_model=8,
        n_layers=1,
        n_heads=2,
        d_ff=16,
        max_local_tokens=8,
        dropout=0.0,
    )
    return SegmentedContinuousDecoder(
        DecoderOnlyTransformer(config),
        MeanPoolMemoryCompressor(8),
        RecurrentMemoryBank(capacity=2, model_width=8),
        segment_length=8,
    )


def test_longmemeval_prompt_preserves_session_order_without_answer() -> None:
    prompt = format_longmemeval_prompt(make_example())

    assert prompt.index("It was red") < prompt.index("Now it is blue")
    assert prompt.endswith("User: Which color is current?\nAssistant:")
    assert "answer_session_ids" not in prompt


def test_longmemeval_normalization_and_token_f1() -> None:
    assert normalized_answer("  Blue, CAR! ") == "blue car"
    assert answer_token_f1("blue car", "the blue car") == 0.8
    assert is_abstention("I cannot answer from this information.")
    assert not is_abstention("blue")


def test_longmemeval_evaluation_returns_category_metrics() -> None:
    result = evaluate_longmemeval(
        make_decoder(),
        [make_example()],
        device="cpu",
        max_new_tokens=2,
        chunk_tokens=16,
    )

    assert result.overall.count == 1
    assert set(result.by_question_type) == {"knowledge-update"}
    assert len(result.predictions) == 1
    assert result.predictions[0].context_bytes > 0
    assert torch.isfinite(torch.tensor(result.predictions[0].token_f1))
