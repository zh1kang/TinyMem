import pytest
import torch

from tinymem.data.babilong import parse_babilong_records
from tinymem.data.schema import ReasoningExample
from tinymem.data.vocabulary import ControlledVocabulary
from tinymem.evaluation.baseline_comparison import (
    BaselineResult,
    evaluate_qa1_baseline,
    require_equal_memory_budget,
)
from tinymem.model.config import ModelConfig
from tinymem.model.transformer import DecoderOnlyTransformer


def make_example() -> ReasoningExample:
    context = (
        "Mary moved to the garden. filler words keep growing until "
        "the answer fact leaves the local window entirely."
    )
    return parse_babilong_records(
        [{"input": context, "question": "Where is Mary? ", "target": "garden"}],
        task_id="qa1",
        split="test",
        source_name="fixture.json",
    )[0]


def make_model(vocab_size: int) -> DecoderOnlyTransformer:
    return DecoderOnlyTransformer(
        ModelConfig(
            vocab_size=vocab_size,
            d_model=8,
            n_layers=1,
            n_heads=2,
            d_ff=16,
            max_local_tokens=4,
        )
    ).eval()


def test_baselines_use_the_real_stream_and_report_equal_memory_bytes() -> None:
    example = make_example()
    vocabulary = ControlledVocabulary.from_texts(
        [example.context, example.question, example.answer]
    )
    model = make_model(len(vocabulary))

    recent = evaluate_qa1_baseline(
        model,
        vocabulary,
        [example],
        baseline="recent",
        capacity=10,
        batch_size=1,
        device="cpu",
        seed=7,
    )
    oracle = evaluate_qa1_baseline(
        model,
        vocabulary,
        [example],
        baseline="oracle",
        capacity=10,
        batch_size=1,
        device="cpu",
        seed=7,
    )

    assert recent.count == oracle.count == 1
    assert recent.outside_window_count == oracle.outside_window_count == 1
    assert recent.memory_bytes == oracle.memory_bytes
    assert require_equal_memory_budget([recent, oracle]) == recent.memory_bytes


def test_local_baseline_reports_zero_memory_without_changing_shared_budget() -> None:
    example = make_example()
    vocabulary = ControlledVocabulary.from_texts(
        [example.context, example.question, example.answer]
    )
    model = make_model(len(vocabulary))
    local = evaluate_qa1_baseline(
        model,
        vocabulary,
        [example],
        baseline="local",
        capacity=10,
        batch_size=1,
        device="cpu",
        seed=11,
    )
    recent = evaluate_qa1_baseline(
        model,
        vocabulary,
        [example],
        baseline="recent",
        capacity=10,
        batch_size=1,
        device="cpu",
        seed=11,
    )

    assert local.capacity == 0
    assert local.memory_bytes == 0
    assert require_equal_memory_budget([local, recent]) == recent.memory_bytes


def test_oracle_rejects_capacity_smaller_than_exact_fact() -> None:
    example = make_example()
    vocabulary = ControlledVocabulary.from_texts(
        [example.context, example.question, example.answer]
    )

    with pytest.raises(ValueError, match="oracle capacity"):
        evaluate_qa1_baseline(
            make_model(len(vocabulary)),
            vocabulary,
            [example],
            baseline="oracle",
            capacity=1,
            batch_size=1,
            device="cpu",
            seed=1,
        )


def test_equal_budget_check_rejects_mismatched_memory_costs() -> None:
    first = BaselineResult("recent", 2, 100, 0, 1, 0, 1, ())
    second = BaselineResult("oracle", 2, 104, 1, 1, 1, 1, ())

    with pytest.raises(ValueError, match="same byte budget"):
        require_equal_memory_budget([first, second])
