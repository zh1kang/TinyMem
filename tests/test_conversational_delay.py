import random

import pytest
import torch

from tinymem.data.schema import ReasoningExample
from tinymem.evaluation.conversational_delay import (
    FILLER_SESSION_HEADER,
    build_delayed_examples,
    format_delayed_prompt,
    sample_filler,
)
from tinymem.evaluation.conversational_qa import (
    MEMORY_CONDITIONS,
    evaluate_conversational_qa,
    resolve_memory_condition,
)
from tinymem.memory.continuous import MeanPoolMemoryCompressor
from tinymem.memory.recurrent_memory import RecurrentMemoryBank
from tinymem.model.config import ModelConfig
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.tokenization.byte_tokenizer import ByteTokenizer
from tinymem.training.conversational_qa import format_conversational_qa_prompt


def make_example() -> ReasoningExample:
    context = "Mary moved to the bathroom.\nJohn went to the hallway."
    return ReasoningExample(
        dataset="babi",
        task_id="qa1",
        split="test",
        context=context,
        question="Where is Mary?",
        answer="bathroom",
        supporting_fact_ids=(1,),
        source_length=len(context),
        source_example_id="qa1_test.txt:episode-000001:question-3",
        context_fact_ids=(1, 2),
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


def test_sample_filler_is_single_line_with_bounded_bytes() -> None:
    text = "alpha\nbeta gamma delta\nepsilon " * 20
    filler = sample_filler(text, byte_length=37, rng=random.Random(3))

    assert "\n" not in filler
    assert 34 <= len(filler.encode("utf-8")) <= 37


def test_sample_filler_trims_split_multibyte_characters() -> None:
    text = "é" * 50
    filler = sample_filler(text, byte_length=5, rng=random.Random(0))

    assert filler == "éé"
    assert len(filler.encode("utf-8")) == 4


def test_sample_filler_accepts_byte_length_larger_than_character_count() -> None:
    filler = sample_filler("ééé", byte_length=5, rng=random.Random(0))

    assert filler == "éé"
    assert len(filler.encode("utf-8")) == 4


def test_delayed_prompt_inserts_filler_before_question_only() -> None:
    example = make_example()
    base = format_conversational_qa_prompt(example)

    prompt = format_delayed_prompt(example, "some ordinary prose")

    assert prompt.startswith(base.split("[question | undated]\n")[0])
    assert prompt.endswith("[question | undated]\nUser: Where is Mary?\nAssistant:")
    assert FILLER_SESSION_HEADER + "User: some ordinary prose\n" in prompt
    assert prompt.count("[question | undated]") == 1
    with pytest.raises(ValueError, match="single line"):
        format_delayed_prompt(example, "two\nlines")


def test_build_delayed_examples_is_deterministic_and_adds_exact_bytes() -> None:
    tokenizer = ByteTokenizer()
    example = make_example()
    filler_text = "".join(chr(97 + index % 26) for index in range(5000))
    base = build_delayed_examples(
        [example],
        tokenizer,
        filler_text=filler_text,
        filler_bytes=0,
        seed=7,
    )[0]

    first = build_delayed_examples(
        [example],
        tokenizer,
        filler_text=filler_text,
        filler_bytes=256,
        seed=7,
    )[0]
    second = build_delayed_examples(
        [example],
        tokenizer,
        filler_text=filler_text,
        filler_bytes=256,
        seed=7,
    )[0]

    assert base.prompt_ids == tuple(
        tokenizer.encode(format_conversational_qa_prompt(example))
    )
    assert first == second
    assert first.answer_ids == base.answer_ids
    header_bytes = len((FILLER_SESSION_HEADER + "User: \n").encode("utf-8"))
    assert len(first.prompt_ids) == len(base.prompt_ids) + header_bytes + 256


def test_resolve_memory_condition_covers_every_name() -> None:
    for name in MEMORY_CONDITIONS:
        intervention, update_memory = resolve_memory_condition(name)
        assert update_memory is (name != "no_writes")
        assert (intervention is None) is (name in ("normal", "no_writes"))
    with pytest.raises(ValueError, match="memory condition"):
        resolve_memory_condition("shuffled")


def test_evaluate_conversational_qa_accepts_memory_conditions() -> None:
    torch.manual_seed(0)
    tokenizer = ByteTokenizer()
    decoder = make_decoder()
    examples = build_delayed_examples(
        [make_example()],
        tokenizer,
        filler_text="lorem ipsum " * 50,
        filler_bytes=32,
        seed=1,
    )

    for name in MEMORY_CONDITIONS:
        intervention, update_memory = resolve_memory_condition(name)
        evaluation = evaluate_conversational_qa(
            decoder,
            examples,
            device="cpu",
            max_new_tokens=3,
            chunk_tokens=8,
            query_memory_intervention=intervention,
            update_memory=update_memory,
        )
        assert evaluation.overall.count == 1


def test_mixed_delay_examples_delay_only_a_fraction() -> None:
    from tinymem.evaluation.conversational_delay import build_mixed_delay_examples

    tokenizer = ByteTokenizer()
    examples = [make_example()] * 200
    filler_text = "".join(chr(97 + index % 26) for index in range(10_000))
    base_length = len(
        build_delayed_examples(
            examples[:1],
            tokenizer,
            filler_text=filler_text,
            filler_bytes=0,
            seed=0,
        )[0].prompt_ids
    )

    mixed = build_mixed_delay_examples(
        examples,
        tokenizer,
        filler_text=filler_text,
        max_filler_bytes=300,
        delayed_fraction=0.5,
        seed=11,
    )
    repeat = build_mixed_delay_examples(
        examples,
        tokenizer,
        filler_text=filler_text,
        max_filler_bytes=300,
        delayed_fraction=0.5,
        seed=11,
    )

    delayed = [item for item in mixed if len(item.prompt_ids) > base_length]
    assert mixed == repeat
    assert 60 <= len(delayed) <= 140
    header_bytes = len((FILLER_SESSION_HEADER + "User: \n").encode("utf-8"))
    assert all(
        len(item.prompt_ids) - base_length - header_bytes <= 300 for item in delayed
    )
    assert all(item.answer_ids == mixed[0].answer_ids for item in mixed)
