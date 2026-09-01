import pytest
import torch

from tinymem.data.babilong import parse_babilong_records
from tinymem.data.schema import ReasoningExample
from tinymem.data.vocabulary import ControlledVocabulary
from tinymem.evaluation.forgetting_curve import (
    delay_label,
    evaluate_local_forgetting_curve,
    qa1_answer_evidence,
    qa1_evidence_delay_tokens,
)
from tinymem.model.config import ModelConfig
from tinymem.model.streaming import StreamingDecoder
from tinymem.model.transformer import DecoderOnlyTransformer


def make_example() -> ReasoningExample:
    context = (
        "Mary moved to the garden. filler words. "
        "John went to the office. more filler. "
        "Mary travelled to the kitchen."
    )
    return parse_babilong_records(
        [{"input": context, "question": "Where is Mary? ", "target": "kitchen"}],
        task_id="qa1",
        split="test",
        source_name="fixture.json",
    )[0]


def test_qa1_delay_uses_the_last_queried_person_fact() -> None:
    example = make_example()
    vocabulary = ControlledVocabulary.from_texts(
        [example.context, example.question, example.answer]
    )

    evidence = qa1_answer_evidence(example)
    delay = qa1_evidence_delay_tokens(example, vocabulary)

    assert evidence.text == "Mary travelled to the kitchen."
    assert delay == 8


@pytest.mark.parametrize(
    ("delay", "label"),
    [
        (0, "0-32"),
        (32, "0-32"),
        (33, "33-64"),
        (128, "65-128"),
        (513, ">512"),
    ],
)
def test_delay_label_uses_inclusive_limits(delay: int, label: str) -> None:
    assert delay_label(delay) == label


def test_cached_streaming_is_not_raw_window_recomputation() -> None:
    torch.manual_seed(71)
    model = DecoderOnlyTransformer(
        ModelConfig(
            vocab_size=16,
            d_model=8,
            n_layers=2,
            n_heads=2,
            d_ff=16,
            max_local_tokens=4,
        )
    ).eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    stream = StreamingDecoder(model, segment_length=4)

    streamed_logits = torch.cat(
        (
            stream.process_segment(input_ids[:, :4]),
            stream.process_segment(input_ids[:, 4:]),
        ),
        dim=1,
    )
    final_window_logits = model(input_ids[:, -4:])

    assert not torch.allclose(streamed_logits[:, -1], final_window_logits[:, -1])


def test_forgetting_curve_counts_each_streamed_example() -> None:
    example = make_example()
    vocabulary = ControlledVocabulary.from_texts(
        [example.context, example.question, example.answer]
    )
    model = DecoderOnlyTransformer(
        ModelConfig(
            vocab_size=len(vocabulary),
            d_model=8,
            n_layers=1,
            n_heads=2,
            d_ff=16,
            max_local_tokens=4,
        )
    ).eval()

    results = evaluate_local_forgetting_curve(
        model,
        vocabulary,
        [example],
        batch_size=1,
        device="cpu",
    )

    assert sum(result.count for result in results) == 1
    assert sum(result.correct for result in results) in (0, 1)
