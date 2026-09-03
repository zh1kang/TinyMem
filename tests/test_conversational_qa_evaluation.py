import math

import torch

from tinymem.data.schema import ReasoningExample
from tinymem.evaluation.conversational_qa import evaluate_conversational_qa
from tinymem.memory.continuous import MeanPoolMemoryCompressor
from tinymem.memory.recurrent_memory import RecurrentMemoryBank
from tinymem.model.config import ModelConfig
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.tokenization.byte_tokenizer import ByteTokenizer
from tinymem.training.conversational_qa import encode_conversational_qa_example


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


def make_example(source_id: str, task_id: str, answer: str) -> ReasoningExample:
    context = f"The value is {answer}."
    return ReasoningExample(
        dataset="babi",
        task_id=task_id,
        split="validation",
        context=context,
        question="What is the value?",
        answer=answer,
        supporting_fact_ids=(1,),
        source_length=len(context),
        source_example_id=source_id,
        context_fact_ids=(1,),
    )


def test_conversational_evaluation_reports_overall_and_task_metrics() -> None:
    tokenizer = ByteTokenizer()
    examples = [
        encode_conversational_qa_example(
            make_example("one", "qa1", "blue"), tokenizer
        ),
        encode_conversational_qa_example(
            make_example("two", "qa2", "green"), tokenizer
        ),
    ]
    decoder = make_decoder()
    decoder.train()

    result = evaluate_conversational_qa(
        decoder,
        examples,
        device="cpu",
        max_new_tokens=2,
        chunk_tokens=16,
    )

    assert result.overall.count == 2
    assert set(result.by_task) == {"babi:qa1", "babi:qa2"}
    assert len(result.predictions) == 2
    assert math.isfinite(result.overall.answer_byte_nll)
    assert decoder.training
