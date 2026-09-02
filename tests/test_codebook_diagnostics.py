import math

import pytest
import torch

from tinymem.evaluation.codebook import (
    collect_codebook_traces,
    continuous_memory_bytes,
    discrete_memory_budget,
    evaluate_codebook_diagnostics,
    matched_discrete_capacity,
    summarize_code_attributes,
    summarize_codebook,
)
from tinymem.memory.discrete_compressor import DiscreteMemoryCompressor
from tinymem.memory.recurrent_memory import RecurrentMemoryBank
from tinymem.model.config import ModelConfig
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.training.controlled_qa import EncodedQAExample


def test_codebook_diagnostics_detect_collapse_and_distortion() -> None:
    indices = torch.tensor([[0, 0, 1], [1, -1, -1]])
    valid = indices >= 0
    probabilities = torch.tensor(
        [
            [[0.8, 0.2, 0.0, 0.0], [0.7, 0.3, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0]],
            [[0.2, 0.8, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        ]
    )
    prequantized = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    quantized = prequantized + valid.unsqueeze(-1) * 0.5

    result = summarize_codebook(
        indices,
        valid,
        probabilities,
        codebook_size=4,
        prequantized=prequantized,
        quantized=quantized,
    )

    assert result.assignments == 4
    assert result.active_codes == 2
    assert result.unused_codes == 2
    assert result.unused_fraction == 0.5
    assert result.most_common_fraction == 0.5
    assert result.hard_perplexity == pytest.approx(2.0)
    assert result.quantization_mse == pytest.approx(0.25)
    assert result.counts == (2, 2, 0, 0)
    assert result.mean_soft_entropy > 0
    assert result.aggregate_soft_perplexity > 1


def test_code_attribute_diagnostics_measure_dependence() -> None:
    indices = torch.tensor([[0, 0, 1, 1]])
    valid = torch.ones_like(indices, dtype=torch.bool)
    attributes = torch.tensor([[0, 0, 1, 1]])

    result = summarize_code_attributes(
        indices,
        valid,
        attributes,
        codebook_size=2,
        attribute_count=2,
    )

    assert result.assignments == 4
    assert result.active_attributes == 2
    assert result.mutual_information_nats == pytest.approx(math.log(2))
    assert result.joint_counts == ((2, 0), (0, 2))


def test_discrete_and_continuous_budget_accounting() -> None:
    continuous_bytes = continuous_memory_bytes(
        capacity=12,
        model_width=64,
        element_size=4,
    )
    capacity = matched_discrete_capacity(continuous_bytes)
    discrete = discrete_memory_budget(
        capacity=capacity,
        codebook_size=256,
        model_width=64,
        element_size=4,
    )

    assert continuous_bytes == 3180
    assert capacity == 187
    assert discrete.code_bits == 8
    assert discrete.tensor_bytes == 3179
    assert discrete.tensor_bytes <= continuous_bytes
    assert continuous_bytes - discrete.tensor_bytes < 17
    assert discrete.logical_bytes == 1707
    assert discrete.shared_codebook_bytes == 65536


def test_codebook_diagnostics_reject_no_valid_assignments() -> None:
    with pytest.raises(ValueError, match="valid assignment"):
        summarize_codebook(
            torch.full((1, 2), -1),
            torch.zeros(1, 2, dtype=torch.bool),
            torch.zeros(1, 2, 4),
            codebook_size=4,
            prequantized=torch.zeros(1, 2, 3),
            quantized=torch.zeros(1, 2, 3),
        )


def test_evaluate_codebook_diagnostics_aggregates_decoder_traces() -> None:
    config = ModelConfig(
        vocab_size=12,
        d_model=8,
        n_layers=1,
        n_heads=2,
        d_ff=16,
        max_local_tokens=4,
    )
    decoder = SegmentedContinuousDecoder(
        DecoderOnlyTransformer(config),
        DiscreteMemoryCompressor(
            8,
            codebook_size=4,
            summary_slots=1,
        ),
        RecurrentMemoryBank(capacity=2, model_width=8),
        segment_length=2,
    )
    examples = [
        EncodedQAExample(
            input_ids=(1, 2, 3, 4),
            answer_id=4,
            source_example_id="a",
        ),
        EncodedQAExample(
            input_ids=(2, 3, 4),
            answer_id=4,
            source_example_id="b",
        ),
    ]

    result = evaluate_codebook_diagnostics(
        decoder,
        examples,
        batch_size=1,
        pad_id=0,
        device="cpu",
    )

    assert result.assignments == 4
    assert len(result.counts) == 4
    assert 1 <= result.active_codes <= 4
    assert result.quantization_mse >= 0

    traces = collect_codebook_traces(
        decoder,
        examples,
        batch_size=2,
        pad_id=0,
        device="cpu",
    )

    assert len(traces) == 2
    assert traces[0]["source_example_id"] == "a"
    assert len(traces[0]["proposed_codes"]) == 2
    assert len(traces[0]["writes_applied"]) == 2
    assert traces[0]["written_codes"] == traces[0]["proposed_codes"]
    assert traces[0]["final_memory_codes"]


@pytest.mark.parametrize("continuous_bytes", [True, 0, 16, 2.5, "17"])
def test_matched_discrete_capacity_rejects_invalid_budget(
    continuous_bytes: object,
) -> None:
    expected = (
        ValueError
        if isinstance(continuous_bytes, int)
        and not isinstance(continuous_bytes, bool)
        else TypeError
    )
    with pytest.raises(expected):
        matched_discrete_capacity(continuous_bytes)
