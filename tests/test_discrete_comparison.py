import copy

import pytest

from tinymem.evaluation.discrete_comparison import (
    compare_discrete_to_continuous,
)


def make_result(*, compressor: str, memory_bytes: int) -> dict[str, object]:
    result = {
        "compressor": compressor,
        "memory_bytes_per_example": memory_bytes,
        "validation_evaluations": [
            {"intervention": "normal", "correct": 6, "count": 10}
        ],
        "delayed_validation_evaluations": [
            {"intervention": "normal", "correct": 4, "count": 10}
        ],
        "codebook_diagnostics": None,
    }
    for field in (
        "seed",
        "base_checkpoint_sha256",
        "training_steps",
        "memory_warmup_steps",
        "segment_length",
        "max_distributed_distractor_tokens",
        "validation_distributed_distractor_tokens",
        "write_gate_kernel_size",
        "training_examples",
        "validation_examples",
    ):
        result[field] = 1
    return result


def test_compare_discrete_to_continuous_requires_matched_bytes_and_protocol() -> None:
    continuous = make_result(compressor="multislot_attention", memory_bytes=3180)
    discrete = make_result(compressor="discrete", memory_bytes=3179)
    discrete["validation_evaluations"][0]["correct"] = 5

    comparison = compare_discrete_to_continuous(continuous, discrete)

    assert comparison["unused_byte_difference"] == 1
    assert comparison["comparisons"]["standard"]["continuous"]["accuracy"] == 0.6
    assert comparison["comparisons"]["standard"]["discrete"]["accuracy"] == 0.5
    assert comparison["comparisons"]["standard"][
        "discrete_minus_continuous_accuracy"
    ] == pytest.approx(-0.1)


def test_compare_discrete_to_continuous_rejects_protocol_mismatch() -> None:
    continuous = make_result(compressor="multislot_attention", memory_bytes=3180)
    discrete = make_result(compressor="discrete", memory_bytes=3179)
    discrete["seed"] = 2

    with pytest.raises(ValueError, match="seed"):
        compare_discrete_to_continuous(continuous, discrete)


def test_compare_discrete_to_continuous_rejects_unmatched_budget() -> None:
    continuous = make_result(compressor="multislot_attention", memory_bytes=3180)
    discrete = make_result(compressor="discrete", memory_bytes=3100)

    with pytest.raises(ValueError, match="code slot"):
        compare_discrete_to_continuous(continuous, discrete)

    oversized = copy.deepcopy(discrete)
    oversized["memory_bytes_per_example"] = 3181
    with pytest.raises(ValueError, match="exceeds"):
        compare_discrete_to_continuous(continuous, oversized)
