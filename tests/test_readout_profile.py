"""Compute profiling exercises training and evaluation without reusable weights."""

import json

import pytest
import torch

from test_readout_experiment import inputs
from test_readout_runner import tiny_reader
from tinymem.research.readout_experiment import profile_arm, verify_profile


@pytest.mark.parametrize("reader_dtype", [torch.float32, torch.bfloat16])
def test_profile_is_sealed_and_contains_no_reusable_checkpoint(tmp_path, tiny_reader, reader_dtype):
    tiny_reader.model.to(dtype=reader_dtype)
    output = tmp_path / "profile"
    result = profile_arm(
        tiny_reader,
        inputs(tiny_reader)["train"],
        output,
        kind="gelu",
        seed=17,
        steps=2,
        learning_rate=.001,
        weight_decay=.01,
        max_new_tokens=1,
        input_identity={"evidence_kind": "tiny_random_cpu_test"},
    )

    assert verify_profile(output) == result
    assert result["kind"] == "readout_compute_profile_v1"
    assert result["checkpoint_reuse"] is False
    assert result["training_steps"] == 2
    assert result["evaluation_histories"] == 2
    assert result["training_peak_memory_bytes"] is None
    assert set(result["files"]) == {"protocol.json", "schedule.json", "metrics.jsonl", "evaluation.json"}
    assert not list(output.glob("*.safetensors"))
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in metrics] == [1, 2]
    assert all(row["seconds"] >= 0 for row in metrics)
    evaluation = json.loads((output / "evaluation.json").read_text())
    assert evaluation["readout_predictions"] == 80
    assert evaluation["full_text_predictions"] == 20
    assert evaluation["seconds"] >= 0


def test_profile_requires_fresh_output_and_detects_damage(tmp_path, tiny_reader):
    output = tmp_path / "profile"
    options = dict(
        kind="affine", seed=17, steps=1, learning_rate=.001,
        weight_decay=.01, max_new_tokens=1,
        input_identity={"evidence_kind": "tiny_random_cpu_test"},
    )
    profile_arm(tiny_reader, inputs(tiny_reader)["train"], output, **options)
    with pytest.raises(FileExistsError):
        profile_arm(tiny_reader, inputs(tiny_reader)["train"], output, **options)
    with (output / "metrics.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="identity"):
        verify_profile(output)
