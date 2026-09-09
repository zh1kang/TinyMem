"""Exercise one complete local arm with tiny random weights and synthetic histories."""
from dataclasses import replace
import json

import pytest
import torch

from test_memory_updates import episode
from test_readout_runner import tiny_reader
from tinymem.research.readout_runner import encode_before
from tinymem.research.readout_experiment import run_arm, verify_run
from tinymem.research.update_protocol import file_sha256


def inputs(reader):
    return {split: tuple(encode_before(reader, episode(i)) for i in indices)
            for split, indices in (("train", (0, 1)), ("development", (2, 3)))}


def test_complete_arm_binds_inputs_and_reload_results(tiny_reader, tmp_path):
    rows = inputs(tiny_reader)
    output = tmp_path / "affine_seed_17"
    result = run_arm(tiny_reader, rows, output, kind="affine", seed=17, steps=2,
                     learning_rate=0.001, weight_decay=0.01, max_new_tokens=1,
                     input_identity={"evidence_kind": "tiny_random_cpu_test"})
    assert verify_run(output) == result
    assert result["kind"] == "readout_arm_complete_v1"
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["steps"] == 2
    assert len(protocol["schedule"]) == 2
    assert protocol["checkpoint_selection"] == "final_only"
    assert protocol["persistent_bytes"] == 66
    assert protocol["splits"] == ["train", "development"]
    assert protocol["input_identity"]["evidence_kind"] == "tiny_random_cpu_test"
    assert protocol["reader_parameters_sha256"] == result["reader_parameters_sha256"]
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in metrics] == [1, 2]
    predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    # 2 phases * 2 splits * 2 histories * 10 queries * 4 state controls + full text once per split.
    assert len(predictions) == 360
    assert {r["phase"] for r in predictions} == {"initial", "final", "reference"}
    assert result["prediction_count"] == len(predictions)
    assert set(result["files"]) == {"protocol.json", "encodings.json", "initial.safetensors",
                                    "final.safetensors", "metrics.jsonl", "predictions.jsonl"}
    assert all(file_sha256(output / name) == digest for name, digest in result["files"].items())
    assert all(p.grad is None and not p.requires_grad for p in tiny_reader.model.parameters())
    with pytest.raises(FileExistsError):
        run_arm(tiny_reader, rows, output, kind="affine", seed=17, steps=2,
                learning_rate=0.001, weight_decay=0.01, max_new_tokens=1,
                input_identity={"evidence_kind": "tiny_random_cpu_test"})
    with (output / "metrics.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="identity"):
        verify_run(output)


@pytest.mark.parametrize("defect", ["overlap", "steps", "learning_rate", "weight_decay", "seed", "token", "identity"])
def test_invalid_run_has_no_output(tiny_reader, tmp_path, defect):
    rows = inputs(tiny_reader)
    options = dict(kind="gelu", seed=17, steps=1, learning_rate=0.001,
                   weight_decay=0.01, max_new_tokens=1, input_identity={"test": True})
    if defect == "overlap":
        rows["development"] = rows["train"]
    elif defect == "token":
        rows["development"] = (replace(rows["development"][0], history_ids=(True,)), rows["development"][1])
    elif defect == "identity":
        options["input_identity"] = {}
    else:
        options[defect] = {"steps": True, "learning_rate": float("nan"),
                           "weight_decay": -1, "seed": -1}[defect]
    with pytest.raises(ValueError):
        run_arm(tiny_reader, rows, tmp_path / "run", **options)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("within_history", [False, True])
def test_source_question_alias_is_rejected_before_forward(tiny_reader, tmp_path, within_history):
    rows = inputs(tiny_reader)
    target = rows["development"][0]
    original = target.source_case_ids[0] if within_history else rows["train"][0].source_case_ids[0]
    alias = original.rsplit(":question-", 1)[0] + ":question-2"
    rows["development"] = (replace(target, source_case_ids=(target.source_case_ids[0], alias)),
                           rows["development"][1])
    output = tmp_path / "alias"
    with pytest.raises(ValueError, match="source"):
        run_arm(tiny_reader, rows, output, kind="affine", seed=17, steps=1,
                learning_rate=0.001, weight_decay=0.01, max_new_tokens=1,
                input_identity={"test": True})
    assert not output.exists()


@pytest.mark.parametrize("key", ["source_group_ids", "source_case_ids", "source_context_sha256"])
def test_source_overlap_is_rejected_before_forward(tiny_reader, tmp_path, key):
    rows = inputs(tiny_reader)
    target = rows["development"][0]
    values = (getattr(rows["train"][0], key)[0], getattr(target, key)[1])
    rows["development"] = (replace(target, **{key: values}), rows["development"][1])

    def forbidden(*args, **kwargs):
        pytest.fail("source overlap reached the reader")

    handle = tiny_reader.model.register_forward_pre_hook(forbidden)
    try:
        with pytest.raises(ValueError, match="source"):
            run_arm(tiny_reader, rows, tmp_path / "overlap", kind="affine", seed=17, steps=1,
                    learning_rate=0.001, weight_decay=0.01, max_new_tokens=1,
                    input_identity={"test": True})
    finally:
        handle.remove()
    assert not (tmp_path / "overlap").exists()


def test_run_preserves_rng_without_seeding_accelerators(tiny_reader, tmp_path, monkeypatch):
    rows = inputs(tiny_reader)
    rng = torch.random.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        pytest.fail("CPU initialization seeded an accelerator")

    monkeypatch.setattr(torch.cuda, "manual_seed_all", forbidden)
    monkeypatch.setattr(torch.mps, "manual_seed", forbidden)
    run_arm(tiny_reader, rows, tmp_path / "rng", kind="gelu", seed=17, steps=1,
            learning_rate=0.001, weight_decay=0.01, max_new_tokens=1,
            input_identity={"test": True})
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_failure_never_seals_partial_run(tiny_reader, tmp_path, monkeypatch):
    from tinymem.research import readout_experiment
    rows = inputs(tiny_reader)

    def fail(*args, **kwargs):
        raise RuntimeError("simulated training failure")

    monkeypatch.setattr(readout_experiment, "train_readout_step", fail)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="simulated"):
        run_arm(tiny_reader, rows, output, kind="gelu", seed=17, steps=1,
                learning_rate=0.001, weight_decay=0.01, max_new_tokens=1,
                input_identity={"test": True})
    assert (output / "initial.safetensors").is_file()
    assert not (output / "complete.json").exists()
    with pytest.raises(FileNotFoundError):
        verify_run(output)
