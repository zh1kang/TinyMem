"""Synthetic artifact integration; tiny Qwen does not establish reader quality."""
import json
from pathlib import Path

import pytest

from test_update_runner import tiny
from test_update_protocol import data_tree, input_tree
from tinymem.research import update_experiment as experiment
from tinymem.research.update_protocol import load_development_data


@pytest.fixture
def world(data_tree, tiny, monkeypatch):
    reader, _ = tiny
    tmp_path, directory = data_tree
    data = load_development_data(directory, root=tmp_path)
    # Scope artifact paths to the temporary root; source/runtime capture still
    # uses real repository files through the original execution function.
    actual_execution = experiment.execution
    recorded = actual_execution(reader)
    monkeypatch.setattr(experiment, "REPOSITORY", tmp_path)
    monkeypatch.setattr(experiment, "execution", lambda reader: recorded)
    original_hash = experiment.file_sha256
    repository = Path(__file__).resolve().parents[1]
    def digest(path):
        if path.is_relative_to(tmp_path) and str(path.relative_to(tmp_path)) in experiment.SOURCES:
            return original_hash(repository / path.relative_to(tmp_path))
        return original_hash(path)
    monkeypatch.setattr(experiment, "file_sha256", digest)
    return reader, data, {"synthetic_reader": True}, tmp_path


def scripted_qualification(world, monkeypatch):
    reader, data, identity, root = world
    original = experiment.evaluate_update_episode
    def scripted(shared_reader, encoded, method, **kwargs):
        if method != "full_context":
            return original(shared_reader, encoded, method, **kwargs)
        # Explicit oracle fixture only for testing gate success. Actual random
        # Qwen qualification failure is independently exercised below.
        cases = (encoded.episode.before, *(b.queries for b in encoded.episode.branches))
        return {"episode_id": encoded.episode.episode_id, "method": method,
                "predictions": [{"case_id": c.case_id, "prediction": c.answer} for stage in cases for c in stage]}
    with monkeypatch.context() as patch:
        patch.setattr(experiment, "evaluate_update_episode", scripted)
        gate = experiment.qualify(reader, data, identity, root / "qualification")
    assert gate["passed"]
    return root / "qualification"


def test_actual_random_reader_fails_gate_and_cannot_launch(world):
    reader, data, identity, root = world
    result = experiment.qualify(reader, data, identity, root / "qualification")
    assert not result["passed"]
    with pytest.raises(ValueError, match="qualification failed"):
        experiment.freeze_launch(reader, data, identity, root / "qualification", root / "launch", steps=1)
    assert not (root / "launch").exists()


def test_six_actual_tiny_training_runs_and_completion_boundary(world, monkeypatch):
    reader, data, identity, root = world
    qualification = scripted_qualification(world, monkeypatch)
    launch_dir = root / "launch"
    launch = experiment.freeze_launch(reader, data, identity, qualification, launch_dir, steps=1)
    assert launch["evidence_kind"] == "synthetic_fixture"
    assert len(launch["runs"]) == 6 and launch["confirmation_opened"] is False
    original_open = Path.open
    def guarded(path, *args, **kwargs):
        if path.name == "confirmation.json":
            pytest.fail("incomplete runs must fail before confirmation access")
        return original_open(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", guarded)
        with pytest.raises(FileNotFoundError):
            experiment.confirmation_episodes(launch_dir, data, launch)
    for run_index, run in enumerate(launch["runs"]):
        result = experiment.train(reader, data, identity, launch_dir, run["method"], run["seed"])
        assert result["completed_steps"] == 1 and result["evidence_kind"] == "synthetic_fixture"
        checkpoint = experiment.verify_training(launch_dir, launch, data, run["method"], run["seed"])
        assert checkpoint.is_file()
        if run_index == 4:
            with monkeypatch.context() as patch:
                patch.setattr(Path, "open", guarded)
                with pytest.raises(FileNotFoundError):
                    experiment.confirmation_episodes(launch_dir, data, launch)
                with pytest.raises(FileNotFoundError):
                    experiment.evaluate(reader, data, identity, launch_dir, "no_memory", split="confirmation")
    with pytest.raises(FileExistsError):
        experiment.train(reader, data, identity, launch_dir, "query_pool", 1337)
    # Actual confirmation authorization, replay, final safetensors loading,
    # generation, scoring and output seal using fresh synthetic source groups.
    heldout = experiment.confirmation_episodes(launch_dir, data, launch)
    assert len(heldout) == 1 and heldout[0].episode_id.endswith("confirmation:0000")
    result = experiment.evaluate(reader, data, identity, launch_dir, "query_pool", seed=1337, split="confirmation")
    assert result["histories"] == 1
    for method in ("recent_native", "latest_vocabulary", "latest_template", "fingerprint", "no_memory", "full_context"):
        result = experiment.evaluate(reader, data, identity, launch_dir, method, split="confirmation")
        assert result["histories"] == 1
    # A valid completed checkpoint cannot be adopted by another launch.
    second = root / "second-launch"
    experiment.freeze_launch(reader, data, identity, qualification, second, steps=2)
    import shutil
    shutil.copytree(launch_dir / "runs", second / "runs")
    with pytest.raises(ValueError):
        experiment.verify_training(second, experiment.read_json(second / "protocol.json"), data, "query_pool", 1337)
    first = launch_dir / launch["runs"][0]["directory"] / "metrics.jsonl"
    first.write_text(first.read_text() + " ")
    with pytest.raises(ValueError, match="identity changed"):
        experiment.verify_training(launch_dir, launch, data, "query_pool", 1337)


def test_qualification_and_launch_tampering_and_wrong_runtime(world, monkeypatch):
    reader, data, identity, root = world
    qualification = scripted_qualification(world, monkeypatch)
    output = root / "launch"
    experiment.freeze_launch(reader, data, identity, qualification, output, steps=2)
    base = experiment.provenance(data, identity, reader)
    with pytest.raises(ValueError, match="runtime"):
        experiment.verify_launch(output, data, {**base, "reader": {"different": True}})
    schedules = output / "schedules.json"
    schedules.write_text("{}\n")
    with pytest.raises(ValueError, match="identity changed"):
        experiment.verify_launch(output, data, base)
    with pytest.raises(FileExistsError):
        experiment.qualify(reader, data, identity, qualification)


def test_partial_qualification_and_invalid_step_count_fail(world):
    reader, data, identity, root = world
    partial = root / "partial"
    partial.mkdir()
    with pytest.raises(FileNotFoundError):
        experiment.freeze_launch(reader, data, identity, partial, root / "launch", steps=1)
    for steps in (True, 0, -1, 1.5):
        with pytest.raises(ValueError, match="positive"):
            experiment.freeze_launch(reader, data, identity, partial, root / "launch", steps=steps)
    assert not (root / "launch").exists()


def test_frozen_peft_adapter_preserves_recurrent_gradients(tiny):
    import torch
    peft = pytest.importorskip("peft")
    from tinymem.research.update_runner import new_update_writer, train_update_step
    reader, encoded = tiny
    reader.model = peft.get_peft_model(reader.model, peft.LoraConfig(
        r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM", lora_dropout=0.0))
    reader.model.requires_grad_(False)
    reader.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reader.model.train()
    writer = new_update_writer(reader, "query_pool", 1337)
    before = {key: value.clone() for key, value in reader.model.state_dict().items()}
    result = train_update_step(reader, writer, encoded, torch.optim.AdamW(writer.parameters()), trace_gradients=True)
    assert len(result["state_gradient_norms"]) == 7
    assert all(value > 0 for value in result["state_gradient_norms"])
    assert all(torch.equal(before[key], value) for key, value in reader.model.state_dict().items())
    assert all(parameter.grad is None for parameter in reader.model.parameters())


def test_profile_uses_only_training_and_cannot_be_a_training_completion(world):
    from dataclasses import replace
    reader, data, identity, root = world
    # No development or confirmation data are supplied to this compute path.
    result = experiment.profile(reader, replace(data, development=()), identity, root / "profile", "query_pool")
    assert result["kind"] == "compute_profile_not_training" and result["checkpoint_reuse"] is False
    metrics = experiment.read_rows(root / "profile" / "metrics.jsonl")
    assert len(metrics) == 10 and all(len(row["state_gradient_norms"]) == 7 for row in metrics)
    assert not list((root / "profile").glob("*.safetensors"))
    assert not list((root / "profile").glob("*predictions*"))


def test_intermediate_checkpoint_coverage_with_stubbed_optimizer_step(world, monkeypatch):
    # Exercise the actual persistence loop at 100 and final101 without treating
    # 101 stubbed updates as model learning. Other tests use real optimizer steps.
    reader, data, identity, root = world
    qualification = scripted_qualification(world, monkeypatch)
    launch_dir = root / "launch"
    launch = experiment.freeze_launch(reader, data, identity, qualification, launch_dir, steps=101)
    monkeypatch.setattr(experiment, "train_update_step", lambda *args, **kwargs: {"fixture_stub": True})
    experiment.train(reader, data, identity, launch_dir, "query_pool", 1337)
    checkpoint = experiment.verify_training(launch_dir, launch, data, "query_pool", 1337)
    assert checkpoint.name == "step_000101.safetensors"
    complete = experiment.read_json(checkpoint.parent / "complete.json")
    assert {"step_000100.safetensors", "optimizer_000100.pt", "step_000101.safetensors", "optimizer_000101.pt"} <= set(complete["files"])


def test_cli_requires_explicit_split_and_rejects_synthetic_data(world, monkeypatch):
    from scripts import run_memory_updates as cli
    reader, data, identity, root = world
    with pytest.raises(SystemExit):
        cli.main(["--data", "unused", "evaluate", "--launch", "unused", "--method", "full_context"])
    monkeypatch.setattr(cli, "load_development_data", lambda path: data)
    monkeypatch.setattr(cli, "shared_reader_identity", lambda: identity)
    def forbidden(*args, **kwargs):
        pytest.fail("synthetic CLI data must be rejected before model loading")
    monkeypatch.setattr(cli, "load_shared_reader", forbidden)
    with pytest.raises(SystemExit):
        cli.main(["--data", "unused", "qualify", "--output", "unused"])
