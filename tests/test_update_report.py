from copy import deepcopy

import pytest

from tinymem.evaluation.update_report import _states, verify_evaluation
from tinymem.research import update_experiment as experiment
from test_update_experiment import world, data_tree, input_tree, tiny, scripted_qualification


def reseal(directory, name, value):
    (directory / name).unlink()
    experiment.write_json(directory / name, value)
    complete = experiment.read_json(directory / "complete.json")
    complete["files"][name] = experiment.file_sha256(directory / name)
    (directory / "complete.json").unlink()
    experiment.write_json(directory / "complete.json", complete)


def test_report_independently_scores_raw_predictions_and_checks_bytes(world, monkeypatch):
    reader, data, identity, root = world
    qualification = scripted_qualification(world, monkeypatch)
    launch_dir = root / "launch"
    launch = experiment.freeze_launch(reader, data, identity, qualification, launch_dir, steps=1)
    experiment.evaluate(reader, data, identity, launch_dir, "latest_template", split="development")
    scored, _ = verify_evaluation(launch_dir, launch, data, data.development, "latest_template", None, "development")
    assert len(scored) == 1
    directory = launch_dir / "evaluations/development/latest_template"
    rows = experiment.read_rows(directory / "predictions.jsonl")
    original = deepcopy(rows)
    rows[0]["metrics"]["rates"]["before.known_accuracy"]["numerator"] = 999
    path = directory / "predictions.jsonl"
    path.unlink()
    with path.open("x") as handle:
        import json
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    complete = experiment.read_json(directory / "complete.json")
    complete["files"][path.name] = experiment.file_sha256(path)
    (directory / "complete.json").unlink()
    experiment.write_json(directory / "complete.json", complete)
    with pytest.raises(ValueError, match="cached metrics"):
        verify_evaluation(launch_dir, launch, data, data.development, "latest_template", None, "development")
    row = original[0]
    row["states"]["before"]["payload"][0][0] = 256
    with pytest.raises(ValueError, match="uint8"):
        _states(row, "latest_template")


def test_control_state_and_mismatched_evaluation_provenance_fail(world, monkeypatch):
    reader, data, identity, root = world
    qualification = scripted_qualification(world, monkeypatch)
    launch_dir = root / "launch"
    launch = experiment.freeze_launch(reader, data, identity, qualification, launch_dir, steps=1)
    experiment.evaluate(reader, data, identity, launch_dir, "no_memory", split="development")
    directory = launch_dir / "evaluations/development/no_memory"
    protocol = experiment.read_json(directory / "protocol.json")
    protocol["launch_complete_sha256"] = "other-launch"
    reseal(directory, "protocol.json", protocol)
    with pytest.raises(ValueError, match="provenance"):
        verify_evaluation(launch_dir, launch, data, data.development, "no_memory", None, "development")
    with pytest.raises(ValueError, match="control"):
        _states({"states": {"hidden": "memory"}, "persistent_bytes": 0}, "no_memory")
