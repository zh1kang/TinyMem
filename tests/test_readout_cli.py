"""CLI preflight and dispatch use the real parser without production inference."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def cli():
    spec = importlib.util.spec_from_file_location("readout_cli", "scripts/run_readout_interface.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_does_not_load_reader(cli, monkeypatch, capsys):
    data = SimpleNamespace(train=(1, 2), development=(3, 4),
        protocol={"design": {"status": "data_and_measurement_design_not_a_training_launch_protocol"}})
    monkeypatch.setattr(cli, "load_development_data", lambda path: data)
    monkeypatch.setattr(cli, "shared_reader_identity", lambda: {"adapter": "test"})
    def forbidden(*args):
        pytest.fail("input check must not load a model or prepare a device")
    monkeypatch.setattr(cli, "load_shared_reader", forbidden)
    monkeypatch.setattr(cli, "prepare_device", forbidden)
    cli.main(["check", "--data", "artifacts/fixture"])
    assert json.loads(capsys.readouterr().out)["model_loaded"] is False


@pytest.mark.parametrize("option,value", [("--steps", "0"), ("--seed", "-1"),
    ("--learning-rate", "nan"), ("--learning-rate", "0"), ("--weight-decay", "-1"),
    ("--histories", "1"), ("--histories", "5")])
def test_bad_profile_options_fail_before_loading(cli, monkeypatch, option, value):
    def forbidden(*args):
        pytest.fail("invalid options must not load data")
    monkeypatch.setattr(cli, "load_development_data", forbidden)
    arguments = ["profile", "--data", "artifacts/fixture", "--device", "cpu",
        "--output", "artifacts/unused-profile-test", "--arm", "affine", "--seed", "17",
        "--steps", "4", "--learning-rate", ".001", "--weight-decay", ".01",
        "--max-new-tokens", "1", "--histories", "2"]
    arguments[arguments.index(option) + 1] = value
    with pytest.raises(SystemExit) as error:
        cli.main(arguments)
    assert error.value.code == 2


def test_profile_dispatch_only_encodes_training(cli, monkeypatch):
    train = [SimpleNamespace(history_ids=tuple(range(n)), history_id=str(n)) for n in (8, 2, 5, 12)]
    data = SimpleNamespace(train=train, development=("must-not-encode",), protocol_sha256="fixture",
        protocol={"design": {"status": "data_and_measurement_design_not_a_training_launch_protocol"}})
    monkeypatch.setattr(cli, "load_development_data", lambda path: data)
    monkeypatch.setattr(cli, "shared_reader_identity", lambda: {})
    monkeypatch.setattr(cli, "prepare_device", lambda device: device)
    monkeypatch.setattr(cli, "load_shared_reader", lambda *args: object())
    def encode(reader, row):
        assert row in train
        return row
    monkeypatch.setattr(cli, "encode_before", encode)
    def profile(reader, rows, output, **options):
        assert [len(row.history_ids) for row in rows] == [2, 12]
        assert options["steps"] == 4
        return {"checked": True}
    monkeypatch.setattr(cli, "profile_arm", profile)
    cli.main(["profile", "--data", "artifacts/fixture", "--device", "cpu",
        "--output", "artifacts/unused-profile-test", "--arm", "affine", "--seed", "17",
        "--steps", "4", "--learning-rate", ".001", "--weight-decay", ".01",
        "--max-new-tokens", "1", "--histories", "2"])


def test_slurm_is_profile_only():
    import subprocess
    path = Path("scripts/della_readout_profile.slurm")
    subprocess.run(["bash", "-n", str(path)], check=True)
    text = path.read_text()
    assert 'run_readout_interface.py profile --device cuda "$@"' in text
    assert "sbatch " not in text
