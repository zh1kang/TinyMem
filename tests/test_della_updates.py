import json
import os
from pathlib import Path
import subprocess
from functools import partial
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def shell(tmp_path):
    executable = tmp_path / "python-stub"
    executable.write_text("#!/usr/bin/env python3\nimport json,os,sys\n"
        "with open(os.environ['CALL_LOG'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "if os.environ.get('FAIL_ON') in sys.argv[1:]: sys.exit(3)\n")
    executable.chmod(0o755)
    env = dict(os.environ, TINYMEM_PYTHON=str(executable), CALL_LOG=str(tmp_path / "calls"), SLURM_JOB_ID="fixture")
    def run(*args, fail=None):
        if fail:
            env["FAIL_ON"] = fail
        result = subprocess.run(["bash", "scripts/della_updates.sh", *args], cwd=ROOT, env=env, capture_output=True, text=True)
        path = Path(env["CALL_LOG"])
        calls = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        return result, calls
    return run


def test_run_orders_qualification_six_new_training_then_confirmation_report(shell):
    result, calls = shell("run", "artifacts/predictions/fixture-run", "1000")
    assert result.returncode == 0, result.stderr
    train = [i for i, call in enumerate(calls) if "train" in call]
    evaluate = [i for i, call in enumerate(calls) if "evaluate" in call]
    assert len(train) == 6 and len(evaluate) == 13
    assert max(train) < min(evaluate)
    assert next(i for i, c in enumerate(calls) if "qualify" in c) < next(i for i, c in enumerate(calls) if "freeze" in c) < min(train)
    assert calls[-1][1] == "scripts.report_memory_updates"
    assert all("scripts.opaque.train" not in call for call in calls)
    assert all("--split" in calls[i] and "confirmation" in calls[i] for i in evaluate)


def test_failed_qualification_stops_before_launch_training_and_confirmation(shell):
    result, calls = shell("run", "artifacts/predictions/fixture-run", "1000", fail="qualify")
    assert result.returncode == 3
    assert not any(any(command in call for command in ("freeze", "train", "evaluate")) for call in calls)


@pytest.mark.parametrize("args", [("run", "out", "0"), ("run", "out", "1.5"), ("run", "out"),
                                  ("unknown", "out"), ("check", "../escape"), ("check", "/absolute")])
def test_invalid_shell_arguments_fail_before_any_command(shell, args):
    result, calls = shell(*args)
    assert result.returncode != 0 and calls == []


def test_shell_syntax_and_existing_diagnostics_checkpoint_reuse():
    for name in ("della_updates.sh", "della_updates.slurm", "della_run.sh", "della.slurm"):
        subprocess.run(["bash", "-n", str(ROOT / "scripts" / name)], check=True)
    text = (ROOT / "scripts/della_run.sh").read_text()
    assert 'for stage in smoke confirmation training-fit oracle' in text
    assert '--training-run "$study/${writer}_seed_1337"' in text
    assert '--training-fit "$output/training_fit/query_pool_seed_1337"' in text
    # Existing all uses checkpoints and diagnostics, not the optional profile/train stage.
    assert 'for stage in smoke profile' not in text


@pytest.fixture
def transfer_fixture(tmp_path, monkeypatch):
    from scripts import update_transfer_manifest as transfer
    from tinymem.research.study_runtime import repository_path

    directory = Path("artifacts/predictions/memory_update_data_20260905_v2")
    reserve = "artifacts/predictions/native_holdout_reserve_20260904/data_manifest.json"
    old = {
        "source_sha256": {"artifacts/predictions/historical_source.py": "fixture", "src/shared.py": "fixture"},
        "runs": [f"artifacts/predictions/old_study/{arm}_seed_{seed}" for arm in ("query_pool", "mean_pool") for seed in (1337, 2027, 4099)],
        "reader_gate": "reader/gate",
        "data": "artifacts/predictions/old_data",
    }
    inputs = [str(directory), str(directory / "protocol.json"), reserve]
    required = [*inputs, *old["runs"], old["reader_gate"], old["data"],
                "artifacts/predictions/historical_source.py", "reader/adapter",
                "data/raw/pretrained/qwen3-1.7b",
                "artifacts/predictions/raw_capacity_audit_20260905/opaque_train_vocabulary.json"]
    for name in required:
        target = tmp_path / name
        if target.suffix:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("synthetic transfer fixture")
        else:
            target.mkdir(parents=True, exist_ok=True)
    old_path = tmp_path / transfer.OLD_STUDY
    old_path.parent.mkdir(parents=True, exist_ok=True)
    old_path.write_text(json.dumps(old))
    monkeypatch.setattr(transfer, "REPOSITORY", tmp_path)
    monkeypatch.setattr(transfer, "repository_path", partial(repository_path, root=tmp_path))
    # Input verification and reader qualification have separate artifact tests.
    monkeypatch.setattr(transfer, "load_development_data", lambda path: SimpleNamespace(protocol={"input_sha256": dict.fromkeys(inputs, "fixture")}))
    monkeypatch.setattr(transfer, "shared_reader_identity", lambda: {"adapter": "reader/adapter"})
    return transfer, directory


def test_transfer_manifest_includes_required_data_but_not_partial_confirmation(transfer_fixture):
    transfer, directory = transfer_fixture
    paths = transfer.transfer_paths(directory)
    assert str(directory) in paths
    assert str(directory / "protocol.json") not in paths
    assert "data/raw/pretrained/qwen3-1.7b" in paths
    assert sum("_seed_" in path for path in paths) == 6
    assert not any("partial_confirmation" in path or "della_" in path or "artifacts/smoke/" in path for path in paths)
    assert "artifacts/predictions/native_holdout_reserve_20260904/data_manifest.json" in paths


def test_transfer_manifest_rejects_missing_input(transfer_fixture, tmp_path):
    transfer, directory = transfer_fixture
    (tmp_path / "reader/adapter").rmdir()
    with pytest.raises(FileNotFoundError, match="reader/adapter"):
        transfer.transfer_paths(directory)
