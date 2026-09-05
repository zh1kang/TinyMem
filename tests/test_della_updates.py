import json
import os
from pathlib import Path
import subprocess

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


def test_transfer_manifest_includes_required_data_but_not_partial_confirmation():
    from scripts.update_transfer_manifest import transfer_paths
    paths = transfer_paths(Path("artifacts/predictions/memory_update_data_20260905_v2"))
    assert "artifacts/predictions/memory_update_data_20260905_v2" in paths
    assert "data/raw/pretrained/qwen3-1.7b" in paths
    assert sum("_seed_" in path for path in paths) == 6
    assert not any("partial_confirmation" in path or "della_" in path or "artifacts/smoke/" in path for path in paths)
    # Historical partial-evidence exclusion manifests are required inputs, not
    # partial output from the current confirmation execution.
    assert "artifacts/predictions/native_holdout_reserve_20260904/data_manifest.json" in paths
