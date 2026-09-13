import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/della_readout_train.slurm"


@pytest.fixture
def shell(tmp_path):
    # Exercise the real shell script, with explicit stubs for cluster commands.
    source = f"#!{sys.executable}\n" + '''import json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["CALL_LOG"], "a") as handle:
    handle.write(json.dumps({"command": name, "args": args,
        "offline": os.environ.get("HF_HUB_OFFLINE"),
        "hf_home": os.environ.get("HF_HOME"),
        "cublas": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}) + "\\n")
if name == "git":
    if args[0] == "diff" and os.environ.get("DIRTY"):
        sys.exit(1)
    if args[0] == "ls-files" and os.environ.get("UNTRACKED"):
        print("src/unreviewed.py")
if name == os.environ.get("FAIL_COMMAND"):
    sys.exit(7)
if name == "srun":
    os.execvp(args[0], args)
'''
    for name in ("git", "module", "conda", "nvidia-smi", "python", "srun"):
        executable = tmp_path / name
        executable.write_text(source)
        executable.chmod(0o755)

    def run(*args, **overrides):
        log = tmp_path / "calls.jsonl"
        env = dict(os.environ, PATH=f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
                   CALL_LOG=str(log), SLURM_CPUS_PER_TASK="4", **overrides)
        env.pop("BASH_ENV", None)
        if "HF_HOME" not in overrides:
            env.pop("HF_HOME", None)
        result = subprocess.run(["bash", str(SCRIPT), *args], cwd=tmp_path,
                                env=env, capture_output=True, text=True)
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return result, calls
    return run


@pytest.mark.parametrize("arm", ["affine", "gelu"])
@pytest.mark.parametrize("seed", [1337, 2027, 4099])
def test_full_training_passes_explicit_schedule_to_one_cuda_run(shell, arm, seed):
    args = ["--data", "artifacts/predictions/memory_update_data_20260905_v2",
            "--output", f"artifacts/predictions/full training/{arm}_seed_{seed}",
            "--arm", arm, "--seed", str(seed), "--steps", "1000",
            "--learning-rate", "0.001", "--weight-decay", "0.01", "--max-new-tokens", "8"]
    result, calls = shell(*args)
    assert result.returncode == 0, result.stderr
    launches = [call for call in calls if call["command"] == "srun"]
    assert len(launches) == 1
    assert launches[0]["args"] == ["python", "scripts/run_readout_interface.py", "run", "--device", "cuda", *args]
    assert launches[0]["offline"] == "1" and launches[0]["cublas"] == ":4096:8"
    python_calls = [call["args"] for call in calls if call["command"] == "python"]
    assert python_calls == [["-m", "pip", "check"], launches[0]["args"][1:]]


@pytest.mark.parametrize("overrides", [{"DIRTY": "1"}, {"UNTRACKED": "1"},
                                       {"FAIL_COMMAND": "module"}, {"FAIL_COMMAND": "python"}])
def test_preflight_failure_prevents_training(shell, overrides):
    result, calls = shell(**overrides)
    assert result.returncode != 0
    assert not any(call["command"] == "srun" for call in calls)


def test_failed_training_exits_without_retry(shell):
    result, calls = shell(FAIL_COMMAND="srun")
    assert result.returncode == 7
    assert sum(call["command"] == "srun" for call in calls) == 1


def test_training_slurm_syntax_and_separate_logs():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    text = SCRIPT.read_text()
    assert "#SBATCH --job-name=tinymem-readout-train" in text
    assert "#SBATCH --output=logs/%x-%j.out" in text
    assert "#SBATCH --error=logs/%x-%j.err" in text
    assert "#SBATCH --chdir=" not in text


@pytest.mark.parametrize("cache", [None, "/staged/reader cache"])
def test_reader_cache_uses_submission_directory_or_explicit_override(shell, cache):
    result, calls = shell(**({"HF_HOME": cache} if cache is not None else {}))
    assert result.returncode == 0, result.stderr
    observed = next(call["hf_home"] for call in calls if call["command"] == "srun")
    if cache is not None:
        assert observed == cache
    else:
        assert observed.endswith("/.cache/huggingface")


@pytest.mark.parametrize("name", ["della.slurm", "della_updates.slurm", "della_readout_profile.slurm", "della_readout_train.slurm"])
def test_public_cluster_scripts_have_no_fixed_workspace(name):
    source = ROOT / "scripts" / name
    subprocess.run(["bash", "-n", str(source)], check=True)
    assert "#SBATCH --chdir=" not in source.read_text()
