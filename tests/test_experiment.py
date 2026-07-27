import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tinymem.model.config import ExperimentConfig
from tinymem.utils.experiment import (
    create_run_directory,
    current_git_commit,
    experiment_id,
)


def test_experiment_id_is_stable() -> None:
    config = ExperimentConfig(seed=47)

    assert experiment_id(config, "abc123") == experiment_id(config, "abc123")


def test_experiment_id_changes_with_config() -> None:
    first = experiment_id(ExperimentConfig(seed=47), "abc123")
    second = experiment_id(ExperimentConfig(seed=48), "abc123")

    assert first != second


def test_experiment_id_changes_with_git_commit() -> None:
    config = ExperimentConfig(seed=47)

    assert experiment_id(config, "abc123") != experiment_id(config, "def456")


def test_experiment_id_rejects_empty_git_commit() -> None:
    with pytest.raises(ValueError, match="git_commit must be nonempty"):
        experiment_id(ExperimentConfig(), "  ")


def test_create_run_directory_is_unique_and_records_identity(tmp_path: Path) -> None:
    config = ExperimentConfig(seed=53)

    first = create_run_directory(tmp_path, config, git_commit="abc123")
    second = create_run_directory(tmp_path, config, git_commit="abc123")

    assert first != second
    assert first.is_dir()
    assert second.is_dir()

    metadata = json.loads((first / "run.json").read_text())
    assert metadata["experiment_id"] == experiment_id(config, "abc123")
    assert metadata["git_commit"] == "abc123"
    assert metadata["config"] == json.loads(json.dumps(config.to_dict()))
    assert "created_at" in metadata


def test_current_git_commit_reads_repository_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], Path]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs["cwd"]))
        return SimpleNamespace(stdout="abc123\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert current_git_commit(tmp_path) == "abc123"
    assert calls == [(["git", "rev-parse", "HEAD"], tmp_path)]


def test_current_git_commit_wraps_git_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_run(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(subprocess, "run", fail_run)

    with pytest.raises(RuntimeError, match="could not determine"):
        current_git_commit(tmp_path)
