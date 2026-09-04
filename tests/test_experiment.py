import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tinymem.model.config import ExperimentConfig
from tinymem.utils.experiment import (
    GitSourceState,
    create_run_directory,
    current_git_commit,
    current_git_source_state,
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
    source_state = GitSourceState(
        commit="abc123",
        dirty=True,
        working_tree_sha256="f" * 64,
    )

    first = create_run_directory(
        tmp_path,
        config,
        git_commit="abc123",
        source_state=source_state,
    )
    second = create_run_directory(tmp_path, config, git_commit="abc123")

    assert first != second
    assert first.is_dir()
    assert second.is_dir()

    metadata = json.loads((first / "run.json").read_text())
    assert metadata["experiment_id"] == experiment_id(config, "abc123")
    assert metadata["git_commit"] == "abc123"
    assert metadata["source_state"] == source_state.to_dict()
    assert metadata["config"] == json.loads(json.dumps(config.to_dict()))
    assert "created_at" in metadata


def test_create_run_directory_rejects_mismatched_source_state(
    tmp_path: Path,
) -> None:
    source_state = GitSourceState(
        commit="different",
        dirty=False,
        working_tree_sha256=None,
    )

    with pytest.raises(ValueError, match="must match"):
        create_run_directory(
            tmp_path,
            ExperimentConfig(),
            git_commit="abc123",
            source_state=source_state,
        )

    assert not list(tmp_path.iterdir())


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


def test_current_git_source_state_fingerprints_dirty_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        (
            SimpleNamespace(stdout="abc123\n"),
            SimpleNamespace(stdout=b"diff content"),
            SimpleNamespace(stdout=b"new.py\0"),
        )
    )
    (tmp_path / "new.py").write_text("value = 1\n", encoding="utf-8")

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: next(responses))

    state = current_git_source_state(tmp_path)

    assert state.commit == "abc123"
    assert state.dirty
    assert state.working_tree_sha256 is not None
    assert len(state.working_tree_sha256) == 64


def test_current_git_source_state_reports_clean_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        (
            SimpleNamespace(stdout="abc123\n"),
            SimpleNamespace(stdout=b""),
            SimpleNamespace(stdout=b""),
        )
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: next(responses))

    assert current_git_source_state(tmp_path) == GitSourceState(
        commit="abc123",
        dirty=False,
        working_tree_sha256=None,
    )
