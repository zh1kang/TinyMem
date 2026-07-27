"""Experiment identity and artifact-directory creation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path

from tinymem.model.config import ExperimentConfig


def current_git_commit(repository_root: str | Path) -> str:
    """Return the commit that identifies the current source snapshot."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repository_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("could not determine the current Git commit") from error

    commit = result.stdout.strip()
    if not commit:
        raise RuntimeError("Git returned an empty commit identifier")
    return commit


def experiment_id(config: ExperimentConfig, git_commit: str) -> str:
    """Build a stable identifier from configuration and source revision."""
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    if not isinstance(git_commit, str):
        raise TypeError("git_commit must be a string")
    if not git_commit.strip():
        raise ValueError("git_commit must be nonempty")

    canonical_config = json.dumps(
        config.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    config_digest = hashlib.sha256(canonical_config).hexdigest()[:12]
    commit_digest = hashlib.sha256(git_commit.strip().encode()).hexdigest()[:12]
    return f"{commit_digest}-{config_digest}"


def create_run_directory(
    artifact_root: str | Path,
    config: ExperimentConfig,
    *,
    git_commit: str,
) -> Path:
    """Create a unique run directory and record its immutable identity."""
    identity = experiment_id(config, git_commit)
    created_at = datetime.now(UTC)
    timestamp = created_at.strftime("%Y%m%dT%H%M%S.%fZ")
    run_token = uuid.uuid4().hex[:8]
    run_directory = Path(artifact_root) / f"{identity}-{timestamp}-{run_token}"
    run_directory.mkdir(parents=True, exist_ok=False)

    metadata = {
        "experiment_id": identity,
        "git_commit": git_commit.strip(),
        "created_at": created_at.isoformat(),
        "config": config.to_dict(),
    }
    (run_directory / "run.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return run_directory
