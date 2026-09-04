"""Experiment identity and artifact-directory creation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from tinymem.model.config import ExperimentConfig


@dataclass(frozen=True)
class GitSourceState:
    """Identify the commit and any uncommitted source-tree content."""

    commit: str
    dirty: bool
    working_tree_sha256: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


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


def current_git_source_state(repository_root: str | Path) -> GitSourceState:
    """Return a reproducible fingerprint for the current Git source tree."""
    root = Path(repository_root)
    commit = current_git_commit(root)
    try:
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        untracked_output = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "could not fingerprint the current Git source tree"
        ) from error

    untracked_paths = sorted(
        path for path in untracked_output.split(b"\0") if path
    )
    if not diff and not untracked_paths:
        return GitSourceState(
            commit=commit,
            dirty=False,
            working_tree_sha256=None,
        )

    digest = hashlib.sha256()
    digest.update(b"tracked-diff\0")
    digest.update(diff)
    for encoded_path in untracked_paths:
        relative_path = os.fsdecode(encoded_path)
        path = root / relative_path
        digest.update(b"untracked\0")
        digest.update(encoded_path)
        digest.update(b"\0")
        if path.is_symlink():
            digest.update(b"symlink\0")
            digest.update(os.fsencode(os.readlink(path)))
        elif path.is_file():
            digest.update(b"file\0")
            digest.update(path.read_bytes())
        else:
            digest.update(b"other\0")
    return GitSourceState(
        commit=commit,
        dirty=True,
        working_tree_sha256=digest.hexdigest(),
    )


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
    source_state: GitSourceState | None = None,
) -> Path:
    """Create a unique run directory and record its immutable identity."""
    if source_state is not None:
        if not isinstance(source_state, GitSourceState):
            raise TypeError("source_state must be a GitSourceState or None")
        if source_state.commit != git_commit.strip():
            raise ValueError("source_state commit must match git_commit")

    identity = experiment_id(config, git_commit)
    created_at = datetime.now(UTC)
    timestamp = created_at.strftime("%Y%m%dT%H%M%S.%fZ")
    run_token = uuid.uuid4().hex[:8]
    run_directory = Path(artifact_root) / f"{identity}-{timestamp}-{run_token}"
    run_directory.mkdir(parents=True, exist_ok=False)

    metadata = {
        "experiment_id": identity,
        "git_commit": git_commit.strip(),
        "source_state": (
            source_state.to_dict() if source_state is not None else None
        ),
        "created_at": created_at.isoformat(),
        "config": config.to_dict(),
    }
    (run_directory / "run.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return run_directory
