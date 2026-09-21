"""Frozen schedules and seals reject drift before any heldout read."""

import json

import pytest

from tinymem.studies.delta.data import build_dataset
from tinymem.studies.delta.protocol import (
    cells, epoch_schedule, prepare_study, require_training_seal, seal_directory,
    settings, verify_completion, verify_study,
)


def test_full_epochs_cover_every_branch_once_without_condition_resampling():
    spec = settings(device="cuda")
    data = build_dataset(seed=spec["data_seed"])
    schedule = epoch_schedule(data.train, spec)
    assert len(schedule) == 4 and sum(map(len, schedule)) == 2560
    expected = sorted(e.id for e in data.train)
    assert all(sorted(i for batch in epoch for i in batch) == expected for epoch in schedule)
    assert schedule[0] != schedule[1]
    assert schedule == epoch_schedule(data.train, spec)
    assert len(cells(spec)) == 12
    for seed in spec["seeds"]:
        paired = [cell for cell in cells(spec) if cell["writer_seed"] == seed]
        assert len({cell["adapter_seed"] for cell in paired}) == 1
        assert len({cell["bridge_seed"] for cell in paired}) == 1


def test_protocol_copy_and_data_reject_drift(tmp_path):
    root = tmp_path / "root"
    (root / "src").mkdir(parents=True)
    (root / "src/core.py").write_text("x = 1\n")
    (root / "pyproject.toml").write_text("[project]\n")
    (root / "uv.lock").write_text("version = 1\n")
    study = tmp_path / "study"
    protocol = prepare_study(root, study, {"fixture": True}, settings(device="cpu", smoke=True))
    assert verify_study(study, root)[0] == protocol
    (root / "src/core.py").write_text("x = 2\n")
    with pytest.raises(ValueError, match="execution sources"):
        verify_study(study, root)
    (root / "src/core.py").write_text("x = 1\n")
    (study / "dataset.json").write_text("{}")
    with pytest.raises(ValueError, match="data changed"):
        verify_study(study, root)


def test_seals_are_bound_to_identity_and_exact_files(tmp_path):
    identity = {"stage": "training", "index": 0}
    (tmp_path / "data.json").write_text("{}")
    seal_directory(tmp_path, identity)
    assert verify_completion(tmp_path, identity)["identity"] == identity
    with pytest.raises(ValueError, match="different study"):
        verify_completion(tmp_path, {**identity, "index": 1})
    (tmp_path / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="outputs changed"):
        verify_completion(tmp_path, identity)
    with pytest.raises(FileExistsError):
        seal_directory(tmp_path, identity)


def test_final_evaluation_requires_every_training_cell(tmp_path):
    (tmp_path / "protocol.json").write_text("{}")
    with pytest.raises(FileNotFoundError):
        require_training_seal(tmp_path, {"cells": [{}]})
    (tmp_path / "training_sealed.json").write_text(json.dumps({"protocol_sha256": "wrong", "training_completions": {}}))
    with pytest.raises(ValueError, match="different protocol"):
        require_training_seal(tmp_path, {"cells": [{}]})
