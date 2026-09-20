"""The oracle diagnostic retains its declared data, seeds, and endpoints."""

from copy import deepcopy
import json

import pytest

from tinymem.research.delta_fact_data import build_dataset
from tinymem.research.delta_fact_protocol import epoch_schedule
from tinymem.research.oracle_fact_protocol import (
    cells, prepare_study, require_training_seal, settings, verify_study,
)


def test_diagnostic_has_three_fresh_readers_and_the_original_full_schedule():
    spec = settings(device="cuda")
    data = build_dataset(seed=spec["data_seed"])
    schedule = epoch_schedule(data.train, spec)
    assert len(schedule) == 4 and sum(map(len, schedule)) == 2560
    expected = sorted(episode.id for episode in data.train)
    assert all(sorted(key for batch in epoch for key in batch) == expected for epoch in schedule)
    assert [cell["seed"] for cell in cells(spec)] == [3101, 3102, 3103]
    assert all(cell["persistent_bytes"] == 258 for cell in cells(spec))
    assert spec["accuracy_exclusion"] is False and spec["checkpoint_selection"] == "fixed_final"
    assert "diagnostic" in spec["purpose"] and "not fresh confirmation" in spec["panel_status"]


def test_oracle_declaration_rejects_budget_code_and_source_changes(tmp_path):
    root = tmp_path / "root"
    (root / "src").mkdir(parents=True)
    (root / "src/core.py").write_text("x = 1\n")
    (root / "pyproject.toml").write_text("[project]\n")
    (root / "uv.lock").write_text("version = 1\n")
    spec = settings(device="cpu", smoke=True)
    changed = deepcopy(spec)
    changed["state_code"]["beta"] = 1.0
    with pytest.raises(ValueError, match="settings differ"):
        prepare_study(root, tmp_path / "invalid", {}, changed)
    study = tmp_path / "study"
    protocol = prepare_study(root, study, {"fixture": True}, spec)
    assert verify_study(study, root)[0] == protocol
    with pytest.raises(FileNotFoundError):
        require_training_seal(study, protocol)
    altered = deepcopy(protocol)
    altered["schedule"][0].pop()
    (study / "protocol.json").write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="schedule"):
        verify_study(study, root)
    (study / "protocol.json").write_text(json.dumps(protocol))
    (root / "src/core.py").write_text("x = 2\n")
    with pytest.raises(ValueError, match="execution sources"):
        verify_study(study, root)
