"""Reports consume sealed real run artifacts, not hand-assembled summaries."""
import json

import pytest

from test_readout_runner import tiny_reader
from test_readout_experiment import inputs
from tinymem.research.readout_experiment import run_arm
from tinymem.research.readout_report import load_run_groups


@pytest.fixture
def completed_run(tmp_path, tiny_reader):
    output = tmp_path / "arm"
    run_arm(
        tiny_reader, inputs(tiny_reader), output,
        kind="affine", seed=17, steps=1,
        learning_rate=.001, weight_decay=.01, max_new_tokens=1,
        input_identity={"evidence_kind": "tiny_random_cpu_test"},
    )
    return output


def test_load_complete_run_groups(completed_run):
    groups = load_run_groups(completed_run)
    assert len(groups) == 18
    assert sum(len(rows) for rows in groups.values()) == 360
    for rows in groups.values():
        assert len(rows) == 20
        assert sum(row["category"] == "update_known" for row in rows) == 16
        assert sum(row["category"] == "update_missing" for row in rows) == 4


def test_load_rejects_unsealed_prediction_change(completed_run):
    path = completed_run / "predictions.jsonl"
    rows = path.read_text().splitlines()
    row = json.loads(rows[0])
    row["case_id"] = "not-a-source-case"
    rows[0] = json.dumps(row)
    path.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError):
        load_run_groups(completed_run)
