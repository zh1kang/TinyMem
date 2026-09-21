"""Behavioral checks for the privileged known-state scorer."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from conftest import WordTokenizer
from tinymem.studies.delta.protocol import cell_identity, read_dataset, seal_directory
from tinymem.studies.artifacts import file_hash
from tinymem.studies.oracle.fit import train_cell
from tinymem.studies.oracle.protocol import prepare_study, seal_training, settings
from tinymem.studies.oracle import scoring
from tinymem.reader.pretrained import PretrainedReader


def _read(answer: str, *, correct: bool = True) -> dict:
    prediction = answer if correct else "not-the-answer"
    return {"prediction": prediction, "generated_ids": [1], "input_positions": 8,
            "memory_positions": 2, "native_envelope_tokens": 6, "correct": correct}


def test_read_metadata_recomputes_correctness_instead_of_trusting_metric():
    read = _read("bathroom")
    read["correct"] = False
    with pytest.raises(ValueError, match="correctness"):
        scoring._read_metadata(read, "bathroom", "update_known")


def test_paired_state_audit_rejects_wording_state_mismatch():
    import torch

    def record(wording, value):
        return SimpleNamespace(prefix_id="p", condition="repeat", target=0, after_write=16,
                               wording=wording, values=torch.full((1, 2, 32), value),
                               valid=torch.ones((1, 2), dtype=torch.bool))

    with pytest.raises(ValueError, match="paired logical"):
        scoring._paired_state_audit((record("familiar", 1), record("heldout", 2)))


def test_aggregate_rejects_omitted_case_and_mutated_metric(tiny_reader, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\n")
    (root / "uv.lock").write_text("version = 1\n")
    study = tmp_path / "study"
    protocol = prepare_study(root, study, {"tiny_fixture": True}, settings(device="cpu", smoke=True))
    dataset = read_dataset(study / "dataset.json")
    base = deepcopy(tiny_reader.model)
    tokenizer = WordTokenizer()

    def fresh():
        return PretrainedReader(deepcopy(base), tokenizer)

    train_cell(fresh(), study, protocol, dataset, 0)
    seal_training(study, protocol)
    scoring.score_cell(fresh(), study, protocol, dataset, 0)
    directory = study / "evaluation" / "0"
    identity = {**cell_identity(study, protocol, 0, "evaluation"),
                "training_seal_sha256": file_hash(study / "training_sealed.json")}
    predictions = directory / "predictions.jsonl"
    original = predictions.read_text().splitlines()

    predictions.write_text("\n".join(original[:-1]) + "\n")
    (directory / "complete.json").unlink()
    seal_directory(directory, identity)
    with pytest.raises(ValueError, match="omits or adds"):
        scoring.aggregate_study(study, protocol)

    predictions.write_text("\n".join(original) + "\n")
    mutated = json.loads(original[0])
    mutated["reads"]["real"]["correct"] = not mutated["reads"]["real"]["correct"]
    predictions.write_text("\n".join([json.dumps(mutated), *original[1:]]) + "\n")
    (directory / "complete.json").unlink()
    seal_directory(directory, identity)
    with pytest.raises(ValueError, match="correctness"):
        scoring.aggregate_study(study, protocol)


def test_bootstrap_reports_absolute_paired_gap_without_threshold():
    rows = []
    for seed in ("3101", "3102", "3103"):
        for prefix in ("p0", "p1"):
            for entity in range(2):
                rows.append({"writer_seed": seed, "prefix_id": prefix,
                             "after_write": 16, "condition": "repeat", "scope": "unspoken",
                             "reads": {"real": {"correct": True}, "zero": {"correct": False},
                                       "donor": {"correct": False}}})
    result = scoring._paired_bootstrap(rows, "zero", condition="repeat", scope="unspoken",
                                       settings={"bootstrap_samples": 17, "bootstrap_seed": 3})
    assert result["mean_gap_pp"] == 100.0
    assert result["interval_pp"] == [100.0, 100.0]
    assert result["confidence"] == .99
