"""Writer studies bind frozen readers, train-only features, and complete schedules."""

from copy import deepcopy
import json

import pytest
import torch

from conftest import WordTokenizer
from tinymem.studies.oracle import protocol as parent_protocol
from tinymem.studies.delta.protocol import read_dataset
from tinymem.studies.distilled.protocol import (
    cells, load_features, prepare_features, prepare_study, require_training_seal, settings, verify_study,
)
from tinymem.studies.oracle.fit import train_cell as train_parent
from tinymem.reader.pretrained import PretrainedReader


@pytest.fixture
def writer_study(tiny_reader, tmp_path):
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    root = tmp_path / "root"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\n")
    (root / "uv.lock").write_text("version = 1\n")
    parent = tmp_path / "parent"
    snapshot = {"tiny_fixture": True}
    parent_spec = parent_protocol.settings(device="cpu", smoke=True)
    old = parent_protocol.prepare_study(root, parent, snapshot, parent_spec)
    base = deepcopy(tiny_reader.model)
    tokenizer = WordTokenizer()

    def fresh():
        return PretrainedReader(deepcopy(base), tokenizer)

    train_parent(fresh(), parent, old, read_dataset(parent / "dataset.json"), 0)
    parent_protocol.seal_training(parent, old)
    study = tmp_path / "study"
    protocol = prepare_study(root, study, snapshot, settings(device="cpu", smoke=True), parent)
    try:
        yield root, study, protocol, fresh
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def test_protocol_binds_parent_and_uses_own_fixed_schedule(writer_study):
    root, study, protocol, fresh = writer_study
    verified, dataset = verify_study(study, root)
    assert verified == protocol
    assert len(protocol["schedule"]) == 1 and len(protocol["schedule"][0]) == 4
    assert all(len(batch) == 4 for batch in protocol["schedule"][0])
    assert all(key in {e.id for e in dataset.train} for batch in protocol["schedule"][0] for key in batch)
    with pytest.raises(FileNotFoundError):
        require_training_seal(study, protocol)
    prepare_features(fresh(), study, protocol, dataset)
    cache = load_features(study, protocol)
    expected = {s.text for e in (*dataset.train, *dataset.validation) for s in (*e.prefix, *e.tail)}
    assert set(cache) == expected
    assert all(v.dtype == torch.float32 and v.device.type == "cpu" and not v.requires_grad for v in cache.values())
    first = next(iter(cache))
    original = cache[first].clone()
    cache[first].zero_()
    assert torch.equal(load_features(study, protocol)[first], original)
    with pytest.raises(FileExistsError):
        prepare_features(fresh(), study, protocol, dataset)
    checkpoint = study / "parent/training/0/checkpoint.safetensors"
    with checkpoint.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="parent study changed"):
        verify_study(study, root)


def test_feature_seal_rejects_changed_cache(writer_study):
    root, study, protocol, fresh = writer_study
    _, dataset = verify_study(study, root)
    prepare_features(fresh(), study, protocol, dataset)
    (study / "features/feature_texts.json").write_text(json.dumps(["test-only text"]))
    with pytest.raises(ValueError, match="outputs changed"):
        load_features(study, protocol)


def test_copied_source_rejects_additional_execution_module(writer_study):
    root, study, _, _ = writer_study
    extra = study / "source/src/extra.py"
    extra.parent.mkdir(parents=True)
    extra.write_text("raise RuntimeError('unexpected execution source')\n")
    with pytest.raises(ValueError, match="source inventory differs"):
        verify_study(study, root)


def test_full_study_keeps_all_three_parent_seeds():
    spec = settings(device="cuda")
    parent = {"settings": {"purpose": "privileged_correct_state_diagnostic"},
              "cells": [{"index": i, "seed": seed} for i, seed in enumerate((3101, 3102, 3103))]}
    assert [cell["seed"] for cell in cells(spec, parent)] == [3101, 3102, 3103]
    parent["cells"].pop()
    with pytest.raises(ValueError, match="all three"):
        cells(spec, parent)


def test_fixed_beta_is_opt_in_and_uses_the_only_supported_study_value():
    learned = settings(device="cpu", smoke=True)
    fixed = settings(device="cpu", smoke=True, fixed_beta=0.75)

    assert "fixed_beta" not in learned
    assert fixed["fixed_beta"] == 0.75
    with pytest.raises(ValueError, match="fixed_beta=0.75"):
        settings(device="cpu", smoke=True, fixed_beta=0.5)


def test_normalization_is_opt_in_and_records_an_explicit_identity():
    learned = settings(device="cpu", smoke=True)
    disabled = settings(device="cpu", smoke=True, normalize_hidden=False)
    normalized = settings(device="cpu", smoke=True, fixed_beta=0.75, normalize_hidden=True)

    assert json.dumps(learned, sort_keys=True) == json.dumps(disabled, sort_keys=True)
    assert "normalize_hidden" not in learned
    assert "hidden_normalization" not in learned
    assert normalized["normalize_hidden"] is True
    assert normalized["hidden_normalization"] == {
        "kind": "layer_norm", "width": 64, "eps": 1e-5,
        "elementwise_affine": False, "placement": "preGELU",
    }
    with pytest.raises(ValueError, match="fixed_beta=0.75"):
        settings(device="cpu", smoke=True, normalize_hidden=True)
    with pytest.raises(TypeError, match="normalize_hidden"):
        settings(device="cpu", smoke=True, normalize_hidden=1)


def test_verify_study_rejects_tampered_fixed_beta_declaration(writer_study):
    root, study, _, _ = writer_study
    protocol_path = study / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["settings"]["fixed_beta"] = 0.5
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="fixed_beta=0.75|procedure differs"):
        verify_study(study, root)


def test_verify_study_rejects_tampered_normalization_identity(writer_study):
    root, study, protocol, _ = writer_study
    normalized = study.with_name("normalized-study")
    protocol = prepare_study(
        root, normalized, protocol["snapshot"],
        settings(device="cpu", smoke=True, fixed_beta=0.75, normalize_hidden=True),
        study / "parent",
    )
    protocol_path = normalized / "protocol.json"
    declared = json.loads(protocol_path.read_text())
    declared["settings"]["hidden_normalization"]["eps"] = 1e-4
    protocol_path.write_text(json.dumps(declared, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="procedure differs"):
        verify_study(normalized, root)
