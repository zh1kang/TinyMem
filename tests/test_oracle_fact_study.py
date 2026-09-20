"""A real tiny-reader fit survives checkpoint reload without changing the base."""

from copy import deepcopy

import pytest
import torch
from safetensors.torch import load_file

from test_readout_runner import tiny_reader
from test_update_runner import WordTokenizer
from tinymem.research.delta_fact_protocol import read_dataset
from tinymem.research.oracle_fact_fit import checkpoint_tensors, load_trained, train_cell
from tinymem.research.oracle_fact_protocol import (
    prepare_study, require_training_seal, seal_training, settings,
)
from tinymem.research.oracle_fact_scoring import aggregate_study, score_cell
from tinymem.research.pretrained import PretrainedReader


def test_fixed_oracle_fit_checkpoint_reload_and_training_seal(tiny_reader, tmp_path):
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

    with pytest.raises(FileNotFoundError):
        require_training_seal(study, protocol)
    result = train_cell(fresh(), study, protocol, dataset, 0)
    assert result["optimizer_steps"] == 4
    assert result["base_before_sha256"] == result["base_after_sha256"]
    assert result["test_scored"] is False and result["parameters"]["writer"] == 0
    path = study / "training/0/checkpoint.safetensors"
    restored_reader = fresh()
    restored_bridge = load_trained(restored_reader, protocol, protocol["cells"][0], path)
    expected = load_file(str(path))
    assert all(torch.equal(tensor, expected[name])
               for name, tensor in checkpoint_tensors(restored_reader, restored_bridge).items())
    assert not any(p.requires_grad or p.grad is not None for p in restored_reader.model.parameters())
    assert not restored_bridge.training
    seal_training(study, protocol)
    require_training_seal(study, protocol)
    scored = score_cell(fresh(), study, protocol, dataset, 0)
    assert scored["rows"] == 224
    summary = aggregate_study(study, protocol)
    assert summary["all_rows"] == 224
    with pytest.raises(FileExistsError):
        train_cell(fresh(), study, protocol, dataset, 0)
    with pytest.raises(FileExistsError):
        seal_training(study, protocol)
    (study / "training/0/metrics.jsonl").write_text("")
    with pytest.raises(ValueError, match="outputs changed"):
        require_training_seal(study, protocol)
