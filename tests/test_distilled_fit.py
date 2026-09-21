from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

pytest_plugins = ["test_distilled_protocol"]

from tinymem.memory.delta_slots import DeltaSlotWriter
from tinymem.studies.delta.data import build_dataset
from tinymem.studies.distilled.fit import _writer, build_example, load_writer, train_cell
from tinymem.studies.distilled.training import trajectory_diagnostics


def _features(dataset, width: int = 4) -> dict[str, torch.Tensor]:
    result = {}
    for episode in (*dataset.train[:1], *dataset.validation[:1]):
        for statement in (*episode.prefix, *episode.tail):
            result[statement.text] = torch.ones(2, width)
    return result


def test_writer_factory_selects_parameter_free_fixed_beta_gate():
    settings = {"writer_hidden_width": 64, "key_width": 8, "fixed_beta": 0.75}
    writer = _writer(4, settings, 123)

    assert writer.fixed_beta == 0.75
    assert "beta_projection.weight" not in writer.state_dict()
    assert "beta_projection.bias" not in writer.state_dict()


def test_writer_factory_selects_normalized_fixed_beta_writer():
    settings = {
        "writer_hidden_width": 64, "key_width": 8, "fixed_beta": 0.75,
        "normalize_hidden": True,
    }
    writer = _writer(4, settings, 123)

    assert writer.fixed_beta == 0.75
    assert writer.normalize_hidden is True
    assert "beta_projection.weight" not in writer.state_dict()


def test_trajectory_diagnostics_separates_rollout_and_one_step() -> None:
    dataset = build_dataset(seed=2026091407, train_prefixes=16, validation_prefixes=16, test_prefixes=16)
    episode = next(item for item in dataset.train if item.condition == "correction")
    writer = DeltaSlotWriter(4, 32, key_width=8)
    writer.eval()
    writer.requires_grad_(False)
    diagnostics = trajectory_diagnostics(writer, (episode,), _features(dataset))

    assert len(diagnostics) == 16
    assert [row["after_write"] for row in diagnostics] == list(range(1, 17))
    assert all(len(row["direct_bits"]) == 4 for row in diagnostics)
    assert all(len(row["truth"]) == 4 for row in diagnostics)
    assert all(row["known_fact_count"] >= 1 for row in diagnostics)
    assert any(row["state_sse"] != row["one_step_state_sse"] for row in diagnostics[1:])


def test_build_example_targets_match_independent_delta_recurrence() -> None:
    dataset = build_dataset(seed=2026091407, train_prefixes=16, validation_prefixes=16, test_prefixes=16)
    episode = next(item for item in dataset.train if item.condition == "correction")
    writer = DeltaSlotWriter(4, 32, key_width=8)
    features = {statement.text: torch.ones(1, 4) for statement in (*episode.prefix, *episode.tail)}
    example = build_example(episode, features, writer)
    matrix = torch.zeros(8, 8)
    expected = []
    for statement in (*episode.prefix, *episode.tail):
        row = statement.entity
        value = 0.5 if statement.value else -0.5
        target_row = torch.zeros(8)
        target_row[0] = value
        residual = target_row - matrix[row]
        matrix[row] = matrix[row] + 0.75 * residual
        expected.append(matrix.reshape(1, 2, 32).clone())
    assert all(torch.allclose(actual, target, atol=1e-7, rtol=0.0) for actual, target in zip(example.targets, expected))


def test_train_cell_uses_real_sealed_feature_protocol(writer_study) -> None:
    root, study, protocol, fresh = writer_study
    from tinymem.studies.distilled.protocol import (
        load_features,
        prepare_features,
        read_dataset,
        verify_study,
        verify_training_result,
    )

    _, dataset = verify_study(study, root)
    prepare_features(fresh(), study, protocol, dataset)
    before = {name: tensor.clone() for name, tensor in load_features(study, protocol).items()}

    result = train_cell(study, protocol, read_dataset(study / "dataset.json"), 0)

    assert result["optimizer_steps"] == 4
    assert result["reader_trained"] is False
    assert result["persistent_bytes"] == 258
    verify_training_result(study, protocol, 0)
    for name, tensor in load_features(study, protocol).items():
        assert torch.equal(tensor, before[name])
    loaded = load_writer(study, protocol, 0)
    assert loaded.training is False
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    assert all(parameter.grad is None for parameter in loaded.parameters())


def test_load_writer_rejects_wrong_checkpoint_schema(writer_study) -> None:
    root, study, protocol, fresh = writer_study
    from tinymem.studies.distilled.protocol import (
        prepare_features,
        read_dataset,
        verify_study,
    )

    _, dataset = verify_study(study, root)
    prepare_features(fresh(), study, protocol, dataset)
    train_cell(study, protocol, read_dataset(study / "dataset.json"), 0)
    checkpoint = study / "training/0/checkpoint.safetensors"
    save_file({"wrong": torch.zeros(1)}, str(checkpoint))
    with pytest.raises(ValueError, match="schema"):
        load_writer(study, protocol, 0)
