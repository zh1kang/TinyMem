"""Behavioral tests for bounded phase-two state evaluation."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from tinymem.memory.delta_slots import DeltaSlotWriter
from tinymem.research.delta_fact_data import build_dataset
from tinymem.research.delta_fact_encoding import build_feature_cache
from tinymem.research.delta_fact_evaluation import (
    StateRecord,
    collect_states,
    delta_geometry,
    fit_probe,
    predict_probe,
    transition_counts,
)


def _records(n=16, width=8):
    rows = []
    for prefix in range(n):
        truth = tuple((prefix >> bit) & 1 for bit in range(4))
        values = torch.zeros(1, 2, width)
        values.reshape(-1)[:4] = torch.tensor(truth, dtype=torch.float32)
        row = StateRecord(
            f"train-{prefix}", f"train/p{prefix:04d}", "train", "familiar",
            "no_write", None, 8, values, torch.ones(1, 2, dtype=torch.bool), truth,
        )
        rows.extend((row, replace(row, episode_id=f"branch-{prefix}")))
    return tuple(rows)


def test_probe_deduplicates_prefixes_and_generalizes_with_grouped_cv():
    model = fit_probe(_records())
    assert model["n_train"] == 16
    assert model["n_prefixes"] == 16
    assert [row["fold"] for row in model["cv"][0]["folds"]] == list(range(4))
    values = torch.stack([row.values[0] for row in _records(4)[::2]])
    expected = np.asarray([row.truth for row in _records(4)[::2]])
    np.testing.assert_array_equal(predict_probe(model, values) >= .5, expected)
    import json
    json.dumps(model, allow_nan=False)


def test_probe_rejects_nontrain_and_keeps_collapsed_features_finite():
    with pytest.raises(ValueError, match="training records"):
        fit_probe((*_records(8), replace(_records(8)[0], split="test")))
    rows = []
    for prefix in range(8):
        row = _records(8)[2 * prefix]
        rows.append(replace(row, values=torch.ones_like(row.values)))
    model = fit_probe(tuple(rows))
    assert np.isfinite(np.asarray(model["weights"])).all()


def test_collection_owns_real_writer_states_and_replays_labels():
    dataset = build_dataset(seed=17, train_prefixes=16, validation_prefixes=16, test_prefixes=16)
    episodes = tuple(row for row in dataset.train if row.prefix_id.endswith("p0000"))
    cache = {statement.text: torch.randn(3, 16) for episode in episodes for statement in (*episode.prefix, *episode.tail)}
    writer = DeltaSlotWriter(16, 8, key_width=4).eval().requires_grad_(False)
    before = {name: value.clone() for name, value in writer.state_dict().items()}
    records = collect_states(writer, episodes, cache)
    assert len(records) == 28
    assert {record.after_write for record in records} == {8, 9, 16}
    assert all(record.values.shape == (1, 2, 8)
               and record.values.numel() * 4 + record.valid.numel() == 66 for record in records)
    assert all(record.values.device.type == "cpu" and record.values.dtype == torch.float32 for record in records)
    assert all(record.valid.all() for record in records)
    assert all(record.truth and len(record.truth) == 4 for record in records)
    assert all(torch.equal(value, before[name]) for name, value in writer.state_dict().items())
    assert len({record.values.data_ptr() for record in records}) == len(records)


def test_collection_requires_frozen_writer_and_delta_geometry_is_descriptive():
    dataset = build_dataset(seed=19, train_prefixes=16, validation_prefixes=16, test_prefixes=16)
    episode = dataset.train[0]
    cache = {statement.text: torch.randn(2, 16) for statement in (*episode.prefix, *episode.tail)}
    writer = DeltaSlotWriter(16, 8, key_width=4)
    with pytest.raises(ValueError, match="evaluation"):
        collect_states(writer, (episode,), cache)
    writer.eval().requires_grad_(False)
    geometry = delta_geometry(writer, cache)
    assert geometry["same_entity_opposite_value"]["n"] > 0
    assert geometry["cross_entity"]["n"] > 0
    assert 0 <= geometry["beta"]["min"] <= geometry["beta"]["max"] <= 1


def test_transition_counts_use_conditional_denominators_and_zero_defaults():
    result = transition_counts([True, True, False, False], [True, False, True, False])
    assert result == {
        "n": 4, "before_correct": 2, "after_correct": 2, "retained_correct": 1,
        "remained_wrong": 1, "damage": 1, "repair": 1,
        "damage_rate": .5, "repair_rate": .5, "change_pp_all": 0.0,
    }
    assert transition_counts([], []) == {
        "n": 0, "before_correct": 0, "after_correct": 0, "retained_correct": 0,
        "remained_wrong": 0, "damage": 0, "repair": 0,
        "damage_rate": None, "repair_rate": None, "change_pp_all": None,
    }
