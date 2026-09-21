from dataclasses import replace

import pytest
import torch
from safetensors.torch import load_file, save_file

from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.studies.delta.data import build_dataset
from tinymem.studies.oracle.state import (
    ORACLE_STATE_BYTES,
    oracle_bits,
    oracle_records,
)


def test_oracle_covers_all_conditions_wordings_and_endpoints_from_text():
    dataset = build_dataset(seed=17, train_prefixes=16, validation_prefixes=16, test_prefixes=16)
    episodes = tuple(row for row in dataset.test if row.prefix_id.endswith("p0000"))
    records = oracle_records(episodes)

    assert len(records) == 56
    assert {(row.wording, row.condition) for row in records} == {
        (wording, condition)
        for wording in ("familiar", "heldout")
        for condition in ("no_write", "repeat", "correction", "balanced")
    }
    assert {row.after_write for row in records} == {8, 9, 16}
    assert all(row.values.shape == (1, 2, 32) for row in records)
    assert all(row.values.device.type == "cpu" and row.values.dtype == torch.float32 for row in records)
    assert all(row.values.numel() * 4 + row.valid.numel() == ORACLE_STATE_BYTES for row in records)
    assert all(oracle_bits(LatentSlotState(row.values, row.valid)) == row.truth for row in records)
    by_episode = {episode.id: episode for episode in episodes}
    paired = {}
    for row in records:
        episode = by_episode[row.episode_id]
        expected = [None] * 4
        for statement in (*episode.prefix, *episode.tail)[:row.after_write]:
            expected[statement.entity] = statement.value
        assert oracle_bits(LatentSlotState(row.values, row.valid)) == tuple(expected)
        key = (row.prefix_id, row.condition, row.target, row.after_write)
        if key in paired:
            assert torch.equal(paired[key], row.values)
        paired[key] = row.values


def test_correction_changes_only_the_target_fact_cell():
    dataset = build_dataset(seed=19, train_prefixes=16, validation_prefixes=16, test_prefixes=16)
    episode = next(row for row in dataset.train if row.condition == "correction" and row.target == 2)
    records = oracle_records((episode,))
    before = next(row for row in records if row.after_write == 8).values
    corrected = next(row for row in records if row.after_write == 9).values
    target_offset = episode.target * 8

    changed = (before != corrected).reshape(-1)
    assert changed[target_offset]
    assert int(changed.sum()) == 1
    assert corrected.reshape(-1)[target_offset] * before.reshape(-1)[target_offset] < 0


def test_oracle_states_are_owned_and_survive_tensor_serialization(tmp_path):
    dataset = build_dataset(seed=23, train_prefixes=16, validation_prefixes=16, test_prefixes=16)
    row = oracle_records((dataset.train[0],))[0]
    state = LatentSlotState(row.values, row.valid)
    assert state.values._base is None and state.valid._base is None
    assert state.values.is_contiguous() and state.valid.is_contiguous()
    assert oracle_bits(state) == row.truth

    path = tmp_path / "oracle.safetensors"
    save_file({"values": row.values, "valid": row.valid}, str(path))
    restored = load_file(str(path))
    restored_state = LatentSlotState(restored["values"].clone(), restored["valid"].clone())
    assert oracle_bits(restored_state) == row.truth
    with pytest.raises(ValueError, match="owned"):
        oracle_bits(LatentSlotState(row.values[:, :, :], row.valid))


def test_oracle_uses_statement_text_independently_of_metadata():
    dataset = build_dataset(seed=29, train_prefixes=16, validation_prefixes=16, test_prefixes=16)
    source = dataset.train[0]
    first = replace(source.prefix[0], entity=1 - source.prefix[0].entity,
                    value=1 - source.prefix[0].value)
    episode = replace(source, prefix=(first, *source.prefix[1:]))
    row = oracle_records((episode,))[0]
    assert oracle_bits(LatentSlotState(row.values, row.valid)) == row.truth


@pytest.mark.parametrize("bad", ["wrong_shape", "invalid", "nonfinite"])
def test_oracle_bits_rejects_invalid_storage(bad):
    values = torch.zeros(1, 2, 32)
    values[0, 0, [0, 8, 16, 24]] = 0.5
    valid = torch.ones(1, 2, dtype=torch.bool)
    if bad == "wrong_shape":
        values = torch.zeros(1, 2, 8)
    elif bad == "invalid":
        valid[0, 1] = False
    else:
        values[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        oracle_bits(LatentSlotState(values, valid))
