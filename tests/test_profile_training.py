"""Disposable profile checkpoints exercise a real tiny Qwen reader."""

from dataclasses import replace
import json

import pytest

from test_delta_training import example
from tinymem.studies.artifacts import CELLS, file_hash, frozen_base_hash, profile_cell
from tinymem.reader.lora import attach_reader_lora
from tinymem.studies.delta.data import build_dataset
from scripts.profile_training import training_schedule


@pytest.mark.parametrize("kind,width", CELLS)
def test_profile_trains_reloads_and_preserves_base(tiny_reader, tmp_path, kind, width):
    attach_reader_lora(tiny_reader, rank=2, checkpointing=False)
    before = frozen_base_hash(tiny_reader)
    path = tmp_path / "cell"
    report = profile_cell(tiny_reader, ((example(),), (example(),)), path,
                          kind=kind, width=width, seed=19, warmup_steps=1, max_new_tokens=2)
    assert frozen_base_hash(tiny_reader) == before
    assert report["checkpoint_and_state_roundtrip"] is True
    assert report["parameter_groups_changed"] == dict(writer=True, bridge=True, adapter=True)
    assert report["persistent_bytes"] == 2 * width * 4 + 2
    assert report["accuracy_scored"] is False and report["checkpoint_reuse"] is False
    assert report == json.loads((path / "report.json").read_text())
    assert all(file_hash(path / name) == digest for name, digest in report["files"].items())
    assert all(not p.requires_grad and p.grad is None for p in tiny_reader.model.parameters())
    with pytest.raises(FileExistsError):
        profile_cell(tiny_reader, ((example(),),), path, kind=kind, width=width, seed=19, warmup_steps=0)


def test_profile_rejects_heldout_before_creating_output(tiny_reader, tmp_path):
    with pytest.raises(ValueError, match="training"):
        profile_cell(tiny_reader, ((replace(example(), split="test"),),), tmp_path / "cell",
                     kind="delta", width=8, seed=19, warmup_steps=0)
    assert not (tmp_path / "cell").exists()


def test_schedule_is_fixed_and_covers_all_write_conditions():
    dataset = build_dataset(train_prefixes=16, validation_prefixes=16, test_prefixes=16)
    schedule = training_schedule(dataset.train, steps=4, batch_size=2, seed=19)
    assert schedule == training_schedule(dataset.train, steps=4, batch_size=2, seed=19)
    assert [batch[0].condition for batch in schedule] == ["no_write", "repeat", "correction", "balanced"]
    assert all(len(batch) == 2 and all(e.split == "train" for e in batch) for batch in schedule)
    with pytest.raises(ValueError, match="training"):
        training_schedule(dataset.test, steps=4, batch_size=2, seed=19)
