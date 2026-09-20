"""Real tiny-reader training through the sealed test and report workflow."""

from copy import deepcopy
import json

import pytest
import torch

from test_readout_runner import tiny_reader
from test_update_runner import WordTokenizer
from tinymem.research.delta_fact_fit import train_cell
from tinymem.research.delta_fact_protocol import prepare_study, read_dataset, seal_training, settings
from tinymem.research.delta_fact_scoring import packed_facts, score_cell, score_reference
from tinymem.research.delta_fact_report import aggregate_study
from tinymem.research.delta_fact_data import build_dataset, replay
from tinymem.research.pretrained import PretrainedReader


class BatchWordTokenizer(WordTokenizer):
    pad_token_id = 0

    def __call__(self, texts, *, padding, add_special_tokens, return_tensors):
        from transformers import BatchEncoding
        assert padding and not add_special_tokens and return_tensors == "pt"
        ids = [self.encode(text) for text in texts]
        size = max(map(len, ids))
        return BatchEncoding({
            "input_ids": torch.tensor([[0] * (size - len(row)) + row for row in ids]),
            "attention_mask": torch.tensor([[0] * (size - len(row)) + [1] * len(row) for row in ids]),
        })


def test_explicit_byte_tracks_values_and_known_flags_independently():
    data = build_dataset(train_prefixes=16, validation_prefixes=16, test_prefixes=16)
    for episode in data.train:
        for position in (4, 8, 9, 16):
            statements = (*episode.prefix, *episode.tail)[:position]
            packed = packed_facts(statements)
            assert packed.untyped_storage().nbytes() == 1
            assert tuple((int(packed[0]) >> entity) & 1 for entity in range(4)) == replay(statements)
            assert int(packed[0]) >> 4 == 15


def test_end_to_end_fixed_training_seal_reference_and_evaluation(tiny_reader, tmp_path):
    root = tmp_path / "source_root"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\n")
    (root / "uv.lock").write_text("version = 1\n")
    study = tmp_path / "study"
    protocol = prepare_study(root, study, {"tiny_fixture": True}, settings(device="cpu", smoke=True))
    dataset = read_dataset(study / "dataset.json")
    base = deepcopy(tiny_reader.model)
    tokenizer = BatchWordTokenizer()

    def fresh():
        return PretrainedReader(deepcopy(base), tokenizer)

    with pytest.raises(FileNotFoundError):
        score_reference(fresh(), study, protocol, dataset)
    trained = train_cell(fresh(), study, protocol, dataset, 0)
    assert trained["optimizer_steps"] == 4 and trained["test_scored"] is False
    assert trained["base_before_sha256"] == trained["base_after_sha256"]
    seal_training(study, protocol)
    reference = score_reference(fresh(), study, protocol, dataset)
    result = score_cell(fresh(), study, protocol, dataset, 0)
    assert result["cases"] == reference["cases_including_shared_prefix_duplicates"]
    rows = [json.loads(line) for line in (study / "evaluation/0/predictions.jsonl").read_text().splitlines()]
    assert {row["after_write"] for row in rows} == {8, 9, 16}
    assert all(row["donor_prefix"] != row["prefix_id"] for row in rows)
    assert all(row["explicit_correct"] for row in rows)
    assert all(set(row["reads"]) == {"memory", "zero", "donor"} for row in rows)
    reference_directory = study / "reference"
    reference_directory.rename(study / "reference_hidden")
    with pytest.raises(FileNotFoundError):
        aggregate_study(study, protocol)
    (study / "reference_hidden").rename(reference_directory)
    aggregate_study(study, protocol)
    assert (study / "report.json").exists()
    with pytest.raises(FileExistsError):
        train_cell(fresh(), study, protocol, dataset, 0)
