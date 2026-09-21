"""Replications keep training fixed and measure writers through separate readers."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from tinymem.studies.distilled import fit as distilled_fit
from tinymem.studies.distilled import report as distilled_report
from tinymem.studies.distilled import scoring as distilled_scoring
from tinymem.studies.distilled import protocol as protocol_module
from tinymem.studies.distilled import replication
from tinymem.studies.delta.data import (
    build_dataset,
    parse_statement,
    validate_dataset,
)


pytest_plugins = ["test_distilled_protocol"]

def signatures(episodes):
    return {tuple((s.entity, s.value) for s in e.prefix) for e in episodes}


def test_replication_preserves_training_and_excludes_all_previous_prefixes():
    original = build_dataset(seed=2026091407)
    fresh = replication.build_replication_dataset(original, smoke=False)
    assert fresh.train == original.train and fresh.validation == original.validation
    assert fresh == replication.build_replication_dataset(original, smoke=False)
    assert not signatures(fresh.test) & signatures((*original.train, *original.validation, *original.test))
    assert len(signatures(fresh.test)) == 64
    assert len(fresh.test) == 1280
    old_texts = {s.text for e in original.test for s in (*e.prefix, *e.tail) if e.wording == "heldout"}
    fresh_texts = {s.text for e in fresh.test for s in (*e.prefix, *e.tail) if e.wording == "heldout"}
    assert not old_texts & fresh_texts
    for episode in fresh.test:
        assert all(parse_statement(s.text) == s for s in (*episode.prefix, *episode.tail))
    paired = {}
    for episode in fresh.test:
        key = episode.prefix_id, episode.condition, episode.target
        logical = tuple((s.entity, s.value) for s in (*episode.prefix, *episode.tail))
        assert paired.setdefault(key, logical) == logical
    heldout = next(e for e in fresh.test if e.wording == "heldout")
    bad = replace(heldout.prefix[0], value=1-heldout.prefix[0].value)
    tampered = replace(heldout, prefix=(bad, *heldout.prefix[1:]))
    with pytest.raises(ValueError, match="metadata"):
        validate_dataset(replace(fresh, test=tuple(tampered if e.id == heldout.id else e for e in fresh.test)))


def test_each_fresh_writer_has_all_three_reader_evaluations():
    spec = protocol_module.settings(device="cuda", fixed_beta=.75, replicate=True)
    parent = {"settings": {"purpose": "privileged_correct_state_diagnostic"},
              "cells": [{"index": i, "seed": seed} for i, seed in enumerate((3101, 3102, 3103))]}
    cells = protocol_module.cells(spec, parent)
    protocol = {"settings": spec, "cells": cells, "readers": parent["cells"]}
    evaluations = replication.evaluation_cells(protocol)
    assert len(cells) == 12 and len(evaluations) == 36
    assert {c["seed"] for c in cells} == set(range(4101, 4113))
    assert all("parent_index" not in c for c in cells)
    assert {(c["seed"], c["reader_seed"]) for c in evaluations} == {
        (seed, reader) for seed in range(4101, 4113) for reader in (3101, 3102, 3103)}
    assert [replication.training_index(c) for c in evaluations] == [i for i in range(12) for _ in range(3)]
    old = {"settings": {}, "cells": [{"index": 0, "seed": 3101, "parent_index": 0}]}
    assert replication.evaluation_cells(old) is old["cells"]


def test_normalized_pairs_start_with_identical_parameters_and_rng():
    a = protocol_module.settings(device="cpu", fixed_beta=.75, replicate=True)
    b = protocol_module.settings(device="cpu", fixed_beta=.75, replicate=True, normalize_hidden=True)
    for seed in a["seeds"]:
        control = distilled_fit._writer(32, a, seed)
        rng = torch.get_rng_state().clone()
        normalized = distilled_fit._writer(32, b, seed)
        assert torch.equal(rng, torch.get_rng_state())
        assert control.state_dict().keys() == normalized.state_dict().keys()
        assert all(torch.equal(v, normalized.state_dict()[k]) for k, v in control.state_dict().items())


def test_replication_pipeline_trains_each_writer_once_and_checks_identities(writer_study):
    root, old_study, old_protocol, fresh_reader = writer_study
    study = old_study.with_name("replication")
    spec = protocol_module.settings(device="cpu", smoke=True, fixed_beta=.75, replicate=True)
    protocol = protocol_module.prepare_study(root, study, old_protocol["snapshot"], spec, old_study / "parent")
    verified, dataset = protocol_module.verify_study(study, root)
    assert verified == protocol and protocol["kind"] == "distilled_fact_replication_v1"
    protocol_module.prepare_features(fresh_reader(), study, protocol, dataset)
    for cell in protocol["cells"]:
        distilled_fit.train_cell(study, protocol, dataset, cell["index"])
    protocol_module.seal_training(study, protocol)
    for cell in replication.evaluation_cells(protocol):
        distilled_scoring.score_cell(fresh_reader(), study, protocol, dataset, cell["index"])
    result = distilled_report.aggregate_study(study, protocol)
    assert result["all_rows"] == 448
    assert len(result["cells"]) == 2
    assert set(result["trajectory"]) == {"4101/reader991", "4102/reader991"}
    assert all("per_seed_reader" in row for row in result["stratified"].values())
    from tinymem.studies.distilled.analysis import summarize_rows

    rows = [json.loads(line) for cell in replication.evaluation_cells(protocol)
            for line in (study / "evaluation" / str(cell["index"]) / "predictions.jsonl").read_text().splitlines()]
    prefix_ids = sorted({row["prefix_id"] for row in rows})
    summary = summarize_rows(rows, seeds=[4101, 4102], readers=[991], prefixes=prefix_ids)
    assert set(summary["endpoints"]) == {"correction", "retention"}
    assert len(summary["repeat_transitions"]) == 4
    duplicate = next(r for r in rows if r["condition"] == "repeat" and r["scope"] == "unspoken"
                     and r["after_write"] == 8)
    with pytest.raises(ValueError, match="duplicate|missing"):
        summarize_rows(rows + [duplicate], seeds=[4101, 4102], readers=[991], prefixes=prefix_ids)
    changed = deepcopy(protocol)
    changed["readers"][0]["seed"] += 1
    (study / "protocol.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="reader selection"):
        protocol_module.verify_study(study, root)


def test_seed_bootstrap_is_paired_and_reports_one_seed_sensitivity():
    from tinymem.studies.distilled.analysis import paired_seed_estimate

    control = {s: .5 for s in (1, 2, 3)}
    normalized = {1: 1.0, 2: .5, 3: .5}
    result = paired_seed_estimate(control, normalized, samples=1000, seed=4)
    assert result["difference"] == pytest.approx(1/6)
    assert result["positive_seeds"] == 1 and result["tied_seeds"] == 2
    assert result["leave_one_seed_out_differences"] == [0., .25, .25]
    same = paired_seed_estimate(control, control, samples=10)
    assert same["interval"] == [0., 0.]
    with pytest.raises(ValueError, match="same nonempty"):
        paired_seed_estimate(control, {1: .5})


def test_crossed_readers_reuse_identical_writer_states(writer_study, monkeypatch):
    from tinymem.studies.delta.fit import load_state_records
    from tinymem.studies.oracle import fit as oracle_fit, protocol as oracle_protocol

    root, old_study, old_protocol, fresh = writer_study
    parent_settings = oracle_protocol.settings

    def three_reader_fixture(*, device, smoke=False):
        spec = parent_settings(device=device, smoke=smoke)
        if smoke:
            spec["seeds"] = [991, 992, 993]
        return spec

    monkeypatch.setattr(oracle_protocol, "settings", three_reader_fixture)
    parent = old_study.with_name("three-readers")
    spec = oracle_protocol.settings(device="cpu", smoke=True)
    parent_protocol = oracle_protocol.prepare_study(root, parent, old_protocol["snapshot"], spec)
    parent_data = protocol_module.read_dataset(parent / "dataset.json")
    for cell in parent_protocol["cells"]:
        oracle_fit.train_cell(fresh(), parent, parent_protocol, parent_data, cell["index"])
    oracle_protocol.seal_training(parent, parent_protocol)
    study = old_study.with_name("crossed-replication")
    spec = protocol_module.settings(device="cpu", smoke=True, fixed_beta=.75, replicate=True)
    protocol = protocol_module.prepare_study(root, study, old_protocol["snapshot"], spec, parent)
    _, dataset = protocol_module.verify_study(study, root)
    protocol_module.prepare_features(fresh(), study, protocol, dataset)
    for cell in protocol["cells"]:
        distilled_fit.train_cell(study, protocol, dataset, cell["index"])
    protocol_module.seal_training(study, protocol)
    for cell in replication.evaluation_cells(protocol):
        distilled_scoring.score_cell(fresh(), study, protocol, dataset, cell["index"])
    report = distilled_report.aggregate_study(study, protocol)
    assert len(report["cells"]) == 6 and report["all_rows"] == 1344
    assert len(report["trajectory"]) == 6
    assert len(list((study / "training").iterdir())) == 2
    for writer_index in range(2):
        baseline = load_state_records(study / "evaluation" / str(writer_index * 3) / "states.safetensors")
        for reader_index in (1, 2):
            states = load_state_records(study / "evaluation" / str(writer_index * 3 + reader_index) / "states.safetensors")
            assert all(torch.equal(a.values, b.values) for a, b in zip(baseline, states, strict=True))
    (study / "report.json").rename(study / "original_report.json")
    directory = study / "evaluation/1"
    rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
    rows[0]["reader_seed"] = "991"
    (directory / "predictions.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    identity = {**protocol_module.evaluation_identity(study, protocol, 1),
                "training_seal_sha256": distilled_report.file_hash(study / "training_sealed.json")}
    (directory / "complete.json").unlink()
    protocol_module.seal_directory(directory, identity)
    with pytest.raises(ValueError, match="row reader"):
        distilled_report.aggregate_study(study, protocol)


def test_equal_answer_totals_remain_exactly_tied_across_reader_strata():
    from tinymem.studies.distilled.analysis import summarize_rows, paired_seed_estimate

    counts = {1: (77, 19, 685, 565, 431, 481), 2: (474, 734, 3, 291, 546, 210)}
    prefixes = [f"p{i}" for i in range(64)]
    rows = []
    for writer, correct_by_stratum in counts.items():
        for i, (reader, wording) in enumerate((r, w) for r in (1, 2, 3) for w in ("heldout", "familiar")):
            position = 0
            for prefix in prefixes:
                for target in range(4):
                    for entity in range(4):
                        condition = "correction" if entity == target else "repeat"
                        correct = entity != target and position < correct_by_stratum[i]
                        for endpoint in ((16,) if entity == target else (8, 16)):
                            rows.append({"writer_seed": str(writer), "reader_seed": str(reader),
                                         "prefix_id": prefix, "wording": wording, "condition": condition,
                                         "scope": "target" if entity == target else "unspoken",
                                         "after_write": endpoint, "target": target, "entity": entity,
                                         "episode_id": f"{prefix}/{condition}/{target}",
                                         "key": f"{prefix}/{condition}/{target}/{entity}/{endpoint}",
                                         "direct_bit_correct": correct,
                                         "reads": {m: {"correct": correct} for m in ("real", "oracle", "zero", "donor")}})
                        position += int(entity != target)
    summary = summarize_rows(rows, seeds=[1, 2], readers=[1, 2, 3], prefixes=prefixes)
    means = summary["endpoints"]["retention"]["per_seed"]
    assert means[1] == means[2] == 2258 / 4608
    estimate = paired_seed_estimate({1: means[1]}, {1: means[2]}, samples=10)
    assert estimate["tied_seeds"] == 1
    assert estimate["positive_seeds"] == estimate["negative_seeds"] == 0
