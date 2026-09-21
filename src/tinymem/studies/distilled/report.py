"""Aggregate sealed supervised learned-writer readout artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

import torch
from safetensors.torch import load_file

from tinymem.reader.prompt import reader_exact_match
from tinymem.studies.distilled import fit as distilled_fit
from tinymem.studies.distilled import protocol as distilled_protocol
from tinymem.studies.distilled import replication as distilled_replication
from tinymem.studies.distilled import training as distilled_training
from tinymem.studies.delta.fit import load_state_records
from tinymem.studies.artifacts import file_hash, write_json
from tinymem.studies.distilled.scoring import (
    ENDPOINTS,
    READ_MODES,
    STATE_BYTES,
    _assert_diagnostic_rows_equal,
    _assert_state_roundtrip,
    _direct_bit_metrics,
    _load_evaluation_features,
    _load_jsonl,
    _off_fact_sse,
    _paired_bootstrap,
    _parent_provenance,
    _record_map,
    _selected_episodes,
    _state_sse,
    _tensor_manifest_hash,
    _trajectory_summary,
    _validate_diagnostics,
    _validate_donor_rows,
    _validate_row,
)
from tinymem.studies.oracle.state import oracle_records


def aggregate_study(study: Path, protocol: dict) -> dict[str, Any]:
    """Validate all cells and produce descriptive learned-state readout metrics."""

    study = Path(study)
    report_path = study / "report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    distilled_protocol.require_training_seal(study, protocol)
    # The protocol module owns full source/data provenance checks.
    distilled_protocol.verify_study(study, study / "source")
    provenance = _parent_provenance(study, protocol)
    dataset = distilled_protocol.read_dataset(study / "dataset.json")
    settings = protocol["settings"]
    episodes = _selected_episodes(dataset.test, settings)
    donor_episodes = tuple(episode for episode in dataset.test if episode.condition == "no_write")
    by_id = {episode.id: episode for episode in episodes}
    expected_keys = {(episode.id, endpoint, entity) for episode in episodes
                     for endpoint in ((8,) if not episode.tail else ENDPOINTS) for entity in range(4)}
    all_rows: list[dict[str, Any]] = []
    trajectory_by_seed: dict[str, dict[str, Any]] = {}
    cell_reports: list[dict[str, Any]] = []
    training_seal = file_hash(study / "training_sealed.json")
    evaluation_cells = distilled_replication.evaluation_cells(protocol)
    expected_cells = {int(cell["index"]) for cell in evaluation_cells}
    writer_reference: dict[int, Path] = {}
    evaluation_root = study / "evaluation"
    actual_dirs = {int(path.name) for path in evaluation_root.iterdir() if path.is_dir()} if evaluation_root.is_dir() else set()
    if actual_dirs != expected_cells:
        raise ValueError("evaluation cells omit or add declared cells")
    for index, cell in enumerate(evaluation_cells):
        directory = study / "evaluation" / str(index)
        identity = {**distilled_protocol.evaluation_identity(study, protocol, index),
                    "training_seal_sha256": training_seal}
        completion = distilled_protocol.verify_completion(directory, identity)
        required = {"predictions.jsonl", "report.json", "diagnostics.json",
                     "features.safetensors", "feature_texts.json",
                     "states.safetensors", "states.json", "donor_states.safetensors", "donor_states.json",
                     "oracle_states.safetensors", "oracle_states.json", "oracle_donor_states.safetensors",
                     "oracle_donor_states.json"}
        if set(completion["files"]) != required:
            raise ValueError("distilled evaluation output schema differs")
        rows = _load_jsonl(directory / "predictions.jsonl")
        seen: set[tuple[Any, ...]] = set()
        for row in rows:
            row = _validate_row(row, by_id, str(cell["seed"]))
            if "replication" in settings and row.get("reader_seed") != str(cell["reader_seed"]):
                raise ValueError("evaluation row reader differs from its sealed cell")
            key = (row["episode_id"], row["after_write"], row["entity"])
            if key in seen:
                raise ValueError("duplicate distilled evaluation row")
            seen.add(key)
        if seen != expected_keys:
            raise ValueError("distilled evaluation omits or adds test cases")
        _validate_donor_rows(rows, episodes, donor_episodes)
        learned = load_state_records(directory / "states.safetensors")
        learned_donor = load_state_records(directory / "donor_states.safetensors")
        oracle = load_state_records(directory / "oracle_states.safetensors")
        oracle_donor = load_state_records(directory / "oracle_donor_states.safetensors")
        learned_map = _record_map(learned, episodes)
        oracle_map = _record_map(oracle, episodes)
        _record_map(learned_donor, donor_episodes)
        _record_map(oracle_donor, donor_episodes)
        canonical_oracle = oracle_records(episodes)
        canonical_donor = oracle_records(donor_episodes)
        _assert_state_roundtrip(canonical_oracle, oracle)
        _assert_state_roundtrip(canonical_donor, oracle_donor)
        all_episodes = tuple({episode.id: episode for episode in (*episodes, *donor_episodes)}.values())
        features = _load_evaluation_features(directory, all_episodes)
        writer_index = distilled_replication.training_index(cell)
        writer = distilled_fit.load_writer(study, protocol, writer_index)
        if writer_index in writer_reference:
            reference = writer_reference[writer_index]
            _assert_state_roundtrip(load_state_records(reference / "states.safetensors"), learned)
            _assert_state_roundtrip(load_state_records(reference / "donor_states.safetensors"), learned_donor)
            reference_features = _load_evaluation_features(reference, all_episodes)
            if any(not torch.equal(features[text], reference_features[text]) for text in features):
                raise ValueError("writer input features differ across reader evaluations")
        else:
            writer_reference[writer_index] = directory
        recomputed_diagnostics = distilled_training.trajectory_diagnostics(
            writer, episodes, features,
        )
        diagnostics = json.loads((directory / "diagnostics.json").read_text())
        _validate_diagnostics(diagnostics, episodes, learned_map)
        _validate_diagnostics(recomputed_diagnostics, episodes, learned_map)
        _assert_diagnostic_rows_equal(diagnostics, recomputed_diagnostics)
        by_row_key = {(row["episode_id"], row["after_write"], row["entity"]): row for row in rows}
        state_sse = []
        off_fact_sse = []
        direct_correct = 0
        direct_total = 0
        for record_key, learned_record in learned_map.items():
            gold = oracle_map[record_key]
            state_sse.append(_state_sse(learned_record, gold))
            off_fact_sse.append(_off_fact_sse(learned_record, gold))
            bits = _direct_bit_metrics(learned_record, gold.truth)
            direct_correct += bits["known_fact_correct"]
            direct_total += bits["known_fact_count"]
            for entity in range(4):
                row = by_row_key[(record_key[0], record_key[-1], entity)]
                if row["direct_bits"] != bits["direct_bits"]:
                    raise ValueError("row direct bits disagree with serialized state")
                if abs(row["state_sse"] - state_sse[-1]) > 1e-6 or abs(row["off_fact_sse"] - off_fact_sse[-1]) > 1e-6:
                    raise ValueError("row state metrics disagree with serialized states")
        stored = json.loads((directory / "report.json").read_text())
        if (stored.get("cell") != cell or stored.get("rows") != len(rows) or stored.get("state_bytes") != STATE_BYTES
                or stored.get("modes") != list(READ_MODES)
                or stored.get("parent_parameters_unchanged") is not True):
            raise ValueError("distilled cell report provenance or counts disagree")
        parent_training = study / "parent" / "training" / str(cell["parent_index"])
        parent_report = json.loads((parent_training / "report.json").read_text())
        parent_checkpoint_hash = _tensor_manifest_hash(
            load_file(str(parent_training / "checkpoint.safetensors"), device="cpu")
        )
        training_writer_report = json.loads((study / "training" / str(writer_index) / "report.json").read_text())
        runtime_fields = ("device", "writer_device", "writer_dtype", "torch_version",
                          "deterministic", "packages")
        if (stored.get("runtime") != parent_report.get("runtime")
                or any(stored.get("writer_runtime", {}).get(field)
                       != training_writer_report.get("runtime", {}).get(field)
                       for field in runtime_fields)
                or stored.get("parent_parameters_sha256_before") != parent_checkpoint_hash
                or stored.get("parent_parameters_sha256_before") != stored.get("parent_parameters_sha256_after")
                or stored.get("unadapted_base_sha256") != parent_report.get("unadapted_base_sha256")
                or stored.get("adapted_base_sha256_before") != parent_report.get("base_after_sha256")
                or stored.get("adapted_base_sha256_after") != stored.get("adapted_base_sha256_before")):
            raise ValueError("distilled cell reports mutated parent reader parameters")
        if not stored.get("raw_state_read"):
            raise ValueError("distilled cell does not document raw state reads")
        all_rows.extend(rows)
        trajectory_key = (f'{cell["seed"]}/reader{cell["reader_seed"]}'
                          if "replication" in settings else str(cell["seed"]))
        trajectory_by_seed[trajectory_key] = _trajectory_summary(diagnostics, str(cell["seed"]))
        cell_reports.append({"cell": cell, "rows": len(rows), "state_records": len(learned),
                             "mean_state_sse": fmean(state_sse), "mean_off_fact_sse": fmean(off_fact_sse),
                             "direct_bit_accuracy": direct_correct / direct_total,
                             "diagnostics": len(diagnostics), "report": stored})
    if not all_rows:
        raise ValueError("distilled evaluation contains no rows")
    stratified: dict[str, Any] = {}
    for wording in sorted({row["wording"] for row in all_rows}):
        for condition in sorted({row["condition"] for row in all_rows}):
            for endpoint in ENDPOINTS:
                for scope in ("all", "target", "unspoken"):
                    selected = [row for row in all_rows if row["wording"] == wording
                                and row["condition"] == condition and row["after_write"] == endpoint
                                and row["scope"] == scope]
                    if not selected:
                        continue
                    by_seed: dict[str, list[dict[str, Any]]] = defaultdict(list)
                    for row in selected:
                        by_seed[str(row["writer_seed"])].append(row)
                    per_seed = {seed: {
                                    **{mode: fmean(float(row["reads"][mode]["correct"]) for row in values)
                                       for mode in READ_MODES},
                                    "direct_bit_accuracy": fmean(float(row["direct_bit_correct"])
                                                                  for row in values),
                                    "donor_following_accuracy": fmean(float(
                                        reader_exact_match(row["reads"]["donor"]["prediction"],
                                                           row["donor_answer"], "update_known"))
                                        for row in values),
                                }
                                for seed, values in by_seed.items()}
                    key = f"{wording}/{condition}/write{endpoint}/{scope}"
                    stratified[key] = {"n": len(selected), "per_seed": per_seed,
                                       "all_seed_mean": {
                                           **{mode: fmean(value[mode] for value in per_seed.values())
                                              for mode in READ_MODES},
                                           "direct_bit_accuracy": fmean(
                                               value["direct_bit_accuracy"] for value in per_seed.values()),
                                           "donor_following_accuracy": fmean(
                                               value["donor_following_accuracy"] for value in per_seed.values()),
                                       }}
                    if "replication" in settings:
                        by_reader: dict[str, list[dict]] = defaultdict(list)
                        for row in selected:
                            by_reader[f'{row["writer_seed"]}/reader{row["reader_seed"]}'].append(row)
                        stratified[key]["per_seed_reader"] = {
                            pair: {mode: fmean(float(row["reads"][mode]["correct"]) for row in values)
                                   for mode in READ_MODES} for pair, values in by_reader.items()}
    interpretation = {"evidence_kind": "supervised learned-writer readout through frozen successful readers",
                      "thresholds": "descriptive metrics only; no arbitrary pass/fail threshold",
                      "controls": "oracle, zero, and other-prefix donor states are scored from raw serialized states"}
    if "fixed_beta" in settings:
        interpretation["fixed_beta"] = settings["fixed_beta"]
    if settings.get("normalize_hidden", False):
        interpretation["normalize_hidden"] = True
        interpretation["hidden_normalization"] = settings["hidden_normalization"]
    report = {"schema": "distilled_fact_aggregate_v1",
              "protocol_sha256": file_hash(study / "protocol.json"),
              "training_seal_sha256": training_seal, "parent": provenance,
              "cells": cell_reports, "all_rows": len(all_rows), "stratified": stratified,
              "trajectory": trajectory_by_seed,
              "primary": {"correction_target_write16": {
                  control: _paired_bootstrap(all_rows, control, condition="correction", scope="target",
                                             settings=settings) for control in ("oracle", "zero", "donor")},
                          "repeat_unspoken_write16": {
                  control: _paired_bootstrap(all_rows, control, condition="repeat", scope="unspoken",
                                             settings=settings) for control in ("oracle", "zero", "donor")}},
              "state_metrics": {"state_bytes": STATE_BYTES,
                                "direct_bit_definition": "sign of exact fact cells; exact zero is undefined and counted incorrect",
                                "readout_state_sse": "raw learned values versus canonical oracle values",
                                "off_fact_sse": "all serialized cells except the four first-slot fact cells"},
              "wordings": sorted({row["wording"] for row in all_rows}),
              "interpretation": interpretation}
    write_json(report_path, report)
    return report
