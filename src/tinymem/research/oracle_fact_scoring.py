"""Score the privileged, known-correct state control for the fact task.

This module has a deliberately separate evaluation path from the learned-writer
comparison.  The oracle state is made by :mod:`oracle_fact_state`, then passed
through the same native memory boundary and reader generation contract.  It is
therefore useful for locating a readout failure without pretending that an
oracle write is a learned-memory result.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import random
from statistics import fmean
from typing import Any
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from tinymem.evaluation.reader_gate import reader_exact_match
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.delta_fact_data import ROOM_PAIRS, Episode, replay
from tinymem.research.delta_fact_encoding import _queries
from tinymem.research.delta_fact_evaluation import StateRecord
from tinymem.research.delta_fact_fit import load_state_records, save_state_records, selected_episodes
from tinymem.research.delta_fact_profile import file_hash, frozen_base_hash, write_json
from tinymem.research.delta_fact_readout import read_answer
from tinymem.research.delta_fact_protocol import cell_identity, seal_directory, verify_completion
from tinymem.research.oracle_fact_protocol import read_dataset, require_training_seal
from tinymem.research import oracle_fact_fit as fit
from tinymem.research.oracle_fact_state import oracle_bits, oracle_records


STATE_WIDTH = 32
STATE_BYTES = 2 * STATE_WIDTH * 4 + 2
ENDPOINTS = (8, 9, 16)
READ_FIELDS = {"prediction", "generated_ids", "input_positions", "memory_positions",
               "native_envelope_tokens", "correct"}


def _record_metadata(record: StateRecord) -> dict[str, Any]:
    values = asdict(record)
    return {key: value for key, value in values.items() if key not in ("values", "valid")}


def _owned_state(record: StateRecord) -> LatentSlotState:
    values = record.values
    valid = record.valid
    if (not isinstance(values, torch.Tensor) or not isinstance(valid, torch.Tensor)
            or values.shape != (1, 2, STATE_WIDTH) or valid.shape != (1, 2)
            or values.dtype != torch.float32 or valid.dtype != torch.bool
            or values.device.type != "cpu" or valid.device.type != "cpu"
            or values.requires_grad or valid.requires_grad or not bool(torch.isfinite(values).all())
            or not bool(valid.all())):
        raise ValueError("oracle records must own finite CPU FP32 258-byte states")
    if values.untyped_storage().nbytes() + valid.untyped_storage().nbytes() != STATE_BYTES:
        raise ValueError("oracle state does not have the declared 258 bytes")
    return LatentSlotState(values.detach().clone().contiguous(), valid.detach().clone().contiguous())


def _truth(episode: Episode, endpoint: int) -> tuple[int, ...]:
    if endpoint not in ENDPOINTS or (endpoint == 9 and not episode.tail) or (endpoint == 16 and not episode.tail):
        raise ValueError("endpoint is not declared for this episode")
    statements = (*episode.prefix, *episode.tail[:max(0, endpoint - 8)])
    result = replay(statements)
    if any(value is None for value in result):
        raise ValueError("test episode does not define all four facts")
    return tuple(int(value) for value in result)


def _oracle_bits(record: StateRecord, truth: tuple[int, ...]) -> tuple[int, ...]:
    bits = tuple(int(value) for value in oracle_bits(_owned_state(record)))
    if bits != truth:
        raise ValueError("oracle state sign decoding disagrees with independent replay")
    if len(bits) != 4 or any(value not in (0, 1) for value in bits):
        raise ValueError("oracle state must decode to four binary facts")
    return bits


def _record_identity(record: StateRecord) -> tuple[Any, ...]:
    return tuple(getattr(record, name) for name in
                 ("episode_id", "prefix_id", "wording", "condition", "target", "after_write"))


def _canonical_state_audit(records: Sequence[StateRecord], episodes: Sequence[Episode]) -> None:
    expected = {_record_identity(record): record for record in oracle_records(episodes)}
    actual = {_record_identity(record): record for record in records}
    if len(records) != len(actual) or actual.keys() != expected.keys():
        raise ValueError("oracle state payloads omit, repeat, or add endpoint records")
    for key, record in actual.items():
        canonical = expected[key]
        if (not torch.equal(record.values, canonical.values) or not torch.equal(record.valid, canonical.valid)
                or _record_metadata(record) != _record_metadata(canonical)):
            raise ValueError("saved oracle state differs from the declared code")


def _paired_state_audit(records: Sequence[StateRecord]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], dict[str, StateRecord]] = defaultdict(dict)
    for record in records:
        key = tuple(getattr(record, name) for name in
                    ("prefix_id", "condition", "target", "after_write"))
        wording = record.wording
        if wording in groups[key]:
            raise ValueError("duplicate oracle state wording for a logical case")
        groups[key][wording] = record
    paired = 0
    for values in groups.values():
        if set(values) != {"familiar", "heldout"}:
            raise ValueError("oracle test panel must contain paired familiar and heldout states")
        familiar, heldout = values["familiar"], values["heldout"]
        if not torch.equal(familiar.values, heldout.values) or not torch.equal(
                familiar.valid, heldout.valid):
            raise ValueError("paired logical familiar and heldout oracle states differ")
        paired += 1
    return {"paired_cases": paired, "state_invariant_across_wording": True,
            "interpretation": "heldout wording is a paired oracle-state audit, not learned wording generalization"}


def _paired_prediction_audit(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        identity = (row["prefix_id"], row["condition"], row["target"], row["after_write"], row["entity"])
        for mode, read in row["reads"].items():
            groups[(*identity, mode)][row["wording"]] = read
    checked = 0
    for values in groups.values():
        if set(values) != {"familiar", "heldout"}:
            raise ValueError("oracle readout panel lacks paired wordings")
        familiar, heldout = values["familiar"], values["heldout"]
        if any(familiar[field] != heldout[field] for field in READ_FIELDS if field != "correct"):
            raise ValueError("paired familiar and heldout oracle predictions differ")
        if familiar["correct"] != heldout["correct"]:
            raise ValueError("paired familiar and heldout oracle correctness differs")
        checked += 1
    return {"paired_prediction_cases": checked, "prediction_invariant_across_wording": True}


def _read_metadata(value: object, answer: str, category: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not (READ_FIELDS - {"correct"} <= set(value) <= READ_FIELDS):
        raise ValueError("oracle read metadata fields differ")
    prediction = value["prediction"]
    generated = value["generated_ids"]
    if type(prediction) is not str or type(generated) is not list:
        raise ValueError("oracle read metadata has invalid prediction fields")
    if any(type(token) is not int or token < 0 for token in generated) or not 1 <= len(generated) <= 8:
        raise ValueError("oracle generated token count differs from the fixed read contract")
    for name in ("input_positions", "memory_positions", "native_envelope_tokens"):
        if type(value[name]) is not int or value[name] < 0:
            raise ValueError("oracle read metadata has invalid position counts")
    if value["memory_positions"] != 2 or value["native_envelope_tokens"] <= 0:
        raise ValueError("oracle read metadata does not preserve the native envelope")
    if value["input_positions"] != value["memory_positions"] + value["native_envelope_tokens"]:
        raise ValueError("oracle read positions do not preserve the native envelope")
    correct = reader_exact_match(prediction, answer, category)
    if "correct" in value and (value["correct"] != correct or type(value["correct"]) is not bool):
        raise ValueError("stored oracle correctness disagrees with exact answer matching")
    return {**value, "correct": correct}


def _state_key(state: LatentSlotState, before_ids: tuple[int, ...], after_ids: tuple[int, ...]) -> tuple[Any, ...]:
    return (state.values.detach().cpu().contiguous().numpy().tobytes(),
            state.valid.detach().cpu().contiguous().numpy().tobytes(), before_ids, after_ids)


def _expected_donors(episodes: Sequence[Episode], donor_episodes: Sequence[Episode]) -> dict[tuple[str, str], Episode]:
    by_wording: dict[str, list[Episode]] = defaultdict(list)
    for episode in donor_episodes:
        by_wording[episode.wording].append(episode)
    donors = {}
    for wording, values in by_wording.items():
        ordered = sorted(values, key=lambda episode: episode.prefix_id)
        if len(ordered) < 2:
            raise ValueError("donor control requires at least two no-write prefixes per wording")
        for index, episode in enumerate(ordered):
            donors[episode.prefix_id, wording] = ordered[(index + 1) % len(ordered)]
    return {(episode.prefix_id, episode.wording): donors[episode.prefix_id, episode.wording]
            for episode in episodes}


def score_cell(reader, study: Path, protocol: dict, dataset, index: int) -> dict[str, Any]:
    """Score one fresh oracle reader cell and seal all outputs."""
    require_training_seal(study, protocol)
    settings, cell = protocol["settings"], protocol["cells"][index]
    directory = study / "evaluation" / str(index)
    directory.mkdir(parents=True, exist_ok=False)
    episodes = tuple(selected_episodes(dataset.test, settings))
    donor_episodes = tuple(episode for episode in dataset.test if episode.condition == "no_write")
    all_episodes = tuple({episode.id: episode for episode in (*episodes, *donor_episodes)}.values())
    donors = _expected_donors(episodes, donor_episodes)
    bridge = fit.load_trained(reader, protocol, cell, study / "training" / str(index) / "checkpoint.safetensors")
    base_before = frozen_base_hash(reader)
    training_report = json.loads((study / "training" / str(index) / "report.json").read_text())
    runtime_before = fit.execution_record(reader)
    if base_before != training_report["base_after_sha256"] or runtime_before != training_report["runtime"]:
        raise ValueError("oracle evaluation base model or execution environment differs from training")
    parameters_before = fit.checkpoint_tensors(reader, bridge)
    records = tuple(oracle_records(all_episodes))
    test_records = tuple(record for record in records if record.episode_id in {e.id for e in episodes})
    donor_records = tuple(record for record in records if record.episode_id in {e.id for e in donor_episodes})
    by_identity = {_record_identity(record): record for record in records}
    if len(by_identity) != len(records):
        raise ValueError("oracle records contain duplicate endpoint identities")
    for record in records:
        truth = _truth(next(e for e in (*episodes, *donor_episodes) if e.id == record.episode_id),
                       record.after_write)
        _oracle_bits(record, truth)
    _paired_state_audit(test_records)
    save_state_records(test_records, directory / "states.safetensors")
    save_state_records(donor_records, directory / "donor_states.safetensors")
    test_records = load_state_records(directory / "states.safetensors")
    donor_records = load_state_records(directory / "donor_states.safetensors")
    for original, loaded in zip(tuple(record for record in records if record.episode_id in {e.id for e in episodes}),
                                test_records, strict=True):
        if not torch.equal(original.values, loaded.values) or not torch.equal(original.valid, loaded.valid):
            raise ValueError("oracle state serialization changed an owned payload")
        _oracle_bits(loaded, tuple(int(v) for v in original.truth))
    for original, loaded in zip(tuple(record for record in records if record.episode_id in {e.id for e in donor_episodes}),
                                donor_records, strict=True):
        if not torch.equal(original.values, loaded.values) or not torch.equal(original.valid, loaded.valid):
            raise ValueError("oracle donor state serialization changed an owned payload")
        _oracle_bits(loaded, tuple(int(v) for v in original.truth))
    by_episode = {episode.id: episode for episode in episodes}
    donor_by_key = {(record.prefix_id, record.wording, record.after_write): record for record in donor_records}
    predictions: dict[tuple[Any, ...], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    with (directory / "predictions.jsonl").open("x") as handle:
        for record in test_records:
            episode = by_episode[record.episode_id]
            endpoint = record.after_write
            current = (*episode.prefix, *episode.tail[:max(0, endpoint - 8)])
            before_ids, queries = _queries(reader, episode, current, endpoint)
            state = _owned_state(record)
            donor_episode = donors[(episode.prefix_id, episode.wording)]
            donor = donor_by_key[(donor_episode.prefix_id, donor_episode.wording, 8)]
            donor_state = _owned_state(donor)
            zero = LatentSlotState(torch.zeros_like(state.values), torch.ones_like(state.valid))
            truth = tuple(int(value) for value in record.truth)
            for entity, query in enumerate(queries):
                answer = ROOM_PAIRS[entity][truth[entity]]
                donor_truth = int(donor.truth[entity])
                outputs = {}
                for mode, memory in (("real", state), ("zero", zero), ("donor", donor_state)):
                    key = _state_key(memory, before_ids, query.after_ids)
                    if key not in predictions:
                        generated = read_answer(reader, bridge, LatentSlotState(
                            memory.values.to(reader.model.device), memory.valid.to(reader.model.device)),
                            torch.tensor(before_ids, device=reader.model.device),
                            torch.tensor(query.after_ids, device=reader.model.device),
                            max_new_tokens=settings["max_new_tokens"])
                        _read_metadata(generated, answer, query.category)
                        predictions[key] = dict(generated)
                    outputs[mode] = _read_metadata({**predictions[key], "correct":
                                                     reader_exact_match(predictions[key]["prediction"], answer, query.category)},
                                                    answer, query.category)
                scope = "all" if episode.target is None else "target" if episode.target == entity else "unspoken"
                rows.append({**_record_metadata(record), "writer_seed": str(cell["seed"]),
                             "key": f"{episode.id}/write{endpoint}/entity{entity}",
                             "entity": entity, "answer": answer, "scope": scope,
                             "truth_bit": truth[entity], "oracle_bits": list(_oracle_bits(record, truth)),
                             "reads": outputs, "donor_prefix": donor_episode.prefix_id,
                             "donor_truth_agrees": donor_truth == truth[entity],
                             "donor_answer": ROOM_PAIRS[entity][donor_truth]})
                handle.write(json.dumps(rows[-1], allow_nan=False) + "\n")
            handle.flush()
    if frozen_base_hash(reader) != base_before:
        raise ValueError("oracle evaluation mutated the reader base")
    parameters_after = fit.checkpoint_tensors(reader, bridge)
    if parameters_before.keys() != parameters_after.keys() or any(
            not torch.equal(parameters_before[name], parameters_after[name]) for name in parameters_before):
        raise ValueError("oracle evaluation mutated the trained bridge or adapter")
    pair_audit = {**_paired_state_audit(test_records), **_paired_prediction_audit(rows)}
    report = {"schema": "oracle_fact_cell_v1", "cell": cell, "rows": len(rows),
              "unique_generated_reads": len(predictions), "state_records": len(test_records),
              "state_bytes": STATE_BYTES, "paired_logical_states": pair_audit,
              "runtime": fit.execution_record(reader), "base_sha256_unchanged": base_before,
              "known_state_control": True,
              "interpretation": "privileged correct-state readout diagnostic; no learned-writer claim"}
    write_json(directory / "report.json", report)
    identity = {**cell_identity(study, protocol, index, "evaluation"),
                "training_seal_sha256": file_hash(study / "training_sealed.json")}
    seal_directory(directory, identity)
    return report


def _percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q, method="linear"))


def _paired_bootstrap(rows: Sequence[dict[str, Any]], control: str, *, condition: str,
                      scope: str, settings: Mapping[str, Any]) -> dict[str, Any]:
    selected = [row for row in rows if row["after_write"] == 16
                and row["condition"] == condition and row["scope"] == scope]
    if not selected:
        return {"status": "absent", "control": control}
    by_seed_prefix: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in selected:
        by_seed_prefix[str(row["writer_seed"]), row["prefix_id"]].append(
            float(row["reads"]["real"]["correct"]) - float(row["reads"][control]["correct"]))
    seeds = sorted({seed for seed, _ in by_seed_prefix})
    prefixes = sorted({prefix for _, prefix in by_seed_prefix})
    if any((seed, prefix) not in by_seed_prefix for seed in seeds for prefix in prefixes):
        raise ValueError("paired bootstrap requires complete seed and prefix coverage")
    per_seed = {seed: 100 * fmean(
        fmean(by_seed_prefix[seed, prefix]) for prefix in prefixes
    ) for seed in seeds}
    draws = []
    rng = random.Random(int(settings.get("bootstrap_seed", 8237)))
    for _ in range(int(settings.get("bootstrap_samples", 10000))):
        sampled = [rng.choice(prefixes) for _ in prefixes]
        draws.append(100 * fmean(
            fmean(fmean(by_seed_prefix[seed, prefix]) for prefix in sampled) for seed in seeds
        ))
    return {"status": "estimated", "control": control, "per_seed_gap_pp": per_seed,
            "mean_gap_pp": fmean(per_seed.values()),
            "interval_pp": [_percentile(draws, .5), _percentile(draws, 99.5)],
            "confidence": .99, "unit": "paired_prefixes_conditional_on_observed_seeds",
            "seeds": seeds, "prefixes": len(prefixes),
            "resamples": int(settings.get("bootstrap_samples", 10000)),
            "bootstrap_seed": int(settings.get("bootstrap_seed", 8237))}


def _validate_row(row: object, episodes: Mapping[str, Episode], writer_seed: str) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError("oracle evaluation rows must be objects")
    required = {"key", "episode_id", "prefix_id", "wording", "condition", "target", "after_write",
                "entity", "scope", "truth_bit", "answer", "oracle_bits", "reads", "donor_prefix",
                "donor_truth_agrees", "donor_answer", "writer_seed"}
    if not required.issubset(row):
        raise ValueError("oracle evaluation row is missing required fields")
    if str(row["writer_seed"]) != writer_seed:
        raise ValueError("oracle row writer seed differs from its sealed cell")
    episode = episodes.get(row["episode_id"])
    if episode is None:
        raise ValueError("oracle row refers to an undeclared episode")
    if row["prefix_id"] != episode.prefix_id or row["wording"] != episode.wording or row["condition"] != episode.condition:
        raise ValueError("oracle row metadata disagrees with the dataset")
    if row["target"] != episode.target:
        raise ValueError("oracle row target disagrees with the dataset")
    if type(row["after_write"]) is not int or row["after_write"] not in ENDPOINTS:
        raise ValueError("oracle row has an invalid endpoint")
    if type(row["entity"]) is not int or row["entity"] not in range(4):
        raise ValueError("oracle row has an invalid entity")
    truth = _truth(episode, row["after_write"])
    if row["truth_bit"] != truth[row["entity"]] or row["answer"] != ROOM_PAIRS[row["entity"]][truth[row["entity"]]]:
        raise ValueError("oracle row truth or answer disagrees with replay")
    expected_scope = "all" if episode.target is None else "target" if episode.target == row["entity"] else "unspoken"
    if row["scope"] != expected_scope:
        raise ValueError("oracle row scope disagrees with the dataset")
    if tuple(row["oracle_bits"]) != truth:
        raise ValueError("oracle row sign audit disagrees with replay")
    if not isinstance(row["reads"], dict) or set(row["reads"]) != {"real", "zero", "donor"}:
        raise ValueError("oracle row controls differ")
    for read in row["reads"].values():
        if not isinstance(read, dict) or set(read) != READ_FIELDS:
            raise ValueError("stored oracle read fields differ")
        _read_metadata(read, row["answer"], "update_known")
    if type(row["donor_truth_agrees"]) is not bool or not isinstance(row["donor_prefix"], str):
        raise ValueError("oracle donor metadata is invalid")
    if not isinstance(row["donor_answer"], str):
        raise ValueError("oracle donor answer metadata is invalid")
    expected_key = f'{row["episode_id"]}/write{row["after_write"]}/entity{row["entity"]}'
    if row["key"] != expected_key:
        raise ValueError("oracle row key differs")
    return row


def _validate_donor_rows(rows: Sequence[dict[str, Any]], episodes: Sequence[Episode],
                         donor_episodes: Sequence[Episode]) -> None:
    expected = _expected_donors(episodes, donor_episodes)
    no_write = {(episode.prefix_id, episode.wording): tuple(int(value) for value in replay(episode.prefix))
                for episode in donor_episodes}
    by_id = {episode.id: episode for episode in episodes}
    for row in rows:
        episode = by_id[row["episode_id"]]
        donor = expected[(episode.prefix_id, episode.wording)]
        if row["donor_prefix"] != donor.prefix_id:
            raise ValueError("oracle donor prefix differs from the declared cyclic donor")
        agrees = no_write[donor.prefix_id, donor.wording][row["entity"]] == row["truth_bit"]
        if row["donor_truth_agrees"] != agrees:
            raise ValueError("oracle donor truth agreement disagrees with replay")
        donor_answer = ROOM_PAIRS[row["entity"]][no_write[donor.prefix_id, donor.wording][row["entity"]]]
        if row["donor_answer"] != donor_answer:
            raise ValueError("oracle donor answer disagrees with the donor truth")


def aggregate_study(study: Path, protocol: dict) -> dict[str, Any]:
    """Validate sealed oracle cells and write a stratified, descriptive report."""
    study = Path(study)
    report_path = study / "report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    require_training_seal(study, protocol)
    dataset = read_dataset(study / "dataset.json")
    settings = protocol["settings"]
    episodes = tuple(selected_episodes(dataset.test, settings))
    episodes_by_id = {episode.id: episode for episode in episodes}
    expected_keys = {(episode.id, endpoint, entity) for episode in episodes
                     for endpoint in ((8,) if not episode.tail else ENDPOINTS) for entity in range(4)}
    all_rows: list[dict[str, Any]] = []
    cell_reports = []
    training_seal = file_hash(study / "training_sealed.json")
    for index, cell in enumerate(protocol["cells"]):
        directory = study / "evaluation" / str(index)
        identity = {**cell_identity(study, protocol, index, "evaluation"),
                    "training_seal_sha256": training_seal}
        completion = verify_completion(directory, identity)
        if set(completion["files"]) != {"predictions.jsonl", "report.json", "states.json", "states.safetensors",
                                        "donor_states.json", "donor_states.safetensors"}:
            raise ValueError("oracle evaluation output schema differs")
        rows = []
        seen = set()
        with (directory / "predictions.jsonl").open() as handle:
            for line in handle:
                if not line.strip():
                    raise ValueError("blank oracle evaluation row")
                row = _validate_row(json.loads(line), episodes_by_id, str(cell["seed"]))
                identity_key = (row["episode_id"], row["after_write"], row["entity"])
                if identity_key in seen:
                    raise ValueError("duplicate oracle evaluation row")
                seen.add(identity_key)
                rows.append(row)
        if seen != expected_keys:
            raise ValueError("oracle evaluation omits or adds test cases")
        _validate_donor_rows(rows, episodes, tuple(episode for episode in dataset.test if episode.condition == "no_write"))
        state_records = load_state_records(directory / "states.safetensors")
        donor_state_records = load_state_records(directory / "donor_states.safetensors")
        _canonical_state_audit(state_records, episodes)
        _canonical_state_audit(donor_state_records, tuple(e for e in dataset.test if e.condition == "no_write"))
        _paired_state_audit(state_records)
        prediction_audit = _paired_prediction_audit(rows)
        stored = json.loads((directory / "report.json").read_text())
        if stored.get("rows") != len(rows) or stored.get("state_bytes") != STATE_BYTES:
            raise ValueError("oracle cell report counts or state bytes disagree")
        if not stored.get("paired_logical_states", {}).get("state_invariant_across_wording"):
            raise ValueError("oracle cell does not document paired logical state invariance")
        if not stored.get("paired_logical_states", {}).get("prediction_invariant_across_wording"):
            raise ValueError("oracle cell does not document paired prediction invariance")
        training_report = json.loads((study / "training" / str(index) / "report.json").read_text())
        if (stored.get("runtime") != training_report.get("runtime")
                or stored.get("base_sha256_unchanged") != training_report.get("base_after_sha256")):
            raise ValueError("oracle evaluation base or runtime differs from its sealed training cell")
        all_rows.extend(rows)
        cell_reports.append({"cell": cell, "rows": len(rows), "report": stored,
                             "paired_predictions": prediction_audit})
    if not all_rows:
        raise ValueError("oracle evaluation contains no rows")
    dependence = {}
    for cell in protocol["cells"]:
        selected = [row for row in all_rows if row["writer_seed"] == str(cell["seed"])]
        dependence[str(cell["seed"])] = {
            "rows": len(selected),
            "prediction_disagreement": {control: sum(
                row["reads"]["real"]["prediction"] != row["reads"][control]["prediction"]
                for row in selected) for control in ("zero", "donor")}}
    # Each mode is validated against the generated text above.  Recompute all
    # absolute rates from rows so a mutated stored metric cannot survive.
    stratified: dict[str, Any] = {}
    for wording in ("familiar", "heldout"):
        for condition in ("no_write", "repeat", "correction", "balanced"):
            for endpoint in ENDPOINTS:
                for scope in ("all", "target", "unspoken"):
                    selected = [row for row in all_rows if row["wording"] == wording and row["condition"] == condition
                                and row["after_write"] == endpoint and row["scope"] == scope]
                    if not selected:
                        continue
                    key = f"{wording}/{condition}/write{endpoint}/{scope}"
                    by_seed = defaultdict(list)
                    for row in selected:
                        for mode in ("real", "zero", "donor"):
                            by_seed[str(row["writer_seed"])].append((mode, row["reads"][mode]["correct"]))
                    per_seed = {}
                    for seed, values in by_seed.items():
                        per_seed[seed] = {mode: sum(value for current, value in values if current == mode) /
                                          sum(current == mode for current, _ in values)
                                          for mode in ("real", "zero", "donor")}
                    stratified[key] = {"n": len(selected), "per_seed": per_seed,
                                       "all_seed_mean": {mode: fmean(value[mode] for value in per_seed.values())
                                                         for mode in ("real", "zero", "donor")}}
    report = {"schema": "oracle_fact_aggregate_v1", "protocol_sha256": file_hash(study / "protocol.json"),
              "training_seal_sha256": training_seal, "cells": cell_reports,
              "all_rows": len(all_rows), "stratified": stratified, "memory_dependence": dependence,
              "primary": {"correction_target_write16": {
                  control: _paired_bootstrap(all_rows, control, condition="correction", scope="target",
                                             settings=settings) for control in ("zero", "donor")},
                  "repeat_unspoken_write16": {
                  control: _paired_bootstrap(all_rows, control, condition="repeat", scope="unspoken",
                                             settings=settings) for control in ("zero", "donor")}},
              "bootstrap": {"confidence": .99, "interval": "99% percentile paired-prefix",
                            "optimization_seeds": "fixed observed seeds; not resampled",
                            "wordings": "identical paired wordings averaged within prefix, not independent samples",
                            "multiplicity": "pointwise; no family-wise correction"},
              "interpretation": {"evidence_kind": "privileged known-correct state readout diagnostic",
                                 "wording": "familiar and heldout rows share parsed oracle states; this does not test learned wording generalization",
                                 "real_control": "real is the oracle state; zero and donor are readout controls",
                                 "thresholds": "descriptive metrics only; no arbitrary pass/fail threshold"}}
    write_json(report_path, report)
    return report
