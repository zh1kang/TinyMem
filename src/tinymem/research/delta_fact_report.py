"""Validate and aggregate the sealed phase-two delta-fact evaluations.

The report treats a prefix as the sampling unit.  Rows inside a prefix are
paired before resampling, and the observed writer seeds are averaged equally.
This keeps the report descriptive of the declared seeds and evaluation panel.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import random
from statistics import fmean
from typing import Any

from tinymem.evaluation.longmemeval import normalized_answer
from tinymem.research.delta_fact_data import ROOM_PAIRS, replay
from tinymem.research.delta_fact_profile import file_hash
from tinymem.research.delta_fact_protocol import (
    cell_identity,
    read_dataset,
    require_training_seal,
    verify_completion,
)


_METHODS = ("lm", "probe", "zero", "donor", "base_full_text", "explicit")
_ENDPOINTS = (9, 16)
_CONDITIONS = ("repeat", "correction", "balanced")
_WORDINGS = ("familiar", "heldout")
_SCOPES = ("target", "unspoken", "all")
_READ_FIELDS = {"prediction", "generated_ids", "input_positions", "memory_positions",
                "native_envelope_tokens", "correct"}


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _row_method(row: dict[str, Any], method: str, *, endpoint: int) -> bool:
    if method == "lm":
        value = row["reads"]["memory"]["correct"]
    elif method == "probe":
        value = row["probe_correct"]
    elif method in ("zero", "donor"):
        value = row["reads"][method]["correct"]
    else:
        value = row[f"{method}_correct"]
    if type(value) is not bool:
        raise ValueError(f"{method} correctness must be boolean at endpoint {endpoint}")
    return value


def _validate_row(row: object) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError("evaluation rows must be JSON objects")
    required = {
        "key", "episode_id", "prefix_id", "wording", "condition", "target",
        "after_write", "entity", "scope", "truth_bit", "answer", "probe_correct",
        "probe_bit", "reads", "base_full_text_correct", "explicit_correct",
        "donor_prefix", "donor_truth_agrees",
    }
    if not required.issubset(row):
        raise ValueError("evaluation row is missing required fields")
    if (type(row["episode_id"]) is not str or not row["episode_id"]
            or type(row["prefix_id"]) is not str or not row["prefix_id"]):
        raise ValueError("evaluation row has an invalid episode identity")
    if row["wording"] not in _WORDINGS or row["condition"] not in ("no_write", *_CONDITIONS):
        raise ValueError("evaluation row has an invalid wording or condition")
    if row["target"] is not None and (type(row["target"]) is not int or row["target"] not in range(4)):
        raise ValueError("evaluation target must be null or an entity index")
    if type(row["after_write"]) is not int or row["after_write"] not in (8, 9, 16):
        raise ValueError("evaluation endpoint must be eight, nine, or sixteen")
    if type(row["entity"]) is not int or row["entity"] not in range(4):
        raise ValueError("evaluation entity must be an index from zero through three")
    if row["scope"] not in _SCOPES:
        raise ValueError("evaluation row has an invalid scope")
    if type(row["truth_bit"]) is not int or row["truth_bit"] not in (0, 1):
        raise ValueError("truth_bit must be zero or one")
    if type(row["answer"]) is not str or not row["answer"]:
        raise ValueError("answer must be a nonempty string")
    if type(row["probe_bit"]) is not int or row["probe_bit"] not in (0, 1):
        raise ValueError("probe_bit must be zero or one")
    if type(row["probe_correct"]) is not bool or row["probe_correct"] != (row["probe_bit"] == row["truth_bit"]):
        raise ValueError("probe correctness disagrees with probe_bit and truth_bit")
    expected_key = f'{row["episode_id"]}/write{row["after_write"]}/entity{row["entity"]}'
    if row["key"] != expected_key:
        raise ValueError("evaluation key does not match row identity")
    if not isinstance(row["reads"], dict):
        raise ValueError("evaluation reads must be an object")
    for mode in ("memory", "zero", "donor"):
        read = row["reads"].get(mode)
        if not isinstance(read, dict) or set(read) != _READ_FIELDS:
            raise ValueError(f"evaluation row has invalid {mode} read metadata")
        if type(read["correct"]) is not bool or type(read["prediction"]) is not str:
            raise ValueError(f"evaluation row has invalid {mode} read result")
        if (type(read["generated_ids"]) is not list
                or any(type(token) is not int or token < 0 for token in read["generated_ids"])):
            raise ValueError(f"evaluation row has invalid {mode} generated ids")
        for field in ("input_positions", "memory_positions", "native_envelope_tokens"):
            if type(read[field]) is not int or read[field] < 0:
                raise ValueError(f"evaluation row has invalid {mode} {field}")
        if (read["memory_positions"] != 2 or read["native_envelope_tokens"] <= 0
                or read["input_positions"] != 2 + read["native_envelope_tokens"]
                or not 1 <= len(read["generated_ids"]) <= 8):
            raise ValueError(f"evaluation row has inconsistent {mode} read metadata")
        if read["correct"] != (normalized_answer(read["prediction"]) == normalized_answer(row["answer"])):
            raise ValueError(f"evaluation row has inconsistent {mode} correctness")
    for method in ("probe", "base_full_text", "explicit"):
        if type(row["probe_correct"] if method == "probe" else row[f"{method}_correct"]) is not bool:
            raise ValueError(f"evaluation row has no boolean {method} result")
    if type(row["donor_prefix"]) is not str or not row["donor_prefix"] or row["donor_prefix"] == row["prefix_id"]:
        raise ValueError("donor must be a different nonempty prefix")
    if type(row["donor_truth_agrees"]) is not bool:
        raise ValueError("donor_truth_agrees must be boolean")
    if row["explicit_correct"] is not True:
        raise ValueError("explicit baseline must be correct")
    for field in ("state_norm", "last_update_norm", "last_residual_norm"):
        if field in row and row[field] is not None and (
                type(row[field]) not in (int, float) or not math.isfinite(row[field]) or row[field] < 0):
            raise ValueError(f"{field} must be a finite nonnegative number")
    if row["condition"] == "no_write":
        if row["target"] is not None or row["scope"] != "all" or row["after_write"] != 8:
            raise ValueError("no-write rows must be all-scope prefix rows")
    elif row["target"] is None:
        if row["condition"] != "balanced" or row["scope"] != "all":
            raise ValueError("only balanced rows may have no target")
    elif row["scope"] not in ("target", "unspoken"):
        raise ValueError("single-target rows must be target or unspoken scope")
    if row["target"] is not None:
        expected_scope = "target" if row["entity"] == row["target"] else "unspoken"
        if row["scope"] != expected_scope:
            raise ValueError("row scope does not match its target and entity")
    if row["condition"] == "balanced" and row["target"] is not None:
        raise ValueError("balanced rows cannot have a target")
    return row


def _load_rows(directory: Path, *, writer_seed: int) -> list[dict[str, Any]]:
    path = directory / "predictions.jsonl"
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                raise ValueError(f"blank evaluation row in {path}")
            row = _validate_row(json.loads(line))
            if "writer_seed" in row and row["writer_seed"] != writer_seed:
                raise ValueError("evaluation row writer_seed disagrees with its cell")
            row = {**row, "writer_seed": writer_seed}
            identity = (row["episode_id"], row["after_write"], row["entity"])
            if identity in seen:
                raise ValueError(f"duplicate evaluation row identity: {identity}")
            seen.add(identity)
            rows.append(row)
    if not rows:
        raise ValueError(f"evaluation rows are empty: {path}")
    return rows


def _metadata(rows: list[dict[str, Any]]) -> dict[tuple[str, int, int], tuple[Any, ...]]:
    return {
        (row["episode_id"], row["after_write"], row["entity"]): (
            row["prefix_id"], row["wording"], row["condition"], row["target"], row["scope"],
            row.get("truth_bit"), row.get("answer")
        )
        for row in rows
    }


def _expected_case_metadata(study: Path, settings: dict[str, Any]) -> dict[tuple[str, int, int], tuple[Any, ...]]:
    """Build the complete evaluation panel from the sealed dataset declaration."""
    dataset = read_dataset(study / "dataset.json")
    episodes = dataset.test
    limit = settings.get("evaluation_prefix_limit")
    if limit is not None:
        prefix_ids = sorted({episode.prefix_id for episode in episodes})[:limit]
        episodes = tuple(episode for episode in episodes if episode.prefix_id in prefix_ids)
    expected = {}
    for episode in episodes:
        endpoints = (8,) if not episode.tail else (8, 9, 16)
        for endpoint in endpoints:
            truth = replay((*episode.prefix, *episode.tail[:endpoint - 8]))
            for entity in range(4):
                scope = "all" if episode.target is None else (
                    "target" if episode.target == entity else "unspoken")
                key = (episode.id, endpoint, entity)
                expected[key] = (
                    episode.prefix_id, episode.wording, episode.condition, episode.target, scope,
                    truth[entity], ROOM_PAIRS[entity][truth[entity]],
                )
    return expected


def _validate_expected_case_metadata(rows: list[dict[str, Any]], expected: dict) -> None:
    actual = _metadata(rows)
    if set(actual) != set(expected):
        raise ValueError("evaluation rows do not match the declared dataset panel")
    if actual != expected:
        raise ValueError("evaluation row metadata disagrees with the declared dataset panel")


def _dataset_no_write_truths(study: Path) -> dict[tuple[str, str], tuple[int, ...]]:
    dataset = read_dataset(study / "dataset.json")
    result = {}
    for episode in dataset.test:
        if episode.condition == "no_write":
            truth = replay(episode.prefix)
            result[episode.prefix_id, episode.wording] = tuple(int(value) for value in truth)
    return result


def _validate_donor_rows(rows: list[dict[str, Any]], truths: dict[tuple[str, str], tuple[int, ...]]) -> None:
    for row in rows:
        donor_truth = truths.get((row["donor_prefix"], row["wording"]))
        if donor_truth is None:
            raise ValueError("donor prefix is not a declared no-write prefix for its wording")
        expected = donor_truth[row["entity"]] == row["truth_bit"]
        if row["donor_truth_agrees"] != expected:
            raise ValueError("donor_truth_agrees disagrees with the declared donor prefix")


def _reference_identity(study: Path) -> dict[str, str]:
    # Keep this lazy because the scoring module also imports optional evaluation machinery.
    from tinymem.research.delta_fact_scoring import reference_identity
    return reference_identity(study)


def _validate_reference_generation(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {"prediction", "generated_ids", "prompt_tokens"}:
        raise ValueError("reference generation metadata fields differ")
    if type(value["prediction"]) is not str or type(value["generated_ids"]) is not list:
        raise ValueError("reference generation metadata has invalid prediction fields")
    if any(type(token) is not int or token < 0 for token in value["generated_ids"]):
        raise ValueError("reference generated ids must be nonnegative integers")
    if (type(value["prompt_tokens"]) is not int or value["prompt_tokens"] <= 0
            or not 1 <= len(value["generated_ids"]) <= 8):
        raise ValueError("reference token counts differ from the read contract")


def _validate_reference(study: Path, evaluation_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    directory = study / "reference"
    verify_completion(directory, _reference_identity(study))
    rows = []
    seen = set()
    with (directory / "predictions.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                raise ValueError("blank reference row")
            row = json.loads(line)
            required = {"key", "answer", "base_full_text", "base_full_text_correct", "explicit_correct", "explicit_bytes"}
            if not isinstance(row, dict) or not required.issubset(row) or row["key"] in seen:
                raise ValueError("reference rows have missing or duplicate identities")
            if type(row["answer"]) is not str or type(row["base_full_text_correct"]) is not bool:
                raise ValueError("reference answer or correctness is invalid")
            if row["explicit_correct"] is not True or type(row["explicit_bytes"]) is not int or row["explicit_bytes"] != 1:
                raise ValueError("reference explicit baseline must be correct")
            _validate_reference_generation(row["base_full_text"])
            if row["base_full_text_correct"] != (normalized_answer(row["base_full_text"]["prediction"]) == normalized_answer(row["answer"])):
                raise ValueError("reference correctness disagrees with its generated answer")
            seen.add(row["key"])
            rows.append(row)
    expected = {(row["key"], row["answer"], row["base_full_text_correct"]) for row in rows}
    evaluation = {(row["key"], row["answer"], row["base_full_text_correct"]) for row in evaluation_rows}
    if expected != evaluation or len(rows) != len(evaluation_rows):
        raise ValueError("reference rows disagree with evaluation answers or base correctness")
    report = json.loads((directory / "report.json").read_text())
    if (report.get("cases_including_shared_prefix_duplicates") != len(rows)
            or report.get("base_full_text_correct") != sum(row["base_full_text_correct"] for row in rows)):
        raise ValueError("reference report counts disagree with reference rows")
    training = json.loads((study / "training" / "0" / "report.json").read_text())
    if (report.get("base_sha256_unchanged") != training.get("unadapted_base_sha256")
            or report.get("runtime") != training.get("runtime")):
        raise ValueError("reference report provenance disagrees with training")
    return {row["key"]: row for row in rows}


def _validate_evaluation_report(directory: Path, rows: list[dict[str, Any]], training: dict[str, Any]) -> None:
    report = json.loads((directory / "report.json").read_text())
    if (report.get("cases") != len(rows)
            or report.get("base_sha256_unchanged") != training.get("base_after_sha256")
            or report.get("runtime") != training.get("runtime")):
        raise ValueError("evaluation report provenance or counts disagree with training")


def _article_only_diagnostics(rows: list[dict[str, Any]], reference: dict[str, dict[str, Any]],
                              wording: str) -> dict[str, dict[str, int]]:
    result = {}
    for method, read_name in (("lm", "memory"), ("zero", "zero"), ("donor", "donor"),
                              ("base_full_text", None)):
        selected = [row for row in rows if row["wording"] == wording]
        mismatches = 0
        for row in selected:
            prediction = (reference[row["key"]]["base_full_text"]["prediction"] if read_name is None
                          else row["reads"][read_name]["prediction"])
            if (not row["base_full_text_correct"] if read_name is None
                    else not row["reads"][read_name]["correct"]):
                if normalized_answer(prediction) == "the " + normalized_answer(row["answer"]):
                    mismatches += 1
        result[method] = {"n": len(selected), "article_only_mismatches": mismatches}
    return result


def _validate_cross_cell_rows(cell_rows: list[list[dict[str, Any]]]) -> None:
    reference_keys = set(_metadata(cell_rows[0]))
    reference_meta = _metadata(cell_rows[0])
    for rows in cell_rows[1:]:
        current = _metadata(rows)
        if set(current) != reference_keys:
            raise ValueError("evaluation cells do not have matching case keys")
        if current != reference_meta:
            raise ValueError("evaluation cells disagree about case metadata")
    for rows in cell_rows:
        # Prefix endpoint-8 values are intentionally duplicated over branches.
        before_by_prefix: dict[tuple[str, str, int], tuple[bool, ...]] = {}
        for row in rows:
            if row["after_write"] != 8:
                continue
            key = (row["prefix_id"], row["wording"], row["entity"])
            values = tuple(_row_method(row, method, endpoint=8) for method in _METHODS)
            previous = before_by_prefix.get(key)
            if previous is not None and previous != values:
                raise ValueError("duplicated prefix results disagree across branches")
            before_by_prefix[key] = values
        by_episode: dict[tuple[str, int], dict[int, dict[str, Any]]] = defaultdict(dict)
        for row in rows:
            by_episode[row["episode_id"], row["entity"]][row["after_write"]] = row
        for (episode_id, entity), endpoints in by_episode.items():
            before = endpoints.get(8)
            if before is None:
                raise ValueError(f"missing before endpoint for {episode_id}/{entity}")
            for endpoint, row in endpoints.items():
                if endpoint == 8 or row["scope"] != "unspoken":
                    continue
                if (row["truth_bit"] != before["truth_bit"]
                        or ("answer" in row and "answer" in before and row["answer"] != before["answer"])):
                    raise ValueError("unspoken before and after truths disagree")


def _validate_case_coverage(rows: list[dict[str, Any]]) -> None:
    """Require every declared branch to contain each required endpoint and entity."""
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[row["episode_id"]].append(row)
    for episode_id, episode_rows in by_episode.items():
        conditions = {(row["condition"], row["target"], row["wording"], row["prefix_id"])
                      for row in episode_rows}
        if len(conditions) != 1:
            raise ValueError(f"episode metadata is inconsistent: {episode_id}")
        condition, _target, _wording, _prefix = next(iter(conditions))
        expected_endpoints = {8} if condition == "no_write" else {8, 9, 16}
        actual_endpoints = {row["after_write"] for row in episode_rows}
        if actual_endpoints != expected_endpoints:
            raise ValueError(f"episode endpoint coverage is incomplete: {episode_id}")
        for endpoint in expected_endpoints:
            entities = {row["entity"] for row in episode_rows if row["after_write"] == endpoint}
            if entities != set(range(4)):
                raise ValueError(f"episode entity coverage is incomplete: {episode_id}/write{endpoint}")


def _summary(rows: list[dict[str, Any]], *, condition: str, wording: str, scope: str,
             endpoint: int, method: str) -> dict[str, Any]:
    selected = [row for row in rows if row["wording"] == wording and row["condition"] == condition
                and row["scope"] == scope and row["after_write"] == endpoint]
    before_by_id = {(row["episode_id"], row["entity"]): row for row in rows if row["after_write"] == 8}
    pairs = []
    for after in selected:
        before = before_by_id.get((after["episode_id"], after["entity"]))
        if before is None:
            raise ValueError("after endpoint has no paired prefix endpoint")
        pairs.append((_row_method(before, method, endpoint=8), _row_method(after, method, endpoint=endpoint)))
    before_correct = sum(left for left, _ in pairs)
    after_correct = sum(right for _, right in pairs)
    retained = sum(left and right for left, right in pairs)
    remained_wrong = sum(not left and not right for left, right in pairs)
    damage = sum(left and not right for left, right in pairs)
    repair = sum(not left and right for left, right in pairs)
    total = len(pairs)
    rates = {
        "before": _fraction(before_correct, total),
        "after": _fraction(after_correct, total),
        "retained_correct": _fraction(retained, before_correct),
        "remained_wrong": _fraction(remained_wrong, total - before_correct),
        "damage": _fraction(damage, before_correct),
        "repair": _fraction(repair, total - before_correct),
    }
    donor_agrees = [row["donor_truth_agrees"] for row in selected if "donor_truth_agrees" in row]
    diagnostics = {}
    for field in ("state_norm", "last_update_norm", "last_residual_norm"):
        values = [float(row[field]) for row in selected if field in row and row[field] is not None]
        if values:
            diagnostics[field] = {"n": len(values), "mean": fmean(values),
                                  "minimum": min(values), "maximum": max(values)}
    return {
        "n": total, "before_correct": before_correct, "after_correct": after_correct,
        "truth_changes": condition == "correction" and scope == "target",
        "retained_correct": retained, "remained_wrong": remained_wrong,
        "damage": damage, "repair": repair,
        "rates": rates, "change_pp": None if total == 0 else 100 * (after_correct - before_correct) / total,
        "donor_truth_agrees": {
            "n": sum(donor_agrees), "total": len(donor_agrees),
            "rate": _fraction(sum(donor_agrees), len(donor_agrees)),
        } if donor_agrees else None,
        "diagnostics": diagnostics,
    }


def _prefix_changes(rows: list[dict[str, Any]], *, wording: str, condition: str, scope: str,
                    method: str, endpoint: int, default_seed: int) -> dict[tuple[str, int], float]:
    before = {(row["episode_id"], row["entity"]): _row_method(row, method, endpoint=8)
              for row in rows if row["after_write"] == 8}
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        if (row["wording"], row["condition"], row["scope"], row["after_write"]) != (wording, condition, scope, endpoint):
            continue
        old = before.get((row["episode_id"], row["entity"]))
        if old is None:
            raise ValueError("after endpoint has no paired prefix endpoint")
        seed = row.get("writer_seed", default_seed)
        if type(seed) is not int:
            raise ValueError("evaluation rows must declare writer_seed")
        grouped[(row["prefix_id"], seed)].append(float(_row_method(row, method, endpoint=endpoint) - old))
    return {key: fmean(values) for key, values in grouped.items()}


def _percentile(values: list[float], probability: float) -> float:
    values = sorted(values)
    index = probability * (len(values) - 1)
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def _bootstrap_difference(left: dict[tuple[str, int], float], right: dict[tuple[str, int], float],
                          *, resamples: int, seed: int) -> dict[str, Any]:
    keys = sorted(set(left) & set(right))
    prefixes = sorted({prefix for prefix, _ in keys})
    seeds = sorted({optimization_seed for _, optimization_seed in keys})
    expected_keys = {(prefix, optimization_seed) for prefix in prefixes for optimization_seed in seeds}
    if not prefixes or not seeds or set(left) != set(right) or set(keys) != expected_keys:
        raise ValueError("paired bootstrap groups do not have complete prefix and seed coverage")
    per_seed = {
        str(optimization_seed): 100 * fmean(
            left[prefix, optimization_seed] - right[prefix, optimization_seed] for prefix in prefixes
        ) for optimization_seed in seeds
    }
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        sampled = [rng.choice(prefixes) for _ in prefixes]
        draws.append(100 * fmean(
            fmean(left[prefix, optimization_seed] - right[prefix, optimization_seed] for prefix in sampled)
            for optimization_seed in seeds
        ))
    return {
        "per_seed_difference_pp": per_seed,
        "mean_seed_difference_pp": fmean(per_seed.values()),
        "interval_pp": [_percentile(draws, 0.005), _percentile(draws, 0.995)],
        "prefixes": len(prefixes), "seeds": seeds,
        "resamples": resamples, "bootstrap_seed": seed,
        "confidence": 0.99, "unit": "paired_prefixes_across_observed_seeds",
        "interpretation": "conditional_on_observed_seeds",
    }


def _comparison(rows_by_cell: dict[tuple[str, int, str], list[dict[str, Any]]], *, width: int,
                wording: str, condition: str, scope: str, method: str,
                left_writer: str, right_writer: str, endpoint: int, settings: dict[str, Any]) -> dict[str, Any]:
    left_cells = [(key, rows) for key, rows in rows_by_cell.items() if key[0] == left_writer and key[1] == width]
    right_cells = [(key, rows) for key, rows in rows_by_cell.items() if key[0] == right_writer and key[1] == width]
    if not left_cells or not right_cells:
        return {"status": "provisional", "reason": "comparison arm is absent"}
    left = {}
    right = {}
    accuracy_context = {}
    for key, rows in left_cells:
        left.update(_prefix_changes(rows, wording=wording, condition=condition, scope=scope,
                                    method=method, endpoint=endpoint, default_seed=int(key[2])))
    for key, rows in right_cells:
        right.update(_prefix_changes(rows, wording=wording, condition=condition, scope=scope,
                                     method=method, endpoint=endpoint, default_seed=int(key[2])))
    for key, rows in (*left_cells, *right_cells):
        counts = _summary(rows, condition=condition, wording=wording, scope=scope,
                          endpoint=endpoint, method=method)
        accuracy_context.setdefault(key[2], {})[key[0]] = {
            name: counts[name] for name in ("n", "before_correct", "after_correct")}
    return {"status": "estimated", "width": width, "wording": wording, "condition": condition,
            "scope": scope, "endpoint": endpoint, "method": method, "left": left_writer,
            "right": right_writer, "accuracy_context_by_seed": accuracy_context,
            **_bootstrap_difference(left, right,
                resamples=settings["bootstrap_samples"], seed=settings["bootstrap_seed"])}


def _secondary_comparison(rows_by_cell: dict[tuple[str, int, str], list[dict[str, Any]]], *, width: int,
                          wording: str, scope: str, method: str, writer: str, endpoint: int,
                          settings: dict[str, Any]) -> dict[str, Any]:
    arms = {}
    for condition in ("repeat", "correction"):
        parts = {}
        for key, rows in rows_by_cell.items():
            if key[0] == writer and key[1] == width:
                parts.update(_prefix_changes(rows, wording=wording, condition=condition, scope=scope,
                                             method=method, endpoint=endpoint, default_seed=int(key[2])))
        arms[condition] = parts
    if not arms["repeat"] or not arms["correction"]:
        return {"status": "provisional", "reason": "secondary branch is absent"}
    return {"status": "estimated", "width": width, "wording": wording,
            "scope": scope, "endpoint": endpoint, "method": method, "writer": writer,
            "left": "repeat", "right": "correction",
            **_bootstrap_difference(arms["repeat"], arms["correction"],
                resamples=settings["bootstrap_samples"], seed=settings["bootstrap_seed"])}


def _no_write_identity(rows: list[dict[str, Any]], wording: str) -> dict[str, Any]:
    selected = [row for row in rows if row["wording"] == wording and row["condition"] == "no_write"
                and row["after_write"] == 8]
    result = {"n": len(selected), "observed_endpoint": 8, "reported_endpoint": 16,
              "before_equals_after": True, "no_extra_write": True}
    for method in _METHODS:
        correct = sum(_row_method(row, method, endpoint=8) for row in selected)
        result[method] = {"before_correct": correct, "after_correct": correct,
                          "change_pp": 0 if selected else None}
    return result


def aggregate_study(study: Path, protocol: dict) -> dict:
    """Validate every sealed evaluation cell and write a fresh aggregate report."""
    study = Path(study)
    report_path = study / "report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    require_training_seal(study, protocol)
    cells = protocol.get("cells")
    settings = protocol.get("settings")
    if not isinstance(cells, list) or not cells or not isinstance(settings, dict):
        raise ValueError("protocol does not declare cells and settings")
    if "dataset_sha256" in protocol and file_hash(study / "dataset.json") != protocol["dataset_sha256"]:
        raise ValueError("study data changed")
    training_seal_sha256 = file_hash(study / "training_sealed.json")
    rows_by_cell: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    cell_summaries = []
    expected_metadata = _expected_case_metadata(study, settings)
    donor_truths = _dataset_no_write_truths(study)
    for index, cell in enumerate(cells):
        if cell.get("index") != index:
            raise ValueError("cell indices must be complete and ordered")
        directory = study / "evaluation" / str(cell["index"])
        identity = {**cell_identity(study, protocol, index, "evaluation"),
                    "training_seal_sha256": training_seal_sha256}
        verify_completion(directory, identity)
        rows = _load_rows(directory, writer_seed=cell["writer_seed"])
        training_report = json.loads((study / "training" / str(cell["index"]) / "report.json").read_text())
        _validate_evaluation_report(directory, rows, training_report)
        _validate_case_coverage(rows)
        _validate_expected_case_metadata(rows, expected_metadata)
        _validate_donor_rows(rows, donor_truths)
        key = (cell["writer"], cell["width"], str(cell["writer_seed"]))
        if key in rows_by_cell:
            raise ValueError("duplicate cell declaration")
        rows_by_cell[key] = rows
        summary = {"cell": cell, "rows": len(rows), "metrics": {}}
        for wording in _WORDINGS:
            for condition in ("repeat", "correction", "balanced"):
                summary["metrics"].setdefault(wording, {}).setdefault(condition, {})
                scopes = ("all",) if condition == "balanced" else ("target", "unspoken")
                for scope in scopes:
                    for endpoint in _ENDPOINTS:
                        for method in _METHODS:
                            summary["metrics"][wording][condition].setdefault(scope, {}).setdefault(str(endpoint), {})[method] = _summary(
                                rows, condition=condition, wording=wording, scope=scope, endpoint=endpoint, method=method
                            )
        cell_summaries.append(summary)
    _validate_cross_cell_rows(list(rows_by_cell.values()))
    reference_rows = _validate_reference(study, next(iter(rows_by_cell.values())))
    for rows in rows_by_cell.values():
        for row in rows:
            reference = reference_rows.get(row["key"])
            if reference is None or (reference["answer"], reference["base_full_text_correct"]) != (
                    row["answer"], row["base_full_text_correct"]):
                raise ValueError("reference answer or correctness disagrees with an evaluation cell")
    for summary in cell_summaries:
        cell = summary["cell"]
        rows = rows_by_cell[(cell["writer"], cell["width"], str(cell["writer_seed"]))]
        summary["article_only_diagnostics"] = {
            wording: _article_only_diagnostics(rows, reference_rows, wording) for wording in _WORDINGS
        }
    widths = sorted({cell["width"] for cell in cells})
    primary = []
    secondary = []
    smoke = settings.get("purpose") == "implementation_smoke"
    if not smoke:
        for width in widths:
            for wording in _WORDINGS:
                primary.append(_comparison(rows_by_cell, width=width, wording=wording, condition="repeat",
                                           scope="unspoken", method="lm", left_writer="delta", right_writer="gated",
                                           endpoint=16, settings=settings))
                for writer in ("delta", "gated"):
                    secondary.append(_secondary_comparison(rows_by_cell, width=width, wording=wording,
                                                           scope="unspoken", method="lm", writer=writer,
                                                           endpoint=16, settings=settings))
    else:
        primary = [{"status": "provisional", "reason": "implementation_smoke has no matched writer arm"}]
        secondary = []
    no_write_identity = []
    for summary in cell_summaries:
        cell = summary["cell"]
        cell_rows = rows_by_cell[(cell["writer"], cell["width"], str(cell["writer_seed"]))]
        for wording in _WORDINGS:
            no_write_identity.append({"cell": cell, "wording": wording,
                                      **_no_write_identity(cell_rows, wording)})
    report = {
        "schema": "delta_fact_aggregate_v1", "protocol_sha256": file_hash(study / "protocol.json"),
        "training_seal_sha256": training_seal_sha256, "cells": cell_summaries,
        "all_rows": sum(len(rows) for rows in rows_by_cell.values()), "primary": primary, "secondary": secondary,
        "no_write_identity": no_write_identity,
        "bootstrap": {"resamples": settings.get("bootstrap_samples", 10000),
                      "seed": settings.get("bootstrap_seed", 8237), "confidence": 0.99,
                      "interval": "99% pointwise percentile paired-prefix",
                      "optimization_seeds": "fixed observed seeds; not resampled",
                      "multiplicity": "pointwise; no family-wise correction",
                      "status": "provisional_implementation_smoke" if smoke else "estimated"},
        "interpretation": {"confidence_scope": "unconditional over all cases",
                           "change_limit": "A smaller recall drop alone is not better memory; inspect initial and final accuracy, including floor effects.",
                           "conditional_rates": "descriptive, conditional on initial correctness",
                           "probe": "fixed trained prefix decoder; no erasure claim",
                           "seed_scope": "conditional on the three declared seeds"},
    }
    with report_path.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return report
