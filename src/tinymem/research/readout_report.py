"""Paired whole-history uncertainty, conditional on the observed seeds."""
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from tinymem.data.opaque_qa1 import ROOMS
from tinymem.evaluation.longmemeval import normalized_answer
from tinymem.research.readout_experiment import verify_run


def load_run_groups(directory):
    """Verify a sealed run and its prediction coverage before aggregation."""
    directory = Path(directory)
    verify_run(directory)
    protocol = json.loads((directory / "protocol.json").read_text())
    encodings = json.loads((directory / "encodings.json").read_text())
    if set(encodings) != {"train", "development"}:
        raise ValueError("unexpected encoded splits")
    expected = {}
    for split, histories in encodings.items():
        for history in histories:
            queries = history["queries"]
            if (len(queries) != 10
                    or sum(q["category"] == "update_known" for q in queries) != 8
                    or sum(q["category"] == "update_missing" for q in queries) != 2):
                raise ValueError("expected eight known and two missing queries")
            for query in queries:
                key = (split, history["history_id"], query["case_id"])
                if key in expected:
                    raise ValueError("duplicate encoded case")
                expected[key] = query
    variants = {(phase, condition) for phase in ("initial", "final")
                for condition in ("normal", "zero", "no_memory", "shuffled")}
    variants.add(("reference", "full_text"))
    seen = set()
    groups = defaultdict(list)
    for line in (directory / "predictions.jsonl").read_text().splitlines():
        row = json.loads(line)
        identity = (row["split"], row["history_id"], row["case_id"])
        variant = (row["phase"], row["condition"])
        key = (*identity, *variant)
        if identity not in expected or variant not in variants or key in seen:
            raise ValueError("unexpected or duplicate prediction")
        seen.add(key)
        query = expected[identity]
        if row["category"] != query["category"] or row["answer"] != query["answer"]:
            raise ValueError("prediction differs from encoded source case")
        if row["answer_tokens"] != len(query["answer_ids"]):
            raise ValueError("incorrect supervised token count")
        if row["condition"] == "full_text":
            if row["persistent_bytes"] is not None or row["temporary_vector_bytes"] != 0:
                raise ValueError("full-text reference has learned-state storage")
        elif row["persistent_bytes"] != 66:
            raise ValueError("incorrect persistent state storage")
        groups[(row["split"], *variant)].append(row)
    if len(seen) != len(expected) * len(variants):
        raise ValueError("incomplete prediction coverage")
    if protocol["arm"] not in {"affine", "gelu"}:
        raise ValueError("unexpected readout arm")
    for rows in groups.values():
        condition_metrics(rows)
    return dict(groups)


def accuracy_metrics(records):
    """Recompute accuracy counts without requiring model likelihoods."""
    records = list(records)
    if not records:
        raise ValueError("predictions must not be empty")
    seen = set()
    histories = defaultdict(list)
    known_histories = defaultdict(list)
    known, missing = [], []
    invalid = false_abstention = 0
    valid_answers = {normalized_answer(room) for room in ROOMS} | {"unknown"}
    for row in records:
        key = row["history_id"], row["case_id"]
        if key in seen:
            raise ValueError("duplicate prediction")
        seen.add(key)
        if not all(isinstance(value, str) and value.strip() for value in key):
            raise ValueError("history and case IDs must be nonempty strings")
        category = row["category"]
        if category not in ("update_known", "update_missing"):
            raise ValueError("invalid category")
        if not isinstance(row["answer"], str) or not isinstance(row["prediction"], str):
            raise ValueError("answer and prediction must be strings")
        answer = normalized_answer(row["answer"])
        prediction = normalized_answer(row["prediction"])
        if (category == "update_missing" and answer != "unknown") or (
            category == "update_known" and answer not in valid_answers - {"unknown"}
        ):
            raise ValueError("answer does not match category")
        correct = prediction == answer
        histories[key[0]].append(correct)
        if category == "update_known":
            known_histories[key[0]].append(correct)
        (known if category == "update_known" else missing).append(correct)
        false_abstention += category == "update_known" and prediction == "unknown"
        invalid += prediction not in valid_answers
    return {
        "queries": len(records),
        "histories": len(histories),
        "known_queries": len(known),
        "known_correct": sum(known),
        "absent_queries": len(missing),
        "absent_correct_count": sum(missing),
        "known_false_abstention_count": false_abstention,
        "invalid_output_count": invalid,
        "known_histories": len(known_histories),
        "all_known_correct_histories": sum(all(values) for values in known_histories.values()),
        "all_known_correct_history_rate": (
            sum(all(values) for values in known_histories.values()) / len(known_histories)
            if known_histories else None
        ),
        "all_queries_correct_histories": sum(all(values) for values in histories.values()),
        "known_accuracy": sum(known) / len(known) if known else None,
        "absent_correct": sum(missing) / len(missing) if missing else None,
        "known_false_abstention": false_abstention / len(known) if known else None,
        "invalid_output_rate": invalid / len(records),
        "all_queries_correct_history_rate": sum(all(values) for values in histories.values()) / len(histories),
    }


def condition_metrics(records):
    """Aggregate model accuracy and CE with explicit denominators."""
    records = list(records)
    metrics = accuracy_metrics(records)
    tokens = 0
    weighted_ce = first_ce = stopping_ce = 0.0
    for row in records:
        count = row["answer_tokens"]
        if type(count) is not int or count < 1:
            raise ValueError("answer_tokens must be a positive integer")
        losses = [row[name] for name in ("answer_ce", "first_answer_ce", "stopping_ce")]
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
               for value in losses):
            raise ValueError("CE values must be finite and nonnegative")
        tokens += count
        weighted_ce += losses[0] * count
        first_ce += losses[1]
        stopping_ce += losses[2]
    return {
        **metrics,
        "answer_tokens": tokens,
        "answer_ce_token_weighted": weighted_ce / tokens,
        "first_answer_ce_query_mean": first_ce / len(records),
        "stopping_ce_query_mean": stopping_ce / len(records),
    }


def _paired_comparison(records, *, field, condition, reference_name, resamples=2000, bootstrap_seed=0):
    """Compare GELU minus affine without treating queries as independent units.

    One history resampling vector is shared across arms and seeds. Intervals
    describe history uncertainty for these seeds, not uncertainty over new seeds.
    """
    if type(resamples) is not int or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    indexed = {}
    for row in records:
        arm, seed = row[field], row["seed"]
        history, case = row["history_id"], row["case_id"]
        if arm not in (reference_name, condition) or type(seed) is not int or seed < 0:
            raise ValueError("invalid arm or seed")
        if not all(isinstance(x, str) and x.strip() for x in (history, case)):
            raise ValueError("history and case IDs must be nonempty strings")
        if row["category"] not in ("known", "missing") or type(row["correct"]) is not bool:
            raise ValueError("invalid category or correctness")
        key = arm, seed, history, case
        if key in indexed:
            raise ValueError("duplicate prediction")
        indexed[key] = row["category"], row["correct"]
    if not indexed:
        raise ValueError("predictions must not be empty")
    seeds = sorted({key[1] for key in indexed})
    coverage = defaultdict(dict)
    for (arm, seed, history, case), (category, _) in indexed.items():
        coverage[arm, seed][history, case] = category
    reference = coverage[reference_name, seeds[0]]
    if not reference or any(coverage[arm, seed] != reference
                            for arm in (reference_name, condition) for seed in seeds):
        raise ValueError("arms and seeds must have identical query/category coverage")
    histories = sorted({history for history, _ in reference})
    for history in histories:
        if {category for (h, _), category in reference.items() if h == history} != {"known", "missing"}:
            raise ValueError("each history must contain both categories")
    rng = np.random.default_rng(bootstrap_seed)
    weights = rng.multinomial(len(histories), np.full(len(histories), 1 / len(histories)), size=resamples)
    result = {}
    for category in ("known", "missing"):
        cases = [[case for (h, case), cat in sorted(reference.items())
                  if h == history and cat == category] for history in histories]
        counts = np.asarray([len(items) for items in cases], dtype=float)
        differences, samples = [], []
        for seed in seeds:
            totals = {}
            for arm in (reference_name, condition):
                totals[arm] = np.asarray([
                    sum(indexed[arm, seed, history, case][1] for case in items)
                    for history, items in zip(histories, cases)
                ], dtype=float)
            delta = totals[condition] - totals[reference_name]
            differences.append(float(delta.sum() / counts.sum()))
            samples.append((weights @ delta) / (weights @ counts))
        interval = np.quantile(np.mean(samples, axis=0), [.025, .975]).tolist()
        result[category] = {
            "mean_difference": math.fsum(differences) / len(seeds),
            "seed_differences": dict(zip(map(str, seeds), differences)),
            "seed_sd": float(np.std(differences, ddof=1)) if len(seeds) > 1 else None,
            "seed_range": [min(differences), max(differences)],
            "interval": interval,
            "confidence": .95,
            "histories": len(histories),
            "resamples": resamples,
            "unit": "whole_history_paired_across_arms_and_seeds",
            "interpretation": "conditional_on_observed_seeds",
        }
    return result


def paired_history_comparison(records, *, resamples=2000, bootstrap_seed=0):
    """Compare GELU minus affine using paired whole-history resampling."""
    return _paired_comparison(
        records, field="arm", condition="gelu", reference_name="affine",
        resamples=resamples, bootstrap_seed=bootstrap_seed,
    )


def paired_control_comparison(records, *, reference, condition="normal",
                              resamples=2000, bootstrap_seed=0):
    """Compare named conditions without relabeling controls as model arms.

    Records must describe one arm; histories and queries are paired across seeds.
    """
    allowed = {"normal", "zero", "no_memory", "shuffled"}
    if condition not in allowed or reference not in allowed or condition == reference:
        raise ValueError("distinct valid control conditions are required")
    records = list(records)
    if len({row.get("arm") for row in records}) > 1:
        raise ValueError("control comparisons must describe one arm")
    metrics = _paired_comparison(
        records, field="condition", condition=condition, reference_name=reference,
        resamples=resamples, bootstrap_seed=bootstrap_seed,
    )
    for metric in metrics.values():
        metric["unit"] = "whole_history_paired_across_conditions_and_seeds"
    return {"contrast": {"condition": condition, "reference": reference},
            "metrics": metrics}
