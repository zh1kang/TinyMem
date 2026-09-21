"""Paired writer-seed estimates and descriptive recall transitions."""

from collections import defaultdict
from statistics import fmean

import numpy as np

from tinymem.studies.distilled.scoring import READ_MODES


def paired_seed_estimate(control: dict[int, float], normalized: dict[int, float], *,
                         samples: int = 10000, seed: int = 2026091502) -> dict:
    """Resample whole paired writers, never questions or reader measurements."""

    if not control or control.keys() != normalized.keys():
        raise ValueError("paired estimate requires the same nonempty writer seeds")
    if type(samples) is not int or samples <= 0 or type(seed) is not int:
        raise ValueError("bootstrap samples and seed must be integers with positive samples")
    seeds = sorted(control)
    values = np.array([[control[s], normalized[s]] for s in seeds], dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("paired accuracies must be finite fractions from zero to one")
    gaps = values[:, 1] - values[:, 0]
    draws = np.random.default_rng(seed).choice(gaps, size=(samples, len(seeds)), replace=True).mean(axis=1)
    leave_one_out = [float(np.delete(gaps, i).mean()) for i in range(len(seeds))] if len(seeds) > 1 else []
    return {"control": float(values[:, 0].mean()), "normalized": float(values[:, 1].mean()),
            "difference": float(gaps.mean()), "interval": np.quantile(draws, [.025, .975]).tolist(),
            "confidence": .95, "interval_scope": "pointwise paired writer seeds; fixed corpus, panel, readers",
            "per_seed": {str(s): {"control": control[s], "normalized": normalized[s],
                                  "difference": normalized[s] - control[s]} for s in seeds},
            "positive_seeds": int((gaps > 0).sum()), "negative_seeds": int((gaps < 0).sum()),
            "tied_seeds": int((gaps == 0).sum()), "leave_one_seed_out_differences": leave_one_out,
            "resamples": samples, "bootstrap_seed": seed}


def summarize_rows(rows: list[dict], *, seeds: list[int], readers: list[int], prefixes: list[str]) -> dict:
    """Use validated complete answer rows; check the crossed primary panels again."""

    grouped = defaultdict(list)
    before_after = defaultdict(dict)
    for row in rows:
        writer, reader = int(row["writer_seed"]), int(row["reader_seed"])
        if writer not in seeds or reader not in readers or row["prefix_id"] not in prefixes:
            raise ValueError("answer row contains an undeclared writer, reader, or prefix")
        if row["wording"] not in ("familiar", "heldout"):
            raise ValueError("answer row contains an undeclared wording")
        key = writer, reader, row["wording"], row["condition"], row["scope"], row["after_write"]
        grouped[key].append(row)
        if row["condition"] == "repeat" and row["scope"] == "unspoken" and row["after_write"] in (8, 16):
            pair = writer, reader, row["wording"], row["episode_id"], row["entity"]
            if row["after_write"] in before_after[pair]:
                raise ValueError("duplicate repeat transition endpoint")
            before_after[pair][row["after_write"]] = row
    endpoints = {}
    for name, condition, scope in (("correction", "correction", "target"), ("retention", "repeat", "unspoken")):
        by_seed = defaultdict(list)
        strata = {}
        for writer in seeds:
            for reader in readers:
                for wording in ("heldout", "familiar"):
                    selected = grouped[writer, reader, wording, condition, scope, 16]
                    expected_n = len(prefixes) * (4 if scope == "target" else 12)
                    if len(selected) != expected_n or len({row["key"] for row in selected}) != expected_n:
                        raise ValueError("primary stratum has missing or duplicate answers")
                    expected = {(prefix, target, entity) for prefix in prefixes for target in range(4)
                                for entity in range(4) if (entity == target) == (scope == "target")}
                    if {(r["prefix_id"], r["target"], r["entity"]) for r in selected} != expected:
                        raise ValueError("primary stratum fact coverage differs")
                    accuracy = {mode: fmean(float(row["reads"][mode]["correct"]) for row in selected)
                                for mode in READ_MODES}
                    correct = {mode: sum(row["reads"][mode]["correct"] for row in selected)
                               for mode in READ_MODES}
                    strata[f"{writer}/reader{reader}/{wording}"] = {
                        "n": expected_n, "correct": correct, **accuracy,
                        "direct_bit_accuracy": fmean(float(row["direct_bit_correct"]) for row in selected)}
                    by_seed[writer].append((correct["real"], expected_n))
        endpoints[name] = {"per_seed": {s: sum(correct for correct, _ in v) / sum(n for _, n in v)
                                        for s, v in by_seed.items()}, "strata": strata}
    transitions = defaultdict(list)
    for (writer, reader, wording, _, _), pair in before_after.items():
        if set(pair) != {8, 16}:
            raise ValueError("repeat transition omits a before or after answer")
        transitions[writer, reader, wording].append(pair)
    result = {}
    for writer in seeds:
        for reader in readers:
            for wording in ("heldout", "familiar"):
                pairs = transitions[writer, reader, wording]
                if len(pairs) != len(prefixes) * 12:
                    raise ValueError("repeat transition coverage differs")
                modes = {}
                for mode in READ_MODES:
                    flags = [(p[8]["reads"][mode]["correct"], p[16]["reads"][mode]["correct"]) for p in pairs]
                    modes[mode] = {"correct_before": sum(a for a, _ in flags),
                                   "correct_after": sum(b for _, b in flags),
                                   "correct_to_wrong": sum(a and not b for a, b in flags),
                                   "wrong_to_correct": sum(not a and b for a, b in flags)}
                result[f"{writer}/reader{reader}/{wording}"] = {"n": len(pairs), "modes": modes}
    return {"endpoints": endpoints, "repeat_transitions": result,
            "weighting": "equal readers and two wording groups within writer, then equal writers",
            "transition_denominator": "matched episode/entity rows; shared-prefix forks are not independent"}
