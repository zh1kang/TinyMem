"""Count-pooled update outcomes with paired history uncertainty, not seed pooling."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np

from tinymem.evaluation.memory_updates import CountRate, UpdateMetrics


def _aligned(runs: Mapping[str, Sequence[UpdateMetrics]]) -> tuple[list[str], dict[str, list[UpdateMetrics]]]:
    if not runs or any(not rows for rows in runs.values()):
        raise ValueError("nonempty run histories are required")
    first = next(iter(runs.values()))
    ids = sorted(row.episode_id for row in first)
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate history identity")
    reference = {row.episode_id: row for row in first}
    if any(row.rates.keys() != first[0].rates.keys() or row.transitions.keys() != first[0].transitions.keys() for row in first):
        raise ValueError("metric coverage differs across histories")
    seen_sources = [group for row in first for group in row.source_group_ids]
    if len(seen_sources) != len(set(seen_sources)):
        raise ValueError("histories must be independent source clusters")
    aligned = {}
    for name, rows in runs.items():
        by_id = {row.episode_id: row for row in rows}
        if len(rows) != len(ids) or len(by_id) != len(rows) or sorted(by_id) != ids:
            raise ValueError("all runs must cover the identical histories exactly once")
        aligned[name] = [by_id[key] for key in ids]
        for key in ids:
            row, expected = by_id[key], reference[key]
            if (row.source_group_ids != expected.source_group_ids or row.rates.keys() != expected.rates.keys()
                    or row.transitions.keys() != expected.transitions.keys()):
                raise ValueError("source or metric coverage differs across runs")
    return ids, aligned


def _interval(samples: np.ndarray, confidence: float) -> dict:
    finite = np.isfinite(samples)
    undefined = int((~finite).sum())
    # Do not silently condition a bootstrap on a nonempty initially-correct set.
    bounds = None if undefined else np.quantile(samples, [(1 - confidence) / 2, (1 + confidence) / 2]).tolist()
    return {"confidence": confidence, "bounds": bounds, "undefined_resamples": undefined,
            "resamples": len(samples), "status": "undefined_resamples" if undefined else "estimated"}


def aggregate_updates(runs: Mapping[str, Sequence[UpdateMetrics]], *, families: Mapping[str, Sequence[str]],
                      contrasts: Sequence[tuple[str, str]], resamples: int = 10000,
                      seed: int = 20260905, confidence: float = 0.95) -> dict:
    """Bootstrap complete histories jointly across methods/seeds and all outcomes.

    Each family's estimand is the unweighted mean of its seed-specific pooled
    ratios, not a ratio pooled across seeds. Baselines are singleton families.
    Intervals condition on these observed optimization seeds; seed SD/range are
    separate descriptive quantities, not extra independent history samples.
    """
    if (type(resamples) is not int or resamples < 2 or type(seed) is not int or seed < 0
            or isinstance(confidence, bool) or not math.isfinite(confidence) or not 0 < confidence < 1):
        raise ValueError("valid bootstrap count, seed, and confidence required")
    ids, aligned = _aligned(runs)
    members = [name for values in families.values() for name in values]
    if not families or any(not values for values in families.values()) or len(set(members)) != len(members) or set(members) != set(runs):
        raise ValueError("families must partition the declared runs exactly once")
    if len(set(contrasts)) != len(contrasts) or any(a not in families or b not in families or a == b for a, b in contrasts):
        raise ValueError("contrasts require distinct declared families")
    rng = np.random.default_rng(seed)
    # One count vector per whole-history resample, shared across all seeds and
    # metrics; no query, event, or optimization-seed resampling.
    weights = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=resamples)
    run_output, samples = {}, {}
    keys = list(next(iter(aligned.values()))[0].rates)
    for name, rows in aligned.items():
        rates, samples[name] = {}, {}
        for key in keys:
            n = np.array([row.rates[key].numerator for row in rows], dtype=np.int64)
            d = np.array([row.rates[key].denominator for row in rows], dtype=np.int64)
            pooled = CountRate(int(n.sum()), int(d.sum()))
            numerator, denominator = weights @ n, weights @ d
            values = np.divide(numerator, denominator, out=np.full(resamples, np.nan), where=denominator > 0)
            samples[name][key] = values
            rates[key] = {**pooled.to_dict(), "history_interval": _interval(values, confidence)}
        transitions = {key: {cell: sum(row.transitions[key][cell] for row in rows) for cell in ("CC", "CW", "WC", "WW")}
                       for key in rows[0].transitions}
        run_output[name] = {"histories": len(rows), "rates": rates, "transitions": transitions}
    family_output, family_samples = {}, {}
    for family, names in families.items():
        rates, family_samples[family] = {}, {}
        for key in keys:
            values = [run_output[name]["rates"][key]["value"] for name in names]
            complete = all(value is not None for value in values)
            estimates = np.array(values, dtype=float)
            draw = np.stack([samples[name][key] for name in names]).mean(axis=0)
            family_samples[family][key] = draw
            rates[key] = {"mean": float(estimates.mean()) if complete else None,
                          "seed_values": dict(zip(names, values, strict=True)),
                          "seed_sd": float(estimates.std(ddof=1)) if complete and len(names) > 1 else None,
                          "seed_range": [float(estimates.min()), float(estimates.max())] if complete else None,
                          "history_interval": _interval(draw, confidence)}
        family_output[family] = {"runs": list(names), "rates": rates}
    comparisons = {}
    for left, right in contrasts:
        result = {}
        for key in keys:
            a, b = (family_output[name]["rates"][key]["mean"] for name in (left, right))
            result[key] = {"difference": None if a is None or b is None else a - b,
                           "history_interval": _interval(family_samples[left][key] - family_samples[right][key], confidence)}
        comparisons[f"{left}-minus-{right}"] = result
    return {"schema": "memory_update_aggregate_v1", "histories": len(ids), "history_ids": ids,
            "bootstrap": {"unit": "whole_history_paired_across_runs", "resamples": resamples, "seed": seed,
                          "confidence": confidence, "interval": "percentile_descriptive_not_multiplicity_adjusted",
                          "optimization_seeds": "fixed_observed_seeds_not_resampled", "undefined_policy": "no_interval_if_any_undefined_resample"},
            "runs": run_output, "families": family_output, "contrasts": comparisons}
