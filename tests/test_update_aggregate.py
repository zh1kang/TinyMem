import math

import numpy as np
import pytest

from tinymem.evaluation.memory_updates import CountRate, UpdateMetrics
from tinymem.evaluation.update_aggregate import aggregate_updates


def metric(index, n, d):
    return UpdateMetrics(f"history-{index}", (f"source-{index}-a", f"source-{index}-b"),
                         {"forgetting": CountRate(n, d)}, {"cohort": {"CC": d - n, "CW": n, "WC": 0, "WW": 0}})


def test_pool_counts_not_history_ratios_and_reproduce_paired_bootstrap():
    runs = {"a1": [metric(0, 1, 1), metric(1, 0, 7)], "a2": [metric(0, 0, 1), metric(1, 2, 7)],
            "b": [metric(0, 0, 1), metric(1, 0, 7)]}
    result = aggregate_updates(runs, families={"a": ["a1", "a2"], "b": ["b"]},
                               contrasts=[("a", "b")], resamples=100, seed=7)
    assert result["runs"]["a1"]["rates"]["forgetting"]["value"] == 1 / 8  # not 1/2
    assert result["runs"]["a1"]["transitions"]["cohort"] == {"CC": 7, "CW": 1, "WC": 0, "WW": 0}
    family = result["families"]["a"]["rates"]["forgetting"]
    assert family["mean"] == 3 / 16
    assert family["seed_sd"] == pytest.approx(np.std([1 / 8, 2 / 8], ddof=1))
    # Independent scalar recomputation of every seeded resample's pooled ratios.
    draws = np.random.default_rng(7).multinomial(2, [0.5, 0.5], size=100)
    expected = []
    for first, second in draws:
        denominator = first + 7 * second
        expected.append((first / denominator + 2 * second / denominator) / 2)
    bounds = np.quantile(expected, [0.025, 0.975])
    comparison = result["contrasts"]["a-minus-b"]["forgetting"]
    assert comparison["difference"] == 3 / 16
    assert comparison["history_interval"]["bounds"] == pytest.approx(bounds)


def test_seed_mean_is_not_pooled_across_seeds_and_pairing_is_preserved():
    runs = {"a1": [metric(0, 1, 1), metric(1, 0, 0)], "a2": [metric(0, 0, 7), metric(1, 0, 7)]}
    result = aggregate_updates(runs, families={"a": ["a1", "a2"]}, contrasts=[], resamples=20)
    assert result["families"]["a"]["rates"]["forgetting"]["mean"] == 0.5  # not 1/15
    same = aggregate_updates({"a": runs["a2"], "b": list(reversed(runs["a2"]))},
                             families={"a": ["a"], "b": ["b"]}, contrasts=[("a", "b")], resamples=20)
    assert same["contrasts"]["a-minus-b"]["forgetting"]["history_interval"]["bounds"] == [0, 0]


def test_undefined_bootstrap_denominators_are_counted_not_discarded():
    runs = {"a": [metric(0, 1, 1), metric(1, 0, 0)]}
    result = aggregate_updates(runs, families={"a": ["a"]}, contrasts=[], resamples=1000)
    rate = result["runs"]["a"]["rates"]["forgetting"]
    assert rate["value"] == 1
    assert rate["history_interval"]["bounds"] is None
    assert rate["history_interval"]["undefined_resamples"] > 0
    empty = aggregate_updates({"a": [metric(0, 0, 0)]}, families={"a": ["a"]}, contrasts=[], resamples=10)
    assert empty["families"]["a"]["rates"]["forgetting"]["mean"] is None
    assert empty["runs"]["a"]["rates"]["forgetting"]["history_interval"]["undefined_resamples"] == 10


@pytest.mark.parametrize("runs,families,contrasts", [
    ({}, {}, []),
    ({"a": [metric(0, 1, 1), metric(0, 1, 1)]}, {"a": ["a"]}, []),
    ({"a": [metric(0, 1, 1)], "b": [metric(1, 1, 1)]}, {"a": ["a"], "b": ["b"]}, []),
    ({"a": [metric(0, 1, 1)]}, {"a": ["a"], "b": ["a"]}, []),
    ({"a": [metric(0, 1, 1)]}, {"a": ["a"]}, [("a", "a")]),
])
def test_bad_cluster_or_family_coverage_fails(runs, families, contrasts):
    with pytest.raises(ValueError):
        aggregate_updates(runs, families=families, contrasts=contrasts, resamples=10)


@pytest.mark.parametrize("kwargs", [{"resamples": True}, {"resamples": 1}, {"confidence": math.nan}, {"seed": -1}])
def test_bad_bootstrap_options_fail(kwargs):
    with pytest.raises(ValueError):
        aggregate_updates({"a": [metric(0, 1, 1)]}, families={"a": ["a"]}, contrasts=[], **kwargs)
