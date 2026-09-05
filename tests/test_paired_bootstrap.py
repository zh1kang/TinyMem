import random
from statistics import fmean

import pytest
import torch

from tinymem.evaluation.paired_bootstrap import paired_history_interval


def scores(values):
    return {f"history-{index}": {1337: value, 2027: value, 4099: value} for index, value in enumerate(values)}


def test_identical_methods_have_exactly_zero_paired_interval():
    values = scores([0, 0.25, 0.5, 1])
    result = paired_history_interval(values, values, resamples=100)
    assert result.difference == result.lower == result.upper == 0
    assert result.histories == 4 and result.optimization_seeds == (1337, 2027, 4099)


def test_constant_advantage_and_order_do_not_change_the_interval():
    left, right = scores([1, 1, 1]), scores([0, 0, 0])
    result = paired_history_interval(left, right, resamples=100)
    assert result.difference == result.lower == result.upper == 1
    reversed_left = {history: dict(reversed(list(values.items()))) for history, values in reversed(list(left.items()))}
    assert paired_history_interval(reversed_left, right, resamples=100) == result
    negative = paired_history_interval(right, left, resamples=100)
    assert negative.difference == negative.lower == negative.upper == -1


def test_history_resampling_matches_independent_seed_averaged_reference():
    left = {"a": {1: 1, 2: 0}, "b": {1: 0.25, 2: 0.5}, "c": {1: 0, 2: 0}}
    right = {"a": {1: 0.5, 2: 0}, "b": {1: 0, 2: 0.5}, "c": {1: 0, 2: 1}}
    result = paired_history_interval(left, right, confidence=0.8, resamples=101, bootstrap_seed=17)
    differences = [0.25, 0.125, -0.5]
    rng = random.Random(17)
    samples = torch.tensor([fmean(rng.choices(differences, k=3)) for _ in range(101)], dtype=torch.float64)
    assert result.difference == fmean(differences)
    assert result.lower == pytest.approx(float(samples.quantile(0.1)))
    assert result.upper == pytest.approx(float(samples.quantile(0.9)))
    assert result.lower < 0 < result.upper
    reversed_left = {history: dict(reversed(list(values.items()))) for history, values in reversed(list(left.items()))}
    assert paired_history_interval(reversed_left, right, confidence=0.8, resamples=101, bootstrap_seed=17) == result


def test_seed_replicates_are_not_treated_as_independent_histories():
    single_left = {key: {1337: values[1337]} for key, values in scores([0, 0.25, 1]).items()}
    single_right = {key: {1337: 0} for key in single_left}
    single = paired_history_interval(single_left, single_right, resamples=100)
    repeated = paired_history_interval(scores([0, 0.25, 1]), scores([0, 0, 0]), resamples=100)
    assert (single.lower, single.upper) == (repeated.lower, repeated.upper)
    assert single.histories == repeated.histories == 3


@pytest.mark.parametrize("option,value", [
    ("confidence", 0), ("confidence", 1), ("confidence", float("nan")), ("confidence", True),
    ("resamples", 1), ("resamples", 2.5), ("resamples", True),
    ("bootstrap_seed", -1), ("bootstrap_seed", True),
])
def test_invalid_bootstrap_options_fail(option, value):
    with pytest.raises(ValueError, match=option):
        paired_history_interval(scores([0, 1]), scores([0, 1]), **{option: value})


def test_misaligned_or_empty_groups_and_seeds_fail():
    left = scores([0, 1])
    for right in ({}, {"history-0": left["history-0"]}):
        with pytest.raises(ValueError, match="history groups"):
            paired_history_interval(left, right)
    with pytest.raises(ValueError, match="history groups"):
        paired_history_interval(scores([1]), scores([1]))
    for malformed in ({"a": {}, "b": {}}, {"a": {True: 1}, "b": {True: 1}}, {"a": {1: 1}, "b": {2: 1}}):
        with pytest.raises(ValueError, match="optimization seeds"):
            paired_history_interval(malformed, malformed)
    with pytest.raises(ValueError, match="history IDs"):
        paired_history_interval({"": {1: 0}, "b": {1: 1}}, {"": {1: 0}, "b": {1: 1}})


@pytest.mark.parametrize("invalid", [-0.1, 1.1, float("nan"), float("inf"), "1"])
def test_invalid_accuracy_fractions_fail(invalid):
    left, right = scores([0, 1]), scores([0, 1])
    left["history-0"][1337] = invalid
    with pytest.raises(ValueError, match="accuracy fractions"):
        paired_history_interval(left, right)


@pytest.mark.parametrize("side,history", [("left", "b"), ("right", "a"), ("right", "b")])
@pytest.mark.parametrize("invalid", [True, 1.0])
def test_seed_key_types_are_checked_in_every_history_and_method(side, history, invalid):
    left = {"a": {1: 0}, "b": {1: 1}}
    right = {"a": {1: 0}, "b": {1: 1}}
    (left if side == "left" else right)[history] = {invalid: 0}
    with pytest.raises(ValueError, match="optimization seeds"):
        paired_history_interval(left, right)
