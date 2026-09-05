from dataclasses import replace

import pytest

from tinymem.evaluation.association_study import ASSOCIATION_COMPARATORS, AssociationStudyDesign, assess_association_study
from tinymem.evaluation.paired_bootstrap import PairedAccuracyInterval


SEEDS = (1337, 2027, 4099)
DESIGN = AssociationStudyDesign(128, SEEDS, 100000, 20260905)


def interval(difference=0.1, lower=0.05, upper=0.15):
    return PairedAccuracyInterval(difference, lower, upper, 0.99, 128, SEEDS, 100000, 20260905)


def evidence():
    return ({name: interval() for name in ASSOCIATION_COMPARATORS},
            {name: interval(0, -0.01, 0.01) for name in ASSOCIATION_COMPARATORS})


def assess(known, absent, **kwargs):
    counts = dict(design=DESIGN, reader_known_correct=973, reader_absent_correct=122,
                  writer_absent_correct=dict.fromkeys(SEEDS, 122))
    return assess_association_study(known, absent, **(counts | kwargs))


def test_positive_requires_every_baseline_and_absent_safety():
    result = assess(*evidence())
    assert result.known_result == "superior" and result.overall_result == "positive"
    assert result.reader_qualified and result.absent_noninferiority and result.absent_observed_threshold
    assert result.practical_positive


@pytest.mark.parametrize("upper,expected", [(-0.01, "inferior"), (0, "inconclusive"), (0.03, "inconclusive")])
def test_one_nonwinning_contrast_prevents_strongest_baseline_claim(upper, expected):
    known, absent = evidence()
    known["latest_template"] = interval(-0.03, -0.05, upper)
    result = assess(known, absent)
    assert result.known_result == result.overall_result == expected
    assert result.absent_noninferiority is None and not result.practical_positive


def test_lower_bound_equal_to_zero_is_not_superiority():
    known, absent = evidence()
    known["mean_pool"] = interval(0.1, 0, 0.2)
    assert assess(known, absent).overall_result == "inconclusive"


@pytest.mark.parametrize("lower,passed", [(-0.05, False), (-0.05001, False), (-0.04999, True)])
def test_absent_noninferiority_boundary_is_strict(lower, passed):
    known, absent = evidence()
    absent["recent_native"] = interval(0, lower, 0.1)
    result = assess(known, absent)
    assert result.absent_noninferiority is passed
    assert result.overall_result == ("positive" if passed else "known_gain_abstention_unresolved")


def test_each_seed_must_meet_observed_absent_threshold():
    result = assess(*evidence(), writer_absent_correct={1337: 128, 2027: 128, 4099: 121})
    assert result.absent_noninferiority and not result.absent_observed_threshold
    assert result.overall_result == "known_gain_abstention_unresolved"


@pytest.mark.parametrize("field,value", [("reader_known_correct", 972), ("reader_absent_correct", 121)])
def test_reader_limitation_does_not_become_a_compression_negative(field, value):
    result = assess(*evidence(), **{field: value})
    assert result.known_result == "superior" and result.overall_result == "reader_limited"
    assert not result.practical_positive


@pytest.mark.parametrize("gain,practical", [(0.04999, False), (0.05, True)])
def test_practical_threshold_is_separate_from_statistical_superiority(gain, practical):
    known, absent = evidence()
    known["mean_pool"] = interval(gain, 0.001, 0.1)
    result = assess(known, absent)
    assert result.overall_result == "positive" and result.practical_positive is practical


@pytest.mark.parametrize("field,value", [("confidence", 0.95), ("histories", 127), ("histories", 128.0),
    ("optimization_seeds", (1337, 2027)), ("optimization_seeds", (1337.0, 2027, 4099)),
    ("optimization_seeds", None),
    ("difference", float("nan")), ("difference", True), ("lower", "0.1"), ("lower", -1.01), ("upper", -0.1),
    ("resamples", 0), ("resamples", 100000.0), ("bootstrap_seed", -1), ("bootstrap_seed", 20260905.0)])
def test_incompatible_or_invalid_intervals_fail(field, value):
    known, absent = evidence()
    known["latest_template"] = replace(known["latest_template"], **{field: value})
    with pytest.raises(ValueError):
        assess(known, absent)


@pytest.mark.parametrize("field,value", [("reader_known_correct", True), ("reader_known_correct", 1025),
    ("reader_absent_correct", -1), ("writer_absent_correct", {1337: 122, 2027: 122}),
    ("writer_absent_correct", {1337.0: 122, 2027: 122, 4099: 122}),
    ("writer_absent_correct", {1337: True, 2027: 122, 4099: 122})])
def test_invalid_counts_fail(field, value):
    with pytest.raises(ValueError):
        assess(*evidence(), **{field: value})


def test_missing_comparator_or_handcrafted_reference_cannot_enter_family():
    known, absent = evidence()
    known["fingerprint"] = known.pop("latest_template")
    with pytest.raises(ValueError):
        assess(known, absent)


@pytest.mark.parametrize("field,value", [("histories", 2), ("optimization_seeds", (1, 2, 3)),
                                        ("resamples", 10000), ("bootstrap_seed", 17)])
def test_internally_consistent_but_wrong_design_fails(field, value):
    known, absent = evidence()
    known = {name: replace(item, **{field: value}) for name, item in known.items()}
    absent = {name: replace(item, **{field: value}) for name, item in absent.items()}
    with pytest.raises(ValueError):
        assess(known, absent)


@pytest.mark.parametrize("field,value", [("worlds", True), ("worlds", 1), ("bootstrap_resamples", 1),
    ("bootstrap_seed", -1), ("seeds", (1, 1, 2)), ("seeds", (1, 2)), ("seeds", (True, 2, 3)),
    ("seeds", (1.0, 2, 3)), ("seeds", [1, 2, 3]), ("seeds", ([1], 2, 3))])
def test_invalid_expected_design_fails(field, value):
    with pytest.raises(ValueError):
        replace(DESIGN, **{field: value})
