from collections import Counter
from dataclasses import asdict, replace
import json
import random

import pytest

from tinymem.data.opaque_qa1 import ROOMS
from tinymem.evaluation.memory_updates import CountRate, UpdatePrediction, score_update_episode
from test_memory_updates import episode


def perfect(row):
    return [UpdatePrediction(case.case_id, case.answer)
            for cases in (row.before, *(branch.queries for branch in row.branches)) for case in cases]


def set_predictions(records, changes):
    return [replace(record, prediction=changes.get(record.case_id, record.prediction)) for record in records]


def wrong(answer):
    return next(room for room in ROOMS if room != answer)


def hand_counted_example():
    row = episode()
    correction = row.branches[2]
    target = row.entities.index(correction.target)
    other = [i for i in range(8) if i != target]
    # On seven unchanged facts: CC=2, CW=2, WC=1, WW=2.
    changes = {row.before[i].case_id: wrong(row.before[i].answer) for i in other[4:]}
    changes.update({correction.queries[i].case_id: wrong(correction.queries[i].answer) for i in (*other[2:4], *other[5:])})
    changes[correction.queries[other[2]].case_id] = "unknown"
    changes[correction.queries[target].case_id] = row.before[target].answer  # stale, not the corrected value
    changes[correction.queries[8].case_id] = "office"  # false room answer for one absent key
    changes[correction.queries[9].case_id] = ""  # malformed/empty, not an invented room or correct abstention
    return row, set_predictions(perfect(row), changes)


def test_hand_counted_transition_table_and_denominators():
    row, records = hand_counted_example()
    result = score_update_episode(row, records)
    assert result.episode_id == row.episode_id
    assert result.source_group_ids == row.source_group_ids
    expected = {
        "before.accuracy": (7, 10), "before.known_accuracy": (5, 8), "before.absent_accuracy": (2, 2),
        "correction.accuracy": (3, 10), "correction.known_accuracy": (3, 8), "correction.absent_accuracy": (0, 2),
        "correction.known_false_abstention": (1, 8), "correction.absent_error": (2, 2),
        "correction.absent_false_answer": (1, 2), "correction.absent_invalid_output": (1, 2),
        "correction.invalid_output": (1, 10),
        "correction.target.before_accuracy": (1, 1), "correction.target.after_accuracy": (0, 1),
        "correction.target.update_accuracy": (0, 1), "correction.target.stale_answer": (1, 1),
        "addition.known_accuracy": (9, 9), "addition.absent_accuracy": (1, 1),
        "addition.target.update_accuracy": (1, 1), "addition.target.stale_answer": (0, 0),
        "repetition.target.update_accuracy": (0, 0), "repetition.target.stale_answer": (0, 0),
    }
    for cohort in ("unchanged_known", "untouched_known"):
        prefix = f"correction.{cohort}"
        assert result.transitions[prefix] == {"CC": 2, "CW": 2, "WC": 1, "WW": 2}
        expected.update({f"{prefix}.{metric}": counts for metric, counts in {
            "before_accuracy": (4, 7), "after_accuracy": (3, 7), "joint_preservation": (2, 7),
            "conditional_preservation": (2, 4), "conditional_forgetting": (2, 4),
            "unconditional_forgetting": (2, 7),
        }.items()})
    for key, counts in expected.items():
        assert result.rates[key] == CountRate(*counts), key
    # Copying the same before measurement into three contrasts must not triple its sample count.
    assert result.rates["before.accuracy"].denominator == 10
    assert sum(result.rates[f"{stage}.accuracy"].denominator for stage in ("before", "addition", "repetition", "correction")) == 40


def test_perfect_answers_keep_known_and_absent_counts_separate_at_every_state():
    row = episode()
    result = score_update_episode(row, perfect(row))
    for stage, known in (("before", 8), ("addition", 9), ("repetition", 8), ("correction", 8)):
        assert result.rates[f"{stage}.accuracy"] == CountRate(10, 10)
        assert result.rates[f"{stage}.known_accuracy"] == CountRate(known, known)
        assert result.rates[f"{stage}.absent_accuracy"] == CountRate(10 - known, 10 - known)
        assert result.rates[f"{stage}.absent_error"].numerator == 0
        assert result.rates[f"{stage}.known_false_abstention"].numerator == 0
    assert result.rates["correction.target.update_accuracy"] == CountRate(1, 1)
    assert result.rates["correction.target.stale_answer"] == CountRate(0, 1)
    for cohort, counts in result.transitions.items():
        assert counts["CC"] in (7, 8)
        assert counts["CW"] == counts["WC"] == counts["WW"] == 0
        assert result.rates[f"{cohort}.conditional_forgetting"] == CountRate(0, counts["CC"])


def test_never_knowing_a_fact_is_not_perfect_retention():
    row = episode()
    result = score_update_episode(row, [replace(record, prediction="unknown") for record in perfect(row)])
    assert result.rates["before.known_accuracy"] == CountRate(0, 8)
    assert result.rates["before.accuracy"] == CountRate(2, 10)
    assert result.rates["addition.accuracy"] == CountRate(1, 10)
    assert result.rates["addition.known_false_abstention"] == CountRate(9, 9)
    assert result.rates["addition.target.update_accuracy"] == CountRate(0, 1)
    assert result.rates["correction.target.stale_answer"] == CountRate(0, 1)
    for cohort, transitions in result.transitions.items():
        assert transitions["CC"] == transitions["CW"] == transitions["WC"] == 0
        assert transitions["WW"] in (7, 8)
        assert result.rates[f"{cohort}.conditional_preservation"].value is None
        assert result.rates[f"{cohort}.conditional_forgetting"].value is None
        assert result.rates[f"{cohort}.after_accuracy"].value == 0.0
        assert result.rates[f"{cohort}.unconditional_forgetting"].value == 0.0
    document = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert document["rates"]["correction.unchanged_known.conditional_forgetting"] == {
        "numerator": 0, "denominator": 0, "value": None,
    }


def test_repetition_target_is_unchanged_but_not_untouched():
    row = episode()
    target = row.entities.index(row.branches[1].target)
    changes = {row.before[i].case_id: wrong(row.before[i].answer) for i in range(8) if i != target}
    result = score_update_episode(row, set_predictions(perfect(row), changes))
    assert result.transitions["repetition.unchanged_known"] == {"CC": 1, "CW": 0, "WC": 7, "WW": 0}
    assert result.transitions["repetition.untouched_known"] == {"CC": 0, "CW": 0, "WC": 7, "WW": 0}
    assert result.rates["repetition.unchanged_known.conditional_preservation"] == CountRate(1, 1)
    assert result.rates["repetition.untouched_known.conditional_preservation"] == CountRate(0, 0)
    assert result.rates["repetition.untouched_known.after_accuracy"] == CountRate(7, 7)
    assert result.rates["repetition.target.after_accuracy"] == CountRate(1, 1)
    assert result.rates["repetition.target.update_accuracy"].value is None


def test_staleness_is_not_conditioned_on_before_correctness_and_unknown_is_not_stale():
    row = episode()
    correction, addition = row.branches[2], row.branches[0]
    target = row.entities.index(correction.target)
    changes = {row.before[target].case_id: "unknown", correction.queries[target].case_id: row.before[target].answer,
               addition.queries[8].case_id: "unknown"}
    result = score_update_episode(row, set_predictions(perfect(row), changes))
    assert result.rates["correction.target.before_accuracy"] == CountRate(0, 1)
    assert result.rates["correction.target.stale_answer"] == CountRate(1, 1)
    assert result.rates["addition.target.stale_answer"] == CountRate(0, 0)
    assert result.rates["addition.known_false_abstention"] == CountRate(1, 9)
    assert result.rates["addition.target.update_accuracy"] == CountRate(0, 1)


def test_history_counts_support_pooled_rates_not_mean_of_conditionals():
    results = []
    for index in range(2):
        row = episode(index)
        correction = row.branches[2]
        target = row.entities.index(correction.target)
        unchanged = [i for i in range(8) if i != target]
        changes = {}
        if index == 0:
            changes.update({row.before[i].case_id: "unknown" for i in unchanged[1:]})
            changes[correction.queries[unchanged[0]].case_id] = "unknown"
        results.append(score_update_episode(row, set_predictions(perfect(row), changes)))
    rates = [result.rates["correction.unchanged_known.conditional_forgetting"] for result in results]
    assert rates == [CountRate(1, 1), CountRate(0, 7)]
    pooled = CountRate(sum(rate.numerator for rate in rates), sum(rate.denominator for rate in rates))
    assert pooled.value == 1 / 8
    assert sum(rate.value for rate in rates) / 2 == 1 / 2  # The tempting but different estimand.


def test_exact_normalization_and_record_order_do_not_change_scores_or_inputs():
    row = episode()
    records = perfect(row)
    expected = score_update_episode(row, records).to_dict()
    noisy = [replace(record, prediction="\n" + record.prediction.upper() + ".! ") for record in records]
    random.Random(5).shuffle(noisy)
    before = (asdict(row), [asdict(record) for record in noisy])
    result = score_update_episode(row, noisy)
    assert result.to_dict() == expected
    assert (asdict(row), [asdict(record) for record in noisy]) == before
    document = result.to_dict()
    document["transitions"]["correction.unchanged_known"]["CC"] = 99
    assert result.transitions["correction.unchanged_known"]["CC"] == 7
    with pytest.raises(TypeError):
        result.rates["before.accuracy"] = CountRate(0, 10)
    with pytest.raises(TypeError):
        result.transitions["correction.unchanged_known"]["CC"] = 99
    external_rates = dict(result.rates)
    external_transitions = {key: dict(counts) for key, counts in result.transitions.items()}
    copied = replace(result, rates=external_rates, transitions=external_transitions)
    external_rates.clear()
    external_transitions["correction.unchanged_known"]["CC"] = 99
    assert copied.to_dict() == result.to_dict()


@pytest.mark.parametrize("output", ["", "!!!", "not unknown", "I do not know", "office or garden", "the office"])
def test_invalid_absent_output_is_an_error_not_a_fabricated_canonical_fact(output):
    row = episode()
    records = set_predictions(perfect(row), {row.before[-1].case_id: output})
    rates = score_update_episode(row, records).rates
    assert rates["before.absent_accuracy"] == CountRate(1, 2)
    assert rates["before.absent_error"] == CountRate(1, 2)
    assert rates["before.absent_false_answer"] == CountRate(0, 2)
    assert rates["before.absent_invalid_output"] == CountRate(1, 2)


@pytest.mark.parametrize("damage", ["missing", "extra", "duplicate_replacing", "other_world", "wrong_stage"])
def test_unpaired_partial_or_duplicate_predictions_fail(damage):
    row = episode()
    records = perfect(row)
    if damage == "missing":
        records.pop()
    elif damage == "extra":
        records.append(records[0])
    elif damage == "duplicate_replacing":
        records[-1] = records[0]
    elif damage == "other_world":
        records[0] = perfect(episode(1))[0]
    else:
        records[0] = replace(records[0], case_id=records[0].case_id.replace(":before:", ":correction:"))
    with pytest.raises(ValueError, match="exactly once"):
        score_update_episode(row, records)


def test_authoritative_gold_is_replayed_instead_of_trusting_score_records():
    row = episode()
    corrupted = replace(row, before=(replace(row.before[0], answer=wrong(row.before[0].answer)), *row.before[1:]))
    with pytest.raises(ValueError, match="replay"):
        score_update_episode(corrupted, perfect(corrupted))
    with pytest.raises(TypeError):
        UpdatePrediction("case", "office", reference="office")
    with pytest.raises(TypeError, match="sequence"):
        score_update_episode(row, {record.case_id: record.prediction for record in perfect(row)})
    with pytest.raises(TypeError, match="records"):
        score_update_episode(row, [asdict(record) for record in perfect(row)])


def test_random_outputs_obey_independent_transition_conservation():
    rng = random.Random(1337)
    row = episode()
    for _ in range(64):
        records = [replace(record, prediction=rng.choice((record.prediction, "unknown", *ROOMS, "")))
                   for record in perfect(row)]
        values = {record.case_id: record.prediction for record in records}
        result = score_update_episode(row, records)
        for branch in row.branches:
            for cohort in ("unchanged_known", "untouched_known"):
                pairs = [(before, after) for entity, before, after in zip(row.entities, row.before, branch.queries, strict=True)
                         if before.answer != "unknown" and before.answer == after.answer
                         and (cohort == "unchanged_known" or entity != branch.target)]
                counts = Counter(("C" if values[before.case_id] == before.answer else "W")
                                 + ("C" if values[after.case_id] == after.answer else "W") for before, after in pairs)
                prefix = f"{branch.kind}.{cohort}"
                assert result.transitions[prefix] == {key: counts[key] for key in ("CC", "CW", "WC", "WW")}
                before = result.rates[f"{prefix}.before_accuracy"]
                after = result.rates[f"{prefix}.after_accuracy"]
                preserved = result.rates[f"{prefix}.joint_preservation"]
                assert before.denominator == after.denominator == len(pairs)
                assert after.numerator - before.numerator == counts["WC"] - counts["CW"]
                assert preserved.numerator <= min(before.numerator, after.numerator)
                retention = result.rates[f"{prefix}.conditional_preservation"]
                forgetting = result.rates[f"{prefix}.conditional_forgetting"]
                if before.numerator:
                    assert retention.value + forgetting.value == pytest.approx(1)
                else:
                    assert retention.value is forgetting.value is None
        for stage in ("before", "addition", "repetition", "correction"):
            error = result.rates[f"{stage}.absent_error"]
            room = result.rates[f"{stage}.absent_false_answer"]
            invalid = result.rates[f"{stage}.absent_invalid_output"]
            assert error.denominator == room.denominator == invalid.denominator
            assert error.numerator == room.numerator + invalid.numerator


@pytest.mark.parametrize("value", [None, 1, True, float("nan"), ["office"]])
def test_nonstring_predictions_are_not_stringified(value):
    with pytest.raises(TypeError, match="string"):
        UpdatePrediction("case", value)


@pytest.mark.parametrize("case_id", ["", " case", 1, None])
def test_invalid_prediction_id(case_id):
    with pytest.raises(ValueError, match="case_id"):
        UpdatePrediction(case_id, "office")


@pytest.mark.parametrize("counts", [(True, 1), (1, True), (1.0, 2), (1, 2.0), (None, 2)])
def test_rate_counts_must_be_integers(counts):
    with pytest.raises(TypeError):
        CountRate(*counts)


@pytest.mark.parametrize("counts", [(-1, 2), (0, -1), (3, 2), (1, 0)])
def test_invalid_rate_counts(counts):
    with pytest.raises(ValueError):
        CountRate(*counts)
