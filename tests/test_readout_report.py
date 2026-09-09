"""Whole-history paired uncertainty for the readout comparison."""
import pytest

from tinymem.research.readout_report import paired_history_comparison


def rows(arm, seed, outcomes):
    return [dict(arm=arm, seed=seed, history_id=h, case_id=f"{h}:{q}",
                 category=category, correct=correct)
            for h, known, absent in outcomes
            for q, category, correct in [(0, "known", known), (1, "missing", absent)]]


def test_pairing_and_seed_summary():
    records = []
    for seed in (1, 2):
        records += rows("affine", seed, [("a", False, True), ("b", True, False)])
        records += rows("gelu", seed, [("a", True, True), ("b", True, False)])
    result = paired_history_comparison(records, resamples=200, bootstrap_seed=4)
    assert result["known"]["mean_difference"] == .5
    assert result["known"]["seed_differences"] == {"1": .5, "2": .5}
    assert result["known"]["seed_sd"] == 0
    assert result["missing"]["interval"] == [0., 0.]
    assert result == paired_history_comparison(list(reversed(records)), resamples=200, bootstrap_seed=4)


def test_unequal_query_counts_pool_counts_not_history_accuracies():
    records = []
    for arm in ("affine", "gelu"):
        for history, count in (("a", 1), ("b", 3)):
            for query in range(count):
                records.append(dict(arm=arm, seed=1, history_id=history,
                                    case_id=f"{history}:{query}", category="known",
                                    correct=arm == "gelu" and history == "a"))
            records.append(dict(arm=arm, seed=1, history_id=history,
                                case_id=f"{history}:missing", category="missing",
                                correct=False))
    result = paired_history_comparison(records, resamples=200, bootstrap_seed=4)
    assert result["known"]["mean_difference"] == .25
    assert result["known"]["seed_differences"] == {"1": .25}


def test_bootstrap_shares_history_draws_across_seeds():
    records = []
    # Opposing seed effects cancel for every paired history draw, not merely
    # for the original sample. Independent per-seed draws would add variance.
    records += rows("affine", 1, [("a", False, False), ("b", True, True)])
    records += rows("gelu", 1, [("a", True, True), ("b", False, False)])
    records += rows("affine", 2, [("a", True, True), ("b", False, False)])
    records += rows("gelu", 2, [("a", False, False), ("b", True, True)])
    result = paired_history_comparison(records, resamples=200, bootstrap_seed=4)
    for category in ("known", "missing"):
        assert result[category]["mean_difference"] == 0
        assert result[category]["interval"] == [0., 0.]


def test_identical_arms_have_zero_paired_interval():
    records = [r for arm in ("affine", "gelu") for seed in (1, 2)
               for r in rows(arm, seed, [("a", True, False), ("b", False, True)])]
    result = paired_history_comparison(records, resamples=100)
    assert result["known"]["interval"] == [0., 0.]


@pytest.mark.parametrize("damage", ["duplicate", "missing", "category"])
def test_rejects_unpaired_records(damage):
    records = rows("affine", 1, [("a", True, True), ("b", False, False)])
    records += rows("gelu", 1, [("a", True, True), ("b", False, False)])
    if damage == "duplicate":
        records.append(dict(records[0]))
    elif damage == "missing":
        records.pop()
    else:
        records[-1]["category"] = "known"
    with pytest.raises(ValueError):
        paired_history_comparison(records)


def test_condition_metrics_recompute_scores_and_weight_ce():
    from tinymem.research.readout_report import condition_metrics
    records = [
        dict(history_id="h", case_id="a", category="update_known", answer="office",
             prediction="OFFICE.", correct=False, answer_tokens=2, answer_ce=1.,
             first_answer_ce=2., stopping_ce=3.),
        dict(history_id="h", case_id="b", category="update_missing", answer="unknown",
             prediction="unknown", correct=False, answer_tokens=4, answer_ce=4.,
             first_answer_ce=4., stopping_ce=5.),
    ]
    result = condition_metrics(records)
    assert result["known_accuracy"] == 1.
    assert result["absent_correct"] == 1.
    assert result["answer_ce_token_weighted"] == 3.
    assert result["first_answer_ce_query_mean"] == 3.
    assert result["stopping_ce_query_mean"] == 4.
    assert result["known_false_abstention"] == 0.
    assert result["all_queries_correct_history_rate"] == 1.


def test_condition_metrics_reject_duplicate_cases():
    from tinymem.research.readout_report import condition_metrics
    row = dict(history_id="h", case_id="a", category="update_known", answer="office",
               prediction="office", answer_tokens=2, answer_ce=1.,
               first_answer_ce=1., stopping_ce=1.)
    with pytest.raises(ValueError, match="duplicate"):
        condition_metrics([row, row])
