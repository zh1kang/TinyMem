from dataclasses import replace

import pytest

from scripts.aggregate_longmemeval_shards import aggregate_predictions
from tinymem.data.longmemeval import LongMemEvalExample
from tinymem.evaluation.longmemeval import (
    LongMemEvalPrediction,
    _aggregate,
    format_longmemeval_prompt,
)
from tinymem.evaluation.longmemeval_abstention import local_abstention_metrics


def record(question_id, abstained, prediction="a"):
    return LongMemEvalPrediction(
        question_id, "knowledge-update", prediction, "a",
        prediction == "a", float(prediction == "a"), abstained, 100, 1,
    )


def test_suffix_label_never_enters_prompt():
    example = LongMemEvalExample("q", "knowledge-update", "where?", "a", "today", (), ())
    changed = replace(example, question_id="q_abs", answer="private answer")
    assert not example.gold_unanswerable
    assert changed.gold_unanswerable
    assert not replace(example, question_id="q_abs_middle").gold_unanswerable
    assert format_longmemeval_prompt(example) == format_longmemeval_prompt(changed)
    assert record("q_abs", False).to_dict()["gold_unanswerable"] is True


def test_direct_and_shard_metrics_share_all_confusion_cells():
    predictions = [
        record("tp_abs", True, "unknown"), record("fn_abs", False),
        record("fp", True, ""), record("tn", False),
    ]
    direct = _aggregate(predictions).to_dict()
    shard = aggregate_predictions([row.to_dict() for row in predictions])
    assert direct == shard
    local = direct["local_abstention"]
    for key in ("true_positives", "false_positives", "false_negatives", "true_negatives"):
        assert local[key] == 1
    assert local["precision"] == local["recall"] == local["false_abstention_rate"] == 0.5
    assert local["empty_output_count"] == 1
    assert not local["official_judge_used"]
    assert direct["exact_accuracy"] == direct["coverage"] == 0.5
    assert direct["selective_accuracy"] == 1.0


def test_legacy_records_recover_labels_without_rescoring_answers():
    row = record("q_abs", False).to_dict()
    del row["gold_unanswerable"]
    result = local_abstention_metrics([row])
    assert result["false_negatives"] == 1
    assert result["recall"] == 0
    assert result["precision"] is None
    assert result["false_abstention_rate"] is None
    del row["question_id"]
    assert local_abstention_metrics([row]) is None


def test_no_gold_abstention_has_undefined_recall():
    result = local_abstention_metrics([record("q", True).to_dict()])
    assert result["recall"] is None
    assert result["precision"] == 0
    assert result["false_abstention_rate"] == 1


@pytest.mark.parametrize("label", [False, 1, None])
def test_inconsistent_explicit_labels_are_rejected(label):
    row = record("q_abs", True).to_dict()
    row["gold_unanswerable"] = label
    with pytest.raises(ValueError, match="disagrees"):
        local_abstention_metrics([row])
