"""Behavioral coverage for report denominators and history-level recall."""

from tinymem.research.readout_report import condition_metrics


def test_known_history_recall_is_separate_from_all_query_recall():
    rows = []
    for history in ("a", "b"):
        for index in range(10):
            known = index < 8
            answer = "kitchen" if known else "unknown"
            prediction = answer
            if (history == "a" and index == 8) or (history == "b" and index == 0):
                prediction = "garden"
            rows.append({
                "history_id": history,
                "case_id": f"{history}-{index}",
                "category": "update_known" if known else "update_missing",
                "answer": answer,
                "prediction": prediction,
                "answer_tokens": 2,
                "answer_ce": 1.0,
                "first_answer_ce": 1.0,
                "stopping_ce": 1.0,
            })

    result = condition_metrics(rows)

    assert result["known_queries"] == 16
    assert result["known_correct"] == 15
    assert result["absent_queries"] == 4
    assert result["absent_correct_count"] == 3
    assert result["all_known_correct_histories"] == 1
    assert result["all_known_correct_history_rate"] == 0.5
    assert result["all_queries_correct_histories"] == 0
    assert result["all_queries_correct_history_rate"] == 0.0
