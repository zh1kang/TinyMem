"""Gold-conditioned local diagnostics, not official semantic QA scores."""

from collections.abc import Mapping, Sequence


ABSTENTION_METADATA = {
    "gold_unanswerable_labels": True,
    "gold_unanswerable_rule": "question_id_suffix_abs",
    "abstention_detector": "empty_or_phrase_v1",
    "official_judge_used": False,
}


def local_abstention_metrics(
    predictions: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    """Recover suffix labels in legacy records and count local detector outcomes."""
    if not predictions:
        raise ValueError("predictions must be nonempty")
    labels = []
    missing_ids = False
    for row in predictions:
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            missing_ids = True
            continue
        gold = question_id.endswith("_abs")
        if "gold_unanswerable" in row and (
            not isinstance(row["gold_unanswerable"], bool)
            or row["gold_unanswerable"] != gold
        ):
            raise ValueError("gold_unanswerable disagrees with question_id")
        detected = row.get("abstained")
        if not isinstance(detected, bool):
            raise ValueError("abstained must be a boolean")
        labels.append((gold, detected))
    if missing_ids:
        return None

    tp = sum(gold and detected for gold, detected in labels)
    fp = sum(not gold and detected for gold, detected in labels)
    fn = sum(gold and not detected for gold, detected in labels)
    tn = sum(not gold and not detected for gold, detected in labels)
    empty_count = (
        sum(not row["prediction"].strip() for row in predictions)
        if all(isinstance(row.get("prediction"), str) for row in predictions)
        else None
    )
    return {
        **ABSTENTION_METADATA,
        "count": len(predictions),
        "gold_unanswerable_count": tp + fn,
        "gold_answerable_count": fp + tn,
        "detected_abstention_count": tp + fp,
        "empty_output_count": empty_count,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "false_abstention_rate": fp / (fp + tn) if fp + tn else None,
    }
