#!/usr/bin/env python3
"""Validate and aggregate disjoint LongMemEval result shards."""

from __future__ import annotations

import argparse
import json
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--expected-examples", type=int, default=500)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/predictions/longmemeval_aggregate"),
    )
    return parser.parse_args()


def aggregate_predictions(predictions: list[dict[str, object]]) -> dict[str, object]:
    if not predictions:
        raise ValueError("predictions must be nonempty")
    covered = [prediction for prediction in predictions if not prediction["abstained"]]
    return {
        "count": len(predictions),
        "exact_accuracy": sum(
            bool(prediction["exact_match"]) for prediction in predictions
        )
        / len(predictions),
        "mean_token_f1": sum(
            float(prediction["token_f1"]) for prediction in predictions
        )
        / len(predictions),
        "coverage": len(covered) / len(predictions),
        "selective_accuracy": (
            sum(bool(prediction["exact_match"]) for prediction in covered)
            / len(covered)
            if covered
            else 0.0
        ),
    }


def main() -> None:
    args = parse_args()
    if args.expected_examples <= 0:
        raise ValueError("expected_examples must be positive")
    documents = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.results
    ]
    provenance_fields = (
        "dataset_variant",
        "dataset_revision",
        "dataset_sha256",
        "manifest_sha256",
        "checkpoint_sha256",
        "max_new_tokens",
        "chunk_tokens",
        "answers_used_in_prompts",
        "training_on_longmemeval",
    )
    first = documents[0]
    for document in documents:
        for field in provenance_fields:
            if document.get(field) != first.get(field):
                raise ValueError(f"shards disagree on {field}")
        results = document.get("results")
        if not isinstance(results, list) or len(results) != 1:
            raise ValueError("each shard must contain one evaluation condition")
        if results[0].get("condition") != "normal":
            raise ValueError("shard aggregation requires the normal condition")

    predictions = [
        prediction
        for document in documents
        for prediction in document["results"][0]["predictions"]
    ]
    question_ids = [prediction["question_id"] for prediction in predictions]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("shards contain duplicate question IDs")
    if len(predictions) != args.expected_examples:
        raise ValueError(
            f"expected {args.expected_examples} predictions, got {len(predictions)}"
        )

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for prediction in predictions:
        grouped[str(prediction["question_type"])].append(prediction)
    overall = aggregate_predictions(predictions)
    by_question_type = {
        name: aggregate_predictions(group)
        for name, group in sorted(grouped.items())
    }
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_directory = args.artifact_root / f"{timestamp}-{uuid.uuid4().hex[:8]}"
    run_directory.mkdir(parents=True, exist_ok=False)
    result_document = {
        "status": "frozen_full_external_test",
        **{field: first[field] for field in provenance_fields},
        "example_count": len(predictions),
        "shards": [str(path.resolve()) for path in args.results],
        "overall": overall,
        "by_question_type": by_question_type,
        "predictions": predictions,
    }
    (run_directory / "results.json").write_text(
        json.dumps(result_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    names = list(by_question_type)
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(
        names,
        [by_question_type[name]["mean_token_f1"] for name in names],
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("mean token F1")
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(run_directory / "longmemeval_by_type.png", dpi=160)
    plt.close(figure)
    print(json.dumps({"overall": overall, "by_question_type": by_question_type}, indent=2))
    print(f"artifacts: {run_directory}")


if __name__ == "__main__":
    main()
