#!/usr/bin/env python3
"""Describe final development answers without opening confirmation data."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from tinymem.data.reader_gate import ReaderCase
from tinymem.evaluation.association_diagnostics import label_association_history, summarize_association_answers
from tinymem.evaluation.longmemeval import normalized_answer
from tinymem.evaluation.reader_gate import reader_exact_match


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--training-run", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
protocol = json.loads((args.training_run / "protocol.json").read_text())
result = json.loads((args.training_run / "results.json").read_text())
assert result["profile"] is False and protocol["steps"] == 1000
assert digest(args.training_run / "step_001000.safetensors") == result["checkpoint_sha256"]
data_path = Path(protocol["arguments"]["data"]) / "development.json"
assert digest(data_path) == protocol["development_sha256"]
worlds = json.loads(data_path.read_text())
prediction_path = args.training_run / "development_predictions.jsonl"
rows = [json.loads(line) for line in prediction_path.read_text().splitlines()]
by_id = {row["case_id"]: row for row in rows}
assert len(by_id) == len(rows) == len(worlds) * 9
expected_ids = {case["case_id"] for row in worlds for case in row["opaque"]["queries"]}
assert set(by_id) == expected_ids
cohorts = {name: defaultdict(Counter) for name in ("last_mention_chunk", "movement_count", "location_changes")}
categories = defaultdict(Counter)
patterns = []
for row in worlds:
    world = row["opaque"]
    cases = [ReaderCase(**item) for item in world["queries"]]
    labels = label_association_history(world["chunks"], cases)
    known = [case for case in cases if case.category == "opaque_qa1_known"]
    pattern = summarize_association_answers(known, {case.case_id: by_id[case.case_id]["prediction"] for case in known})
    patterns.append({"world_id": world["world_id"], **asdict(pattern),
                     "prediction_counts": dict(Counter(normalized_answer(by_id[case.case_id]["prediction"]) for case in known))})
    for case in cases:
        prediction = by_id[case.case_id]
        assert prediction["world_id"] == world["world_id"]
        assert all(prediction[key] == value for key, value in asdict(case).items())
        correct = reader_exact_match(prediction["prediction"], case.answer, case.category)
        assert type(prediction["exact_match"]) is bool and prediction["exact_match"] == correct
        categories[case.category].update(count=1, correct=int(correct))
        if case.category == "opaque_qa1_known":
            for name, grouped in cohorts.items():
                grouped[str(getattr(labels[case.case_id], name))].update(count=1, correct=int(correct))
assert dict(categories) == result["development"]
for grouped in cohorts.values():
    assert sum(item["count"] for item in grouped.values()) == len(worlds) * 8
    assert sum(item["correct"] for item in grouped.values()) == categories["opaque_qa1_known"]["correct"]
sources = [Path(__file__), Path("src/tinymem/evaluation/association_diagnostics.py"),
           Path("src/tinymem/evaluation/reader_gate.py"), Path("src/tinymem/evaluation/longmemeval.py"),
           Path("src/tinymem/data/symbolic_world.py")]
report = {
    "claim": "descriptive_consumed_development_not_confirmation_or_checkpoint_selection",
    "training_run": str(args.training_run), "seed": protocol["seed"], "writer_kind": protocol["writer_kind"],
    "worlds": len(worlds), "categories": categories, "known_query_cohorts": cohorts,
    "known_world_answer_patterns": patterns,
    "all_same_prediction_worlds": sum(item["distinct_predictions"] == 1 for item in patterns),
    "distinct_prediction_count_histogram": dict(Counter(item["distinct_predictions"] for item in patterns)),
    "hindsight_constant_answer_ceiling_correct": sum(item["constant_answer_ceiling_correct"] for item in patterns),
    "interpretation": "reference-derived_constant_answer_ceiling_is_not_a_trained_competitor;_accuracy_below_it_does_not_establish_query_blindness",
    "artifact_sha256": {str(path): digest(path) for path in (
        args.training_run / "protocol.json", args.training_run / "results.json", data_path, prediction_path)},
    "source_sha256": {str(path): digest(path) for path in sources},
}
args.output.mkdir(parents=True, exist_ok=False)
(args.output / "results.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
for path in sources:
    (args.output / path.name).write_bytes(path.read_bytes())
print(json.dumps({key: value for key, value in report.items() if key not in ("known_world_answer_patterns", "artifact_sha256", "source_sha256")}, indent=2))
