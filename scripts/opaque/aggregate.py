#!/usr/bin/env python3
"""Summarize one frozen six-run confirmation study without treating queries as independent."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from tinymem.evaluation.paired_bootstrap import paired_history_interval
from tinymem.research.study_runtime import (
    check_repository, repository_path, sha256, validate_execution,
)

from tinymem.evaluation.reader_gate import reader_exact_match


def digest(path):
    return sha256(path)


def load_predictions(run, study_hash):
    protocol = json.loads((run / "protocol.json").read_text())
    result = json.loads((run / "results.json").read_text())
    assert protocol["split"] == "confirmation"
    assert protocol["study_protocol_sha256"] == study_hash
    assert digest(run / "predictions.jsonl") == result["predictions_sha256"]
    assert digest(run / "states.json") == result["states_sha256"]
    rows = [json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines()]
    identities = [(row["condition"], row["case_id"]) for row in rows]
    assert len(identities) == len(set(identities))
    for row in rows:
        assert row["exact_match"] == reader_exact_match(row["prediction"], row["answer"], row["category"])
    return protocol, rows


def scores(rows, condition, category):
    grouped = defaultdict(list)
    for row in rows:
        if row["condition"] == condition and row["category"] == category:
            grouped[row["world_id"]].append(int(row["exact_match"]))
    expected_queries = 8 if category == "opaque_qa1_known" else 1
    assert len(grouped) == 128 and all(len(values) == expected_queries for values in grouped.values())
    return {world: statistics.fmean(values) for world, values in grouped.items()}


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--study-protocol", type=repository_path, required=True)
parser.add_argument("--baseline-run", type=repository_path, required=True)
parser.add_argument("--memory-runs", type=repository_path, nargs=6, required=True)
parser.add_argument("--output", type=repository_path, required=True)
args = parser.parse_args()
check_repository()
frozen = json.loads(args.study_protocol.read_text())
study_hash = digest(args.study_protocol)
assert all(digest(Path(path)) == expected for path, expected in frozen["source_sha256"].items())
seeds = frozen["seeds"]
assert len(seeds) == len(set(seeds)) == 3
assert len({run.resolve() for run in args.memory_runs}) == 6
baseline_protocol, baseline = load_predictions(args.baseline_run, study_hash)
execution = baseline_protocol.get("execution", {})
validate_execution(execution)
assert baseline_protocol["data_protocol_sha256"] == frozen["data_protocol_sha256"]
assert baseline_protocol["reader_gate_results_sha256"] == frozen["reader_gate_results_sha256"]
assert baseline_protocol["vocabulary_sha256"] == frozen["vocabulary_sha256"]
methods = {"baseline": {seed: baseline for seed in seeds}}
inputs = [args.baseline_run]
observed_training_runs = set()
for run in args.memory_runs:
    protocol, rows = load_predictions(run, study_hash)
    validate_execution(protocol.get("execution", {}), execution)
    assert protocol["data_protocol_sha256"] == frozen["data_protocol_sha256"]
    assert protocol["adapter_sha256"] == baseline_protocol["adapter_sha256"]
    assert protocol["split_sha256"] == baseline_protocol["split_sha256"]
    writer, seed = protocol["writer_kind"], protocol["seed"]
    assert writer in frozen["writers"] and seed in seeds
    assert seed not in methods.setdefault(writer, {})
    training_run = repository_path(protocol["training_run"])
    assert str(training_run) in frozen["runs"]
    assert training_run.resolve() not in observed_training_runs
    observed_training_runs.add(training_run.resolve())
    training_protocol = json.loads((training_run / "protocol.json").read_text())
    training_result = json.loads((training_run / "results.json").read_text())
    assert digest(training_run / "protocol.json") == protocol["training_protocol_sha256"]
    assert digest(training_run / "results.json") == protocol["training_results_sha256"]
    assert training_protocol["study_protocol_sha256"] == study_hash
    assert training_protocol["writer_kind"] == writer and training_protocol["seed"] == seed
    assert training_result["checkpoint_sha256"] == protocol["checkpoint_sha256"]
    methods[writer][seed] = rows
    inputs.append(run)
assert observed_training_runs == {Path(run).resolve() for run in frozen["runs"]}
assert all(set(by_seed) == set(seeds) for by_seed in methods.values())
cells, table = {}, []
references = {}
for method, by_seed in methods.items():
    conditions = frozen["baseline_conditions" if method == "baseline" else "memory_conditions"]
    assert len(conditions) == len(set(conditions))
    for rows in by_seed.values():
        assert {row["condition"] for row in rows} == set(conditions)
        assert {row["category"] for row in rows} == {"opaque_qa1_known", "opaque_qa1_missing"}
        assert len(rows) == 128 * 9 * len(conditions)
    for condition in conditions:
        for category in ("opaque_qa1_known", "opaque_qa1_missing"):
            values = {seed: scores(rows, condition, category) for seed, rows in by_seed.items()}
            worlds = set(values[seeds[0]])
            assert all(set(items) == worlds for items in values.values())
            cells[(method, condition, category)] = {
                world: {seed: values[seed][world] for seed in seeds} for world in sorted(worlds)
            }
            for seed, rows in by_seed.items():
                expected = {}
                for row in rows:
                    if row["condition"] != condition or row["category"] != category:
                        continue
                    expected[row["case_id"]] = (row["world_id"], row["context"], row["question"], row["answer"])
                key = (condition.split(":")[0], category)
                assert references.setdefault(key, expected) == expected
            per_seed = {seed: statistics.fmean(items.values()) for seed, items in values.items()}
            table.append({
                "method": method, "condition": condition, "category": category,
                "mean_accuracy": statistics.fmean(per_seed.values()),
                "optimization_seed_sd": None if method == "baseline" else statistics.stdev(per_seed.values()),
                "per_seed_accuracy": per_seed, "worlds": len(worlds),
                "queries_per_checkpoint": len(worlds) * (8 if category == "opaque_qa1_known" else 1),
                "deterministic_baseline": method == "baseline",
            })
comparisons = []
for comparison in frozen["statistical_comparisons"]:
    category = comparison["category"]
    left = cells[(*comparison["left"], category)]
    right = cells[(*comparison["right"], category)]
    interval = paired_history_interval(left, right, confidence=comparison["confidence"],
                                       resamples=frozen["bootstrap_resamples"], bootstrap_seed=frozen["bootstrap_seed"])
    per_seed = {seed: statistics.fmean(left[world][seed] - right[world][seed] for world in left) for seed in seeds}
    comparisons.append({**comparison, **asdict(interval), "per_seed_difference": per_seed,
                        "optimization_difference_sd": statistics.stdev(per_seed.values())})
args.output.mkdir(parents=True, exist_ok=False)
source_paths = [Path(__file__), Path("src/tinymem/evaluation/paired_bootstrap.py"),
                Path("src/tinymem/evaluation/reader_gate.py"), Path("src/tinymem/evaluation/longmemeval.py")]
provenance = {"study_protocol_sha256": study_hash, "execution": execution,
              "input_hashes": {str(run): {name: digest(run / name) for name in ("protocol.json", "results.json", "predictions.jsonl", "states.json")} for run in inputs},
              "source_sha256": {str(path): digest(path) for path in source_paths},
              "interval_scope": "paired_world_sampling_conditional_on_these_three_checkpoints_not_optimization_population",
              "short": "transfer_not_matched_short_training", "counterfactual": "global_room_permutation_not_isolated_binding_swap",
              "fingerprint": "handcrafted_approximate_lookup_bypasses_Qwen_not_raw_or_learned_compression"}
(args.output / "summary.json").write_text(json.dumps({"provenance": provenance, "accuracy": table, "comparisons": comparisons}, indent=2, sort_keys=True) + "\n")
(args.output / "paired_world_scores.json").write_text(json.dumps({"|".join(key): value for key, value in cells.items()}, indent=2, sort_keys=True) + "\n")
for name, rows in (("accuracy", table), ("comparisons", comparisons)):
    with (args.output / f"{name}.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
for path in source_paths:
    (args.output / path.name).write_bytes(path.read_bytes())
from scripts.report_opaque_study import build_report, plot_report

report = build_report(frozen, json.loads((args.output / "summary.json").read_text()))
report["provenance"] = {"study_protocol_sha256": study_hash, "summary_sha256": digest(args.output / "summary.json"),
                        "execution": execution, "source_sha256": provenance["source_sha256"]}
report_directory = args.output / "report"
report_directory.mkdir(exist_ok=False)
(report_directory / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
plot_report(report, report_directory, figure_label=f"portable {execution['runtime']['device']} evaluation")
print(json.dumps(report["assessment"], indent=2), flush=True)
