#!/usr/bin/env python3
"""Freeze the six-run protocol after reader qualification and two cost profiles."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import torch
from safetensors.torch import load_file


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--query-profile", type=Path, required=True)
parser.add_argument("--mean-profile", type=Path, required=True)
parser.add_argument("--vocabulary", type=Path, required=True)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
assert args.steps == 1000
profiles, profile_results, initial_projections = {}, {}, {}
for kind, path in (("query_pool", args.query_profile), ("mean_pool", args.mean_profile)):
    protocol = json.loads((path / "protocol.json").read_text())
    result = json.loads((path / "results.json").read_text())
    metrics = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
    assert result["profile"] is True and protocol["steps"] == 10 and protocol["seed"] == 1337
    assert protocol["writer_kind"] == kind and protocol["stored_bytes"] == 66
    assert protocol["aggregation_width"] == (64 if kind == "query_pool" else 82)
    assert protocol["shared_parameters"] == (185040 if kind == "query_pool" else 185066)
    assert protocol["memory_width"] == 8 and protocol["slots"] == 2
    assert protocol["segment_length_limit"] == 512 and protocol["max_actual_chunk"] == 132
    assert protocol["learning_rate"] == 0.001 and protocol["weight_decay"] == 0.01 and protocol["clip_norm"] == 1.0
    assert protocol["batch"] == "one_world_nine_serial_query_losses"
    assert protocol["loss"] == "mean_of_nine_per_query_answer_token_CEs_including_EOT"
    assert protocol["bptt"] == "full_four_chunks_no_detach" and protocol["reader_frozen"] is True
    assert protocol["reader_dtype"] == "bfloat16" and protocol["writer_dtype"] == "float32"
    assert protocol["training_cache"] is False and protocol["checkpoint_selection"] == "final_only"
    assert protocol["sampling"] == "seeded_shuffle_without_replacement_each_epoch"
    assert all(digest(Path(source)) == expected for source, expected in protocol["source_sha256"].items())
    assert digest(path / "input_artifact_hashes.json") == result["input_artifact_hashes_sha256"]
    input_hashes = json.loads((path / "input_artifact_hashes.json").read_text())
    for name, field in (("protocol.json", "protocol_sha256"), ("native_encodings.json", "native_encodings_sha256"),
                        ("schedule.json", "schedule_sha256")):
        assert digest(path / name) == input_hashes[field]
    schedule = json.loads((path / "schedule.json").read_text())
    assert schedule == [181, 1, 179, 217, 161, 25, 228, 36, 81, 234]
    encodings = json.loads((path / "native_encodings.json").read_text())
    assert [row["world_id"] for row in metrics] == [encodings["train"][index]["world_id"] for index in schedule]
    assert digest(path / "initial_writer.safetensors") == protocol["initial_writer_sha256"]
    initial = load_file(path / "initial_writer.safetensors")
    assert sum(tensor.numel() for tensor in initial.values()) == protocol["shared_parameters"]
    assert all(torch.isfinite(tensor).all() for tensor in initial.values())
    initial_projections[kind] = initial["read_projection.weight"]
    assert [row["step"] for row in metrics] == list(range(1, 11))
    assert result["forward_tokens"] == sum(row["forward_tokens"] for row in metrics)
    assert result["supervised_tokens"] == sum(row["supervised_tokens"] for row in metrics)
    assert result["forward_tokens"] == (13523 if kind == "query_pool" else 13513)
    assert result["supervised_tokens"] == 247
    assert all(math.isfinite(row["seconds"]) and row["seconds"] > 0 for row in metrics)
    assert all(math.isfinite(row["answer_ce"]) and math.isfinite(row["gradient_norm"]) for row in metrics)
    assert result["median_step_seconds_after_first"] == statistics.median(row["seconds"] for row in metrics[1:])
    assert math.isfinite(result["training_seconds"]) and result["training_seconds"] >= sum(row["seconds"] for row in metrics)
    assert digest(path / "metrics.jsonl") == result["metrics_sha256"]
    assert digest(path / "step_000010.safetensors") == result["checkpoint_sha256"]
    for row in metrics:
        assert len(row["state_gradient_norms"]) == 4
        assert all(math.isfinite(value) and value > 0 for value in row["state_gradient_norms"])
        assert row["valid_slot_counts"] == ([2, 2, 2, 2] if kind == "query_pool" else [1, 2, 2, 2])
    profiles[kind], profile_results[kind] = protocol, result
query, mean = profiles["query_pool"], profiles["mean_pool"]
generator = torch.Generator(device="cpu").manual_seed(1337)
expected_projection = torch.empty(2048, 8).uniform_(-1 / math.sqrt(8), 1 / math.sqrt(8), generator=generator)
assert all(torch.equal(projection, expected_projection) for projection in initial_projections.values())
assert digest(args.query_profile / "native_encodings.json") == digest(args.mean_profile / "native_encodings.json")
for key in ("data_protocol_sha256", "training_sha256", "development_sha256", "reader_gate_protocol_sha256",
            "reader_gate_results_sha256", "adapter_sha256", "snapshot"):
    assert query[key] == mean[key]
data = Path(query["arguments"]["data"])
reader_gate = Path(query["arguments"]["reader_gate"])
assert digest(data / "protocol.json") == query["data_protocol_sha256"]
assert digest(data / "train.json") == query["training_sha256"]
assert digest(data / "development.json") == query["development_sha256"]
assert digest(reader_gate / "protocol.json") == query["reader_gate_protocol_sha256"]
assert digest(reader_gate / "results.json") == query["reader_gate_results_sha256"]
assert json.loads((reader_gate / "results.json").read_text())["reader_accepted"] is True
expected_vocabulary = sorted({token for row in encodings["train"] for token in row["queries"][0]["history_ids"]})
assert len(expected_vocabulary) == 215
assert json.loads(args.vocabulary.read_text()) == expected_vocabulary
seeds, writers = [1337, 2027, 4099], ["query_pool", "mean_pool"]
run_order = [("query_pool", 1337), ("mean_pool", 1337), ("mean_pool", 2027),
             ("query_pool", 2027), ("query_pool", 4099), ("mean_pool", 4099)]
raw = ["recent_native", "recent_vocabulary", "latest_vocabulary", "latest_template"]
comparators = [["mean_pool", "opaque:normal"], *[["baseline", f"opaque:{method}"] for method in raw]]
comparisons = []
for category in ("opaque_qa1_known", "opaque_qa1_missing"):
    for right in comparators:
        comparisons.append({"family": "known_superiority" if category.endswith("known") else "absent_noninferiority",
                            "left": ["query_pool", "opaque:normal"], "right": right,
                            "category": category, "confidence": 0.99})
for writer in writers:
    for condition in ("drop", "zero"):
        comparisons.append({"family": "descriptive_memory_sensitivity", "left": [writer, "opaque:normal"],
                            "right": [writer, f"opaque:{condition}"], "category": "opaque_qa1_known", "confidence": 0.95})
baseline_methods = ["drop", *raw, "fingerprint", "full_history"]
sources = [Path(__file__), *(Path("artifacts/predictions") / name for name in (
    "train_opaque_memory.py", "evaluate_opaque_memory.py", "evaluate_opaque_baselines.py", "aggregate_opaque_study.py")),
    *(Path("src/tinymem") / name for name in (
        "research/native_training.py", "research/memory_prompt.py", "research/recurrent_memory.py", "research/prefix_reader.py",
        "research/pretrained.py", "memory/query_pool_slots.py", "memory/mean_pool_slots.py", "memory/recurrent_slots.py",
        "memory/packed_tokens.py", "memory/vocabulary_tokens.py", "memory/latest_fact_tokens.py", "memory/template_facts.py",
        "memory/fingerprint_facts.py", "memory/storage.py", "data/opaque_qa1.py", "data/symbolic_world.py",
        "evaluation/reader_gate.py", "evaluation/longmemeval.py", "evaluation/paired_bootstrap.py"))]
protocol = {
    "protocol": "opaque_qa1_fixed_byte_comparison_v1", "claim_scope": "one_derived_task_one_budget_not_a_storage_frontier",
    "data": str(data), "reader_gate": str(reader_gate), "vocabulary": str(args.vocabulary),
    "data_protocol_sha256": query["data_protocol_sha256"], "reader_gate_protocol_sha256": query["reader_gate_protocol_sha256"],
    "reader_gate_results_sha256": query["reader_gate_results_sha256"], "vocabulary_sha256": digest(args.vocabulary),
    "adapter_sha256": query["adapter_sha256"], "snapshot": query["snapshot"],
    "writers": writers, "seeds": seeds, "steps": args.steps,
    "runs": [str(args.output / f"{kind}_seed_{seed}") for kind, seed in run_order],
    "stored_bytes": 66, "slots": 2, "memory_width": 8, "state_dtype": "float32",
    "shared_parameters": {"query_pool": 185040, "mean_pool": 185066},
    "parameter_match": "26_parameter_difference_not_exact", "reader_frozen": True,
    "optimizer": {"name": "AdamW", "learning_rate": 0.001, "weight_decay": 0.01, "clip_norm": 1.0},
    "training": "one_world_per_update_mean_nine_serial_query_CEs_full_four_chunk_BPTT_no_detach",
    "sampling": "paired_seed_shuffle_without_replacement_each_epoch", "checkpoint": "final_only_no_development_selection",
    "compute_match": "same_updates_worlds_queries_not_exact_FLOPs", "local_model_concurrency": 1,
    "profile_weights_reused": False, "confirmation_before_all_six_completed": False,
    "baseline_conditions": [f"{variant}:{method}" for variant in ("opaque", "short") for method in baseline_methods],
    "memory_conditions": ["opaque:normal", "opaque:different_history", "opaque:drop", "opaque:zero", "counterfactual:normal", "short:normal"],
    "statistical_comparisons": comparisons,
    "statistics": {
        "unit": "world", "worlds": 128, "known_queries_per_world": 8, "absent_queries_per_world": 1,
        "bootstrap": "paired_percentile_100000_resamples_seed_20260905_seed_differences_averaged_within_world",
        "scope": "world_sampling_conditional_on_three_checkpoints_report_optimization_seed_SD_separately",
        "known_family": "five_99_percent_two_sided_intervals_Bonferroni_nominal_95_percent_family_coverage",
        "statistical_superiority": "all_five_known_difference_lower_bounds_strictly_above_zero",
        "statistical_inferiority": "at_least_one_known_upper_bound_strictly_below_zero_identify_winning_baseline",
        "practical_gain": "minimum_known_point_difference_at_least_0.05_separate_from_statistical_superiority",
        "absent_safe": "after_known_family_passes_all_five_absent_lower_bounds_above_minus_0.05_and_each_seed_at_least_122_of_128_correct",
        "absent_threshold_scope": "observed_operational_threshold_not_population_confidence_guarantee",
        "reader_qualification_on_confirmation": "full_opaque_known_and_absent_each_at_least_0.95_or_reader_limitation_must_be_reported",
        "reader_minimum_correct": {"known": 973, "absent": 122},
        "null_result": "nonpositive_or_overlapping_interval_is_not_equivalence_or_general_impossibility",
        "different_history": "descriptive_only_cyclic_donors_couple_recipient_worlds_no_iid_world_interval",
        "drop_zero": "descriptive_95_percent_intervals_outside_confirmatory_family_no_multiple_comparison_claim",
    },
    "bootstrap_resamples": 100000, "bootstrap_seed": 20260905,
    "diagnostics": ["known_accuracy_by_last_update_chunk_1_to_4", "known_accuracy_by_update_count",
                    "short_names_transfer_not_matched_training", "global_room_permutation_not_isolated_binding_swap"],
    "fingerprint_scope": "handcrafted_approximate_lookup_bypasses_Qwen_not_raw_or_learned_compression",
    "latency_scope": "generation_only_not_complete_read_cost",
    "profile_hashes": {kind: {name: digest(path / name) for name in ("protocol.json", "results.json", "metrics.jsonl")}
                       for kind, path in (("query_pool", args.query_profile), ("mean_pool", args.mean_profile))},
    "profile_median_step_seconds": {kind: result["median_step_seconds_after_first"] for kind, result in profile_results.items()},
    "projected_training_seconds_excluding_evaluation": 3 * args.steps * sum(result["median_step_seconds_after_first"] for result in profile_results.values()),
    "source_sha256": {str(path): digest(path) for path in sources},
    "failure_next_steps": "preserve_confirmation_as_consumed_then_diagnose_on_train_development_only_no_architecture_selection_on_confirmation",
}
args.output.mkdir(parents=True, exist_ok=False)
(args.output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
for path in sources:
    (args.output / path.name).write_bytes(path.read_bytes())
print(json.dumps({"protocol_sha256": digest(args.output / "protocol.json"),
                  "projected_training_hours": protocol["projected_training_seconds_excluding_evaluation"] / 3600,
                  "runs": protocol["runs"]}, indent=2), flush=True)
