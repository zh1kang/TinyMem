#!/usr/bin/env python3
"""Measure final training-set fit without selecting examples or checkpoints."""

import argparse
import importlib.metadata
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file

from tinymem.data.reader_gate import ReaderCase
from tinymem.evaluation.association_diagnostics import summarize_association_answers
from tinymem.evaluation.longmemeval import normalized_answer
from tinymem.research.study_runtime import (
    attach_execution, check_repository, prepare_device, repository_path, sha256,
)

from tinymem.evaluation.reader_gate import reader_exact_match
from tinymem.research.memory_prompt import encode_history_chunks, encode_memory_example
from tinymem.research.prefix_reader import generate_prefix_answer, prefix_answer_loss
from tinymem.research.pretrained import load_qwen_reader, verify_qwen_snapshot
from tinymem.research.recurrent_memory import NativeRecurrentMemory
from tinymem.utils.experiment import current_git_source_state
from tinymem.utils.seed import seed_everything


def digest(path):
    return sha256(path)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--training-run", type=repository_path, required=True)
parser.add_argument("--study-protocol", type=repository_path, required=True)
parser.add_argument("--output", type=repository_path, required=True)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
args = parser.parse_args()
check_repository()
study = json.loads(args.study_protocol.read_text())
study_hash = digest(args.study_protocol)
assert all(digest(Path(path)) == expected for path, expected in study["source_sha256"].items())
assert str(args.training_run) in study["runs"]
training = json.loads((args.training_run / "protocol.json").read_text())
result = json.loads((args.training_run / "results.json").read_text())
assert training["study_protocol_sha256"] == study_hash
assert training["seed"] == 1337 and training["steps"] == study["steps"] == 1000
assert result["profile"] is False
assert training["writer_kind"] in ("query_pool", "mean_pool")
checkpoint = args.training_run / "step_001000.safetensors"
assert digest(checkpoint) == result["checkpoint_sha256"]
assert digest(args.training_run / "metrics.jsonl") == result["metrics_sha256"]
metrics = [json.loads(line) for line in (args.training_run / "metrics.jsonl").read_text().splitlines()]
assert [row["step"] for row in metrics] == list(range(1, 1001))
data = repository_path(training["arguments"]["data"])
assert digest(data / "protocol.json") == study["data_protocol_sha256"] == training["data_protocol_sha256"]
assert digest(data / "train.json") == training["training_sha256"]
worlds = json.loads((data / "train.json").read_text())
assert len(worlds) == 256
encodings_path = args.training_run / "native_encodings.json"
inputs_path = args.training_run / "input_artifact_hashes.json"
assert digest(inputs_path) == result["input_artifact_hashes_sha256"]
inputs = json.loads(inputs_path.read_text())
assert digest(args.training_run / "protocol.json") == inputs["protocol_sha256"]
assert digest(encodings_path) == inputs["native_encodings_sha256"]
encodings = json.loads(encodings_path.read_text())["train"]
assert len(encodings) == len(worlds)
adapter = repository_path(training["adapter"])
assert training["adapter_sha256"] == study["adapter_sha256"]
assert all(digest(adapter / name) == expected for name, expected in training["adapter_sha256"].items())
assert training["reader_gate_results_sha256"] == study["reader_gate_results_sha256"]
assert training["reader_gate_protocol_sha256"] == study["reader_gate_protocol_sha256"]
ids = [row["opaque"]["world_id"] for row in worlds]
assert len(set(ids)) == len(ids)
assert ids == [row["world_id"] for row in encodings]
for row in worlds:
    cases = row["opaque"]["queries"]
    assert len(cases) == 9
    assert Counter(case["category"] for case in cases) == {"opaque_qa1_known": 8, "opaque_qa1_missing": 1}
    assert all(case["history_id"] == cases[0]["history_id"] and case["context"] == cases[0]["context"] for case in cases)
if args.dry_run:
    print(json.dumps({"claim": "preflight_only_no_inference", "writer_kind": training["writer_kind"],
                      "worlds": len(worlds), "queries": len(worlds) * 9,
                      "checkpoint_sha256": digest(checkpoint), "training_sha256": digest(data / "train.json")}))
    raise SystemExit(0)

device = prepare_device(args.device)
snapshot = verify_qwen_snapshot(Path("data/raw/pretrained/qwen3-1.7b"))
assert snapshot == training["snapshot"] == study["snapshot"]
args.output.mkdir(parents=True, exist_ok=False)
sources = [Path(__file__), *(Path("src/tinymem") / name for name in (
    "research/recurrent_memory.py", "research/prefix_reader.py", "research/memory_prompt.py", "research/pretrained.py",
    "memory/query_pool_slots.py", "memory/mean_pool_slots.py", "memory/recurrent_slots.py",
    "evaluation/association_diagnostics.py", "evaluation/reader_gate.py", "evaluation/longmemeval.py", "utils/seed.py"))]
seed_everything(training["seed"])
protocol = {
    "protocol": "opaque_qa1_final_training_fit_v1", "claim": "training_fit_not_generalization_or_confirmation",
    "source": current_git_source_state(Path.cwd()).to_dict(), "snapshot": snapshot,
    "study_protocol_sha256": study_hash, "training_run": str(args.training_run),
    "training_protocol_sha256": digest(args.training_run / "protocol.json"),
    "training_results_sha256": digest(args.training_run / "results.json"),
    "checkpoint_sha256": digest(checkpoint), "adapter_sha256": training["adapter_sha256"],
    "training_sha256": digest(data / "train.json"), "native_encodings_sha256": digest(encodings_path),
    "selection": "all_256_training_worlds_all_nine_queries_no_subset_or_checkpoint_selection",
    "seed": 1337, "writer_kind": training["writer_kind"], "conditions": ["opaque:normal"],
    "stored_state_bytes": 66, "source_sha256": {str(path): digest(path) for path in sources},
    "packages": {item.metadata["Name"]: item.version for item in importlib.metadata.distributions()},
    "numerics": {"device": device.type, "reader_dtype": "bfloat16", "writer_dtype": "float32",
                 "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                 "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                 "float32_matmul_precision": torch.get_float32_matmul_precision()},
    "generation": {"greedy": True, "max_new_tokens": 8, "batch_size": 1, "cache": False, "padding": False},
    "objective": "mean_query_token_mean_answer_ce_including_eot",
    "development_confirmation_or_external_evaluated": False,
    "native_encodings_scope": "saved_file_contains_consumed_development_encodings_but_only_training_records_are_used",
}
attach_execution(protocol, sources, device)
(args.output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
for path in sources:
    (args.output / path.name).write_bytes(path.read_bytes())
reader = load_qwen_reader(Path("data/raw/pretrained/qwen3-1.7b"), device=device, dtype=torch.bfloat16)
reader.model = PeftModel.from_pretrained(reader.model, adapter, is_trainable=False, local_files_only=True, use_safetensors=True)
reader.model.eval().requires_grad_(False)
writer = NativeRecurrentMemory(2048, memory_width=8, slots=2, segment_length=512,
                               writer_kind=training["writer_kind"], aggregation_width=training["aggregation_width"]).to(device)
writer.load_state_dict(load_file(checkpoint), strict=True)
writer.eval().requires_grad_(False)
predictions, states, patterns = [], [], []
started = time.perf_counter()
with torch.inference_mode(), (args.output / "predictions.jsonl").open("x") as handle:
    for row, saved in zip(worlds, encodings, strict=True):
        world = row["opaque"]
        cases = [ReaderCase(**case) for case in world["queries"]]
        examples = [encode_memory_example(reader, case) for case in cases]
        chunks = encode_history_chunks(reader, cases[0], world["chunks"])
        assert json.loads(json.dumps([asdict(example) for example in examples])) == saved["queries"]
        assert json.loads(json.dumps(chunks)) == saved["history_chunks"]
        state = writer.writer.empty(1)
        for chunk in chunks:
            state = writer.write(reader, state, torch.tensor(chunk, device=device))
            assert state.nbytes == 66
        memory = writer.memory_vectors(state)
        states.append({"world_id": world["world_id"], "values": state.values.tolist(), "valid": state.valid.tolist()})
        current_predictions = {}
        for case, example in zip(cases, examples, strict=True):
            before, after, answer = (torch.tensor(value, device=device) for value in
                                     (example.before_ids, example.after_ids, example.answer_ids))
            generated = generate_prefix_answer(reader, before, memory, after, max_new_tokens=8)
            ce = float(prefix_answer_loss(reader, before, memory, after, answer))
            prediction = {"condition": "opaque:normal", "world_id": world["world_id"], **asdict(case), **generated,
                          "answer_ce": ce, "answer_tokens_including_stop": len(example.answer_ids),
                          "exact_match": reader_exact_match(generated["prediction"], case.answer, case.category)}
            predictions.append(prediction)
            current_predictions[case.case_id] = generated["prediction"]
            handle.write(json.dumps(prediction, sort_keys=True, allow_nan=False) + "\n")
        known = [case for case in cases if case.category == "opaque_qa1_known"]
        pattern = summarize_association_answers(known, {case.case_id: current_predictions[case.case_id] for case in known})
        patterns.append({"world_id": world["world_id"], **asdict(pattern)})
        handle.flush()
        print(json.dumps({"worlds_complete": len(states), "predictions": len(predictions)}), flush=True)
(args.output / "states.json").write_text(json.dumps(states, indent=2, sort_keys=True, allow_nan=False) + "\n")
grouped = defaultdict(list)
for prediction in predictions:
    grouped[prediction["category"]].append(prediction)
summary = {
    "claim": protocol["claim"], "seconds": time.perf_counter() - started,
    "categories": {category: {"count": len(rows), "correct": sum(row["exact_match"] for row in rows),
                               "mean_answer_ce": statistics.mean(row["answer_ce"] for row in rows),
                               "prediction_counts": dict(Counter(normalized_answer(row["prediction"]) for row in rows))}
                   for category, rows in grouped.items()},
    "mean_query_answer_ce": statistics.mean(row["answer_ce"] for row in predictions),
    "ideal_uniform_seven_answer_sequences_ce": statistics.mean(math.log(7) / row["answer_tokens_including_stop"] for row in predictions),
    "uniform_reference_scope": "hypothetical_exact_answer_sequence_distribution_not_an_evaluated_model_or_trained_baseline",
    "known_world_answer_patterns": patterns,
    "protocol_sha256": digest(args.output / "protocol.json"),
    "predictions_sha256": digest(args.output / "predictions.jsonl"),
    "states_sha256": digest(args.output / "states.json"),
}
(args.output / "results.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
print(json.dumps({key: value for key, value in summary.items() if key != "known_world_answer_patterns"}, indent=2))
