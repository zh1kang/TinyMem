#!/usr/bin/env python3
"""Final-only memory interventions after a fixed multi-seed training protocol."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file

from tinymem.data.reader_gate import ReaderCase
from tinymem.research.study_runtime import (
    attach_execution, check_repository, prepare_device, repository_path, sha256, synchronize,
)

from tinymem.evaluation.reader_gate import reader_exact_match, summarize_reader_predictions
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
parser.add_argument("--split", choices=("development", "confirmation"), required=True)
parser.add_argument("--output", type=repository_path, required=True)
parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
args = parser.parse_args()
check_repository()
device = prepare_device(args.device)
frozen = json.loads(args.study_protocol.read_text())
study_hash = digest(args.study_protocol)
assert all(digest(Path(path)) == expected for path, expected in frozen["source_sha256"].items())
assert str(args.training_run) in frozen["runs"]
training = json.loads((args.training_run / "protocol.json").read_text())
training_result = json.loads((args.training_run / "results.json").read_text())
assert training["study_protocol_sha256"] == study_hash
assert training_result["profile"] is False
assert training["steps"] == frozen["steps"] and training["seed"] in frozen["seeds"]
assert training["writer_kind"] in frozen["writers"]
if args.split == "confirmation":
    assert len(frozen["runs"]) == 6
    assert len({Path(run).resolve() for run in frozen["runs"]}) == 6
    completed_pairs = set()
    for run in frozen["runs"]:
        run = Path(run)
        result = json.loads((run / "results.json").read_text())
        protocol = json.loads((run / "protocol.json").read_text())
        assert result["profile"] is False and protocol["study_protocol_sha256"] == study_hash
        assert protocol["steps"] == frozen["steps"]
        assert protocol["data_protocol_sha256"] == frozen["data_protocol_sha256"]
        assert protocol["reader_gate_results_sha256"] == frozen["reader_gate_results_sha256"]
        assert protocol["reader_gate_protocol_sha256"] == frozen["reader_gate_protocol_sha256"]
        assert digest(run / f"step_{frozen['steps']:06d}.safetensors") == result["checkpoint_sha256"]
        assert digest(run / "metrics.jsonl") == result["metrics_sha256"]
        metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        assert [row["step"] for row in metrics] == list(range(1, frozen["steps"] + 1))
        assert result["forward_tokens"] == sum(row["forward_tokens"] for row in metrics)
        assert result["supervised_tokens"] == sum(row["supervised_tokens"] for row in metrics)
        completed_pairs.add((protocol["writer_kind"], protocol["seed"]))
    assert len(completed_pairs) == 6
    assert completed_pairs == {(writer, seed) for writer in frozen["writers"] for seed in frozen["seeds"]}
data = repository_path(training["arguments"]["data"])
data_protocol = json.loads((data / "protocol.json").read_text())
assert digest(data / "protocol.json") == frozen["data_protocol_sha256"] == training["data_protocol_sha256"]
checkpoint = args.training_run / f"step_{training['steps']:06d}.safetensors"
assert digest(checkpoint) == training_result["checkpoint_sha256"]
adapter = repository_path(training["adapter"])
assert all(digest(adapter / name) == expected for name, expected in training["adapter_sha256"].items())
gate_run = repository_path(training["arguments"]["reader_gate"])
assert digest(gate_run / "results.json") == frozen["reader_gate_results_sha256"] == training["reader_gate_results_sha256"]
assert digest(gate_run / "protocol.json") == training["reader_gate_protocol_sha256"] == frozen["reader_gate_protocol_sha256"]
qualified = json.loads((gate_run / "results.json").read_text())
gate_protocol = json.loads((gate_run / "protocol.json").read_text())
assert qualified["reader_accepted"] is True
qualified_hashes = qualified["adapter_sha256"] if gate_protocol["protocol"] == "opaque_qa1_reader_continuation_v1" else gate_protocol["adapter_sha256"]
assert training["adapter_sha256"] == qualified_hashes
snapshot = verify_qwen_snapshot(Path("data/raw/pretrained/qwen3-1.7b"))
assert snapshot == training["snapshot"]
reader = load_qwen_reader(Path("data/raw/pretrained/qwen3-1.7b"), device=device, dtype=torch.bfloat16)
reader.model = PeftModel.from_pretrained(reader.model, adapter, is_trainable=False, local_files_only=True, use_safetensors=True)
reader.model.eval().requires_grad_(False)
seed_everything(training["seed"])
writer = NativeRecurrentMemory(2048, memory_width=8, slots=2, segment_length=512,
                               writer_kind=training["writer_kind"], aggregation_width=training["aggregation_width"]).to(device)
writer.load_state_dict(load_file(checkpoint))
writer.eval().requires_grad_(False)
assert digest(data / f"{args.split}.json") == data_protocol["data_sha256"][f"{args.split}.json"]
worlds = json.loads((data / f"{args.split}.json").read_text())
args.output.mkdir(parents=True, exist_ok=False)
sources = [Path(__file__), *(Path("src/tinymem") / name for name in (
    "research/recurrent_memory.py", "research/prefix_reader.py", "research/memory_prompt.py", "research/pretrained.py",
    "memory/query_pool_slots.py", "memory/mean_pool_slots.py", "memory/recurrent_slots.py",
    "data/opaque_qa1.py", "data/symbolic_world.py", "evaluation/reader_gate.py", "evaluation/longmemeval.py", "memory/storage.py"))]
protocol = {
    "protocol": "opaque_qa1_memory_interventions_v1", "source": current_git_source_state(Path.cwd()).to_dict(),
    "snapshot": snapshot, "study_protocol_sha256": study_hash, "training_run": str(args.training_run),
    "training_protocol_sha256": digest(args.training_run / "protocol.json"), "training_results_sha256": digest(args.training_run / "results.json"),
    "checkpoint_sha256": digest(checkpoint), "adapter_sha256": training["adapter_sha256"],
    "data_protocol_sha256": digest(data / "protocol.json"), "split": args.split,
    "split_sha256": digest(data / f"{args.split}.json"), "seed": training["seed"], "writer_kind": training["writer_kind"],
    "conditions": ["opaque:normal", "opaque:different_history", "opaque:drop", "opaque:zero", "counterfactual:normal", "short:normal"],
    "different_history": "next_opaque_world_in_frozen_order_cyclic_no_shared_entity_IDs",
    "short_condition": "transfer_diagnostic_not_matched_short_training",
    "counterfactual": "same_entities_global_room_permutation_not_individual_binding_swap",
    "stored_state_bytes": 66, "source_sha256": {str(path): digest(path) for path in sources},
    "generation": {"greedy": True, "max_new_tokens": 8, "batch_size": 1, "cache": False, "padding": False},
    "read_seconds": "generation_only_excludes_state_projection_and_answer_CE",
}
attach_execution(protocol, sources, device)
(args.output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
for path in sources:
    (args.output / path.name).write_bytes(path.read_bytes())
prepared = {}
state_records, predictions = [], []
started = time.perf_counter()
with torch.inference_mode():
    for variant in ("opaque", "counterfactual", "short"):
        prepared[variant] = []
        for row in worlds:
            world = row[variant]
            cases = [ReaderCase(**item) for item in world["queries"]]
            examples = [encode_memory_example(reader, case) for case in cases]
            chunks = encode_history_chunks(reader, cases[0], world["chunks"])
            state = writer.writer.empty(1)
            synchronize(device)
            tick = time.perf_counter()
            for chunk in chunks:
                state = writer.write(reader, state, torch.tensor(chunk, device=device))
                assert state.nbytes == 66
            synchronize(device)
            state_records.append({"variant": variant, "world_id": world["world_id"], "values": state.values.tolist(),
                                  "valid": state.valid.tolist(), "write_seconds": time.perf_counter() - tick,
                                  "history_chunks": chunks, "cases": [asdict(case) for case in cases],
                                  "native_examples": [asdict(example) for example in examples]})
            prepared[variant].append((world, cases, examples, state))
    with (args.output / "predictions.jsonl").open("x") as handle:
        for variant, values in prepared.items():
            for index, (world, cases, examples, state) in enumerate(values):
                memory = writer.memory_vectors(state)
                conditions = {"normal": memory}
                if variant == "opaque":
                    other = values[(index + 1) % len(values)]
                    assert set(world["entities"]).isdisjoint(other[0]["entities"])
                    conditions.update(different_history=writer.memory_vectors(other[-1]),
                                      drop=memory[:0], zero=torch.zeros_like(memory))
                for condition, current in conditions.items():
                    for case, example in zip(cases, examples, strict=True):
                        synchronize(device)
                        tick = time.perf_counter()
                        generated = generate_prefix_answer(reader, torch.tensor(example.before_ids, device=device), current,
                                                           torch.tensor(example.after_ids, device=device), max_new_tokens=8)
                        synchronize(device)
                        read_seconds = time.perf_counter() - tick
                        answer_ce = float(prefix_answer_loss(reader, torch.tensor(example.before_ids, device=device), current,
                                                            torch.tensor(example.after_ids, device=device), torch.tensor(example.answer_ids, device=device)))
                        prediction = {"condition": f"{variant}:{condition}", "world_id": world["world_id"], **asdict(case), **generated,
                                      "read_seconds": read_seconds, "answer_ce": answer_ce,
                                      "exact_match": reader_exact_match(generated["prediction"], case.answer, case.category)}
                        predictions.append(prediction)
                        handle.write(json.dumps(prediction, sort_keys=True) + "\n")
                    handle.flush()
                print(json.dumps({"variant": variant, "world_id": world["world_id"], "predictions": len(predictions)}), flush=True)
(args.output / "states.json").write_text(json.dumps(state_records, indent=2, sort_keys=True) + "\n")
summary = summarize_reader_predictions(predictions)
result = {"by_condition": summary["by_condition"], "seconds": time.perf_counter() - started,
          "predictions_sha256": digest(args.output / "predictions.jsonl"), "states_sha256": digest(args.output / "states.json")}
(args.output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2), flush=True)
