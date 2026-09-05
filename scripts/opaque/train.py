#!/usr/bin/env python3
"""Profile or train matched-state memory writers on the frozen association task."""

import argparse
import importlib.metadata
import json
import math
import random
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import save_file

from tinymem.data.reader_gate import ReaderCase
from tinymem.research.study_runtime import (
    allocation_metrics, attach_execution, check_repository, prepare_device, repository_path, sha256, synchronize,
)

from tinymem.evaluation.reader_gate import reader_exact_match
from tinymem.research.memory_prompt import encode_history_chunks, encode_memory_example
from tinymem.research.native_training import native_history_answer_loss
from tinymem.research.prefix_reader import generate_prefix_answer
from tinymem.research.pretrained import load_qwen_reader, verify_qwen_snapshot
from tinymem.research.recurrent_memory import NativeRecurrentMemory
from tinymem.utils.experiment import current_git_source_state
from tinymem.utils.seed import seed_everything


def digest(path):
    return sha256(path)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=repository_path, required=True)
parser.add_argument("--reader-gate", type=repository_path, required=True)
parser.add_argument("--output", type=repository_path, required=True)
parser.add_argument("--writer-kind", choices=("query_pool", "mean_pool"), required=True)
parser.add_argument("--seed", type=int, choices=(1337, 2027, 4099), default=1337)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--profile", action="store_true")
parser.add_argument("--study-protocol", type=repository_path)
parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
args = parser.parse_args()
check_repository()
device = prepare_device(args.device)
if args.profile:
    assert args.steps == 10 and args.seed == 1337
else:
    assert args.study_protocol is not None
    frozen = json.loads(args.study_protocol.read_text())
    assert args.steps == frozen["steps"] and args.seed in frozen["seeds"]
    assert args.writer_kind in frozen["writers"]
    assert all(digest(Path(path)) == expected for path, expected in frozen["source_sha256"].items())
gate = json.loads((args.reader_gate / "protocol.json").read_text())
gate_result = json.loads((args.reader_gate / "results.json").read_text())
assert gate_result["reader_accepted"] is True
data_protocol = json.loads((args.data / "protocol.json").read_text())
assert digest(args.data / "protocol.json") == gate["data_protocol_sha256"]
assert digest(args.data / "train.json") == data_protocol["data_sha256"]["train.json"]
assert digest(args.data / "development.json") == data_protocol["data_sha256"]["development.json"]
if not args.profile:
    assert frozen["data_protocol_sha256"] == digest(args.data / "protocol.json")
    assert frozen["reader_gate_results_sha256"] == digest(args.reader_gate / "results.json")
    assert frozen["reader_gate_protocol_sha256"] == digest(args.reader_gate / "protocol.json")
if gate["protocol"] == "opaque_qa1_reader_qualification_v1":
    adapter, adapter_hashes = repository_path(gate["adapter"]), gate["adapter_sha256"]
elif gate["protocol"] == "opaque_qa1_reader_continuation_v1":
    adapter, adapter_hashes = repository_path(gate_result["final_adapter"]), gate_result["adapter_sha256"]
else:
    raise ValueError("unsupported reader qualification protocol")
assert all(digest(adapter / name) == expected for name, expected in adapter_hashes.items())
args.output.mkdir(parents=True, exist_ok=False)
snapshot = verify_qwen_snapshot(Path("data/raw/pretrained/qwen3-1.7b"))
assert snapshot == gate["snapshot"]
reader = load_qwen_reader(Path("data/raw/pretrained/qwen3-1.7b"), device=device, dtype=torch.bfloat16)
reader.model = PeftModel.from_pretrained(reader.model, adapter, is_trainable=False, local_files_only=True, use_safetensors=True)
reader.model.requires_grad_(False)
reader.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
reader.model.train()
train = json.loads((args.data / "train.json").read_text())
development = json.loads((args.data / "development.json").read_text())
encoded = {}
for split, rows in (("train", train), ("development", development)):
    encoded[split] = []
    for row in rows:
        world = row["opaque"]
        cases = [ReaderCase(**case) for case in world["queries"]]
        examples = [encode_memory_example(reader, case) for case in cases]
        chunks = encode_history_chunks(reader, cases[0], world["chunks"])
        assert len(examples) == 9 and len(chunks) == 4
        encoded[split].append((world, cases, examples, chunks))
max_chunk = max(len(chunk) for split in encoded.values() for _, _, _, chunks in split for chunk in chunks)
assert max_chunk <= 512
seed_everything(args.seed)
width = 64 if args.writer_kind == "query_pool" else 82
writer = NativeRecurrentMemory(2048, memory_width=8, slots=2, segment_length=512,
                               writer_kind=args.writer_kind, aggregation_width=width).to(device)
generator = torch.Generator(device="cpu").manual_seed(args.seed)
projection = torch.empty(2048, 8).uniform_(-1 / math.sqrt(8), 1 / math.sqrt(8), generator=generator)
with torch.no_grad():
    writer.read_projection.weight.copy_(projection.to(device))
assert writer.writer.empty(1).nbytes == 66
parameters = list(writer.parameters())
parameter_count = sum(parameter.numel() for parameter in parameters)
assert parameter_count == (185040 if args.writer_kind == "query_pool" else 185066)
save_file({name: value.detach().cpu().contiguous() for name, value in writer.state_dict().items()}, args.output / "initial_writer.safetensors")
sources = [Path(__file__), *(Path("src/tinymem") / name for name in (
    "research/native_training.py", "research/memory_prompt.py", "research/recurrent_memory.py", "research/prefix_reader.py",
    "research/pretrained.py", "memory/query_pool_slots.py", "memory/mean_pool_slots.py", "memory/recurrent_slots.py"))]
protocol = {
    "protocol": "opaque_qa1_memory_training_v1", "claim": "compute_profile_not_confirmation" if args.profile else "fixed_budget_training_no_checkpoint_selection",
    "source": current_git_source_state(Path.cwd()).to_dict(), "snapshot": snapshot, "arguments": vars(args) | {
        key: str(value) for key, value in vars(args).items() if isinstance(value, Path)},
    "data_protocol_sha256": digest(args.data / "protocol.json"), "training_sha256": digest(args.data / "train.json"),
    "development_sha256": digest(args.data / "development.json"), "reader_gate_protocol_sha256": digest(args.reader_gate / "protocol.json"),
    "reader_gate_results_sha256": digest(args.reader_gate / "results.json"), "adapter": str(adapter), "adapter_sha256": adapter_hashes,
    "study_protocol_sha256": None if args.profile else digest(args.study_protocol),
    "source_sha256": {str(path): digest(path) for path in sources},
    "initial_writer_sha256": digest(args.output / "initial_writer.safetensors"),
    "initial_projection": "independent_CPU_generator_seed_uniform_linear_initialization_copied_to_both_methods",
    "packages": {item.metadata["Name"]: item.version for item in importlib.metadata.distributions()},
    "writer_kind": args.writer_kind, "aggregation_width": width, "shared_parameters": parameter_count,
    "memory_width": 8, "slots": 2, "stored_bytes": 66, "segment_length_limit": 512, "max_actual_chunk": max_chunk,
    "seed": args.seed, "steps": args.steps, "learning_rate": 0.001, "weight_decay": 0.01, "clip_norm": 1.0,
    "batch": "one_world_nine_serial_query_losses", "sampling": "seeded_shuffle_without_replacement_each_epoch",
    "loss": "mean_of_nine_per_query_answer_token_CEs_including_EOT", "bptt": "full_four_chunks_no_detach",
    "reader_frozen": True, "reader_dtype": "bfloat16", "writer_dtype": "float32", "training_cache": False,
    "checkpoint_selection": "final_only", "checkpoint_every": 100,
    "memory_measurement": "sampled_boundary_allocations_not_peak", "confirmation_or_external_used": False,
}
attach_execution(protocol, sources, device)
(args.output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
(args.output / "native_encodings.json").write_text(json.dumps({
    split: [{"world_id": world["world_id"], "queries": [asdict(example) for example in examples], "history_chunks": chunks}
            for world, _, examples, chunks in values] for split, values in encoded.items()
}, indent=2, sort_keys=True) + "\n")
for path in sources:
    (args.output / path.name).write_bytes(path.read_bytes())
sampler = random.Random(args.seed)
schedule = []
while len(schedule) < args.steps:
    epoch = list(range(len(encoded["train"])))
    sampler.shuffle(epoch)
    schedule.extend(epoch)
schedule = schedule[:args.steps]
(args.output / "schedule.json").write_text(json.dumps(schedule) + "\n")
(args.output / "input_artifact_hashes.json").write_text(json.dumps({
    "protocol_sha256": digest(args.output / "protocol.json"),
    "native_encodings_sha256": digest(args.output / "native_encodings.json"),
    "schedule_sha256": digest(args.output / "schedule.json"),
}, indent=2, sort_keys=True) + "\n")
optimizer = torch.optim.AdamW(parameters, lr=0.001, weight_decay=0.01)
states = []
if args.profile:
    original_write = writer.write

    def observe_write(shared_reader, state, ids):
        updated = original_write(shared_reader, state, ids)
        updated.values.retain_grad()
        states.append(updated)
        return updated

    writer.write = observe_write
metrics = []
started = time.perf_counter()
print(f"artifacts: {args.output}", flush=True)
with (args.output / "metrics.jsonl").open("x") as handle:
    for step, index in enumerate(schedule, 1):
        world, _, examples, chunks = encoded["train"][index]
        states.clear()
        optimizer.zero_grad(set_to_none=True)
        synchronize(device)
        step_started = time.perf_counter()
        loss = native_history_answer_loss(reader, writer, examples, history_chunks=chunks)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        assert all(value.grad is None for value in reader.model.parameters())
        optimizer.step()
        synchronize(device)
        write_tokens = sum(map(len, chunks)) + (6 if args.writer_kind == "query_pool" else 5)
        read_tokens = sum(len(example.before_ids) + 2 + len(example.after_ids) + len(example.answer_ids) - 1 for example in examples)
        metric = {"step": step, "world_id": world["world_id"], "answer_ce": float(loss.detach()), "gradient_norm": float(norm),
                  "forward_tokens": write_tokens + read_tokens, "supervised_tokens": sum(len(example.answer_ids) for example in examples),
                  "seconds": time.perf_counter() - step_started, **allocation_metrics(device)}
        if args.profile:
            gradients = [float(state.values.grad.norm()) for state in states]
            assert len(gradients) == 4 and all(math.isfinite(value) and value > 0 for value in gradients)
            metric["state_gradient_norms"] = gradients
            metric["valid_slot_counts"] = [int(state.valid.sum()) for state in states]
        metrics.append(metric)
        handle.write(json.dumps(metric, sort_keys=True) + "\n")
        handle.flush()
        if args.profile or step % 10 == 0:
            print(json.dumps(metric), flush=True)
        if step % 100 == 0 or step == args.steps:
            save_file({name: value.detach().cpu().contiguous() for name, value in writer.state_dict().items()}, args.output / f"step_{step:06d}.safetensors")
            torch.save({"step": step, "optimizer": optimizer.state_dict(), "protocol_sha256": digest(args.output / "protocol.json")},
                       args.output / f"optimizer_{step:06d}.pt")
training_seconds = time.perf_counter() - started
result = {"profile": args.profile, "training_seconds": training_seconds, "median_step_seconds_after_first": statistics.median(row["seconds"] for row in metrics[1:]),
          "forward_tokens": sum(row["forward_tokens"] for row in metrics), "supervised_tokens": sum(row["supervised_tokens"] for row in metrics),
          "checkpoint_sha256": digest(args.output / f"step_{args.steps:06d}.safetensors"), "confirmation_evaluated": False,
          "input_artifact_hashes_sha256": digest(args.output / "input_artifact_hashes.json"),
          "metrics_sha256": digest(args.output / "metrics.jsonl"),
          "token_accounting": "logical_forward_inputs_excluding_checkpoint_recomputation"}
if not args.profile:
    reader.model.eval()
    reader.model.gradient_checkpointing_disable()
    predictions = []
    evaluation_started = time.perf_counter()
    with torch.inference_mode(), (args.output / "development_predictions.jsonl").open("x") as handle:
        for world, cases, examples, chunks in encoded["development"]:
            state = writer.writer.empty(1)
            for chunk in chunks:
                state = writer.write(reader, state, torch.tensor(chunk, device=device))
            memory = writer.memory_vectors(state)
            for case, example in zip(cases, examples, strict=True):
                generated = generate_prefix_answer(reader, torch.tensor(example.before_ids, device=device), memory,
                                                   torch.tensor(example.after_ids, device=device), max_new_tokens=8)
                row = {"world_id": world["world_id"], **asdict(case), **generated,
                       "exact_match": reader_exact_match(generated["prediction"], case.answer, case.category)}
                predictions.append(row)
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
    result["development"] = {category: {"correct": sum(row["exact_match"] for row in predictions if row["category"] == category),
                                          "count": sum(row["category"] == category for row in predictions)}
                             for category in ("opaque_qa1_known", "opaque_qa1_missing")}
    result["evaluation_seconds"] = time.perf_counter() - evaluation_started
(args.output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2), flush=True)
