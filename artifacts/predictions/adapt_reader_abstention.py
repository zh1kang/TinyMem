#!/usr/bin/env python3
"""Fixed-budget visible-evidence abstention continuation of the shared reader."""

import argparse
import hashlib
import importlib.metadata
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
from peft import PeftModel

from tinymem.data.qa1_queries import qa1_world_queries
from tinymem.data.reader_gate import ReaderCase
from tinymem.evaluation.reader_gate import reader_exact_match, reader_messages, summarize_reader_predictions
from tinymem.research.pretrained import load_qwen_reader, verify_qwen_snapshot
from tinymem.research.reader_adaptation import encode_reader_answer, reader_answer_loss
from tinymem.utils.experiment import current_git_source_state
from tinymem.utils.seed import seed_everything


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expand(rows):
    result = []
    for row in rows:
        case = ReaderCase(**row["case"])
        result.extend(qa1_world_queries(case, ("Mary", "John", "Daniel", "Sandra")) if case.category == "babi_qa1" else (case,))
    assert len({case.case_id for case in result}) == len(result)
    return result


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-run", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()
source_protocol = json.loads((args.source_run / "protocol.json").read_text())
manifest_path = args.source_run / "data_manifest.json"
assert digest(manifest_path) == source_protocol["data_manifest_sha256"]
source_data = json.loads(manifest_path.read_text())
adapter = Path(json.loads((args.source_run / "results.json").read_text())["best_checkpoint"])
adapter_protocol = json.loads((adapter / "reader_adapter_protocol.json").read_text())
assert adapter_protocol["training_manifest_sha256"] == digest(manifest_path)
assert adapter_protocol["training_protocol_sha256"] == digest(args.source_run / "protocol.json")
gate_run = Path(source_protocol["arguments"]["gate_run"])
gate_manifest = gate_run / "data_manifest.json"
assert digest(gate_manifest) == adapter_protocol["excluded_gate_manifest_sha256"]
gate_data = json.loads(gate_manifest.read_text())
train, development = (expand(source_data[split]) for split in ("train", "development"))
gate = [ReaderCase(**row["case"]) for row in gate_data["cases"] if row["condition"] == "full_context"]
partial_gate = [query for case in gate if case.category == "babi_qa1" for query in qa1_world_queries(case, ("Mary", "John", "Daniel", "Sandra"))]
for field in ("context", "history_id"):
    train_values = {getattr(case, field) for case in train}
    development_values = {getattr(case, field) for case in development}
    gate_values = {getattr(case, field) for case in gate}
    assert train_values.isdisjoint(development_values)
    assert train_values.isdisjoint(gate_values)
    assert development_values.isdisjoint(gate_values)
counts = {name: dict(Counter(case.category for case in rows)) for name, rows in (("train", train), ("development", development), ("partial_gate", partial_gate))}
print(json.dumps({"data_counts": counts}), flush=True)
if args.dry_run:
    raise SystemExit(0)
args.output.mkdir(parents=True, exist_ok=False)
model_dir = Path(source_protocol["arguments"]["model_dir"])
snapshot = verify_qwen_snapshot(model_dir)
reader = load_qwen_reader(model_dir, device=torch.device("mps"), dtype=torch.bfloat16)
reader.model = PeftModel.from_pretrained(reader.model, adapter, is_trainable=True, local_files_only=True, use_safetensors=True)
reader.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
reader.model.train()
seed_everything(1337)
encoded = {name: [encode_reader_answer(reader, case) for case in rows] for name, rows in (("train", train), ("development", development))}
evaluation = []
for condition, case in ([("full_context", case) for case in gate] + [("question_only", case) for case in gate] + [("partial_context", case) for case in partial_gate]):
    messages = reader_messages(case, condition="question_only" if condition == "question_only" else "full_context")
    prompt = reader.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = reader.tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) + 16 > reader.model.config.max_position_embeddings:
        raise ValueError("evaluation prompt exceeds context")
    evaluation.append({"condition": condition, "case": asdict(case), "prompt": prompt, "prompt_ids": ids})
evaluation_text = json.dumps(evaluation, indent=2, sort_keys=True) + "\n"
(args.output / "evaluation_manifest.json").write_text(evaluation_text)
manifest = json.dumps({
    name: [{"case": asdict(case), "tokens": asdict(tokens)} for case, tokens in zip(rows, encoded[name], strict=True)]
    for name, rows in (("train", train), ("development", development))
}, indent=2, sort_keys=True) + "\n"
(args.output / "data_manifest.json").write_text(manifest)
parameters = [value for name, value in reader.model.named_parameters() if value.requires_grad]
assert parameters and all("lora_" in name for name, value in reader.model.named_parameters() if value.requires_grad)
sources = [Path(__file__), *(Path("src/tinymem") / name for name in (
    "data/qa1_queries.py", "data/symbolic_world.py", "research/reader_adaptation.py", "research/pretrained.py", "evaluation/reader_gate.py"))]
protocol = {
    "protocol": "reader_partial_evidence_continuation_v1", "claim": "development_reader_qualification_not_compression",
    "source": current_git_source_state(Path.cwd()).to_dict(), "snapshot": snapshot,
    "source_run": str(args.source_run), "source_manifest_sha256": digest(manifest_path),
    "source_adapter": str(adapter), "source_adapter_sha256": {name: digest(adapter / name) for name in ("adapter_config.json", "adapter_model.safetensors", "reader_adapter_protocol.json")},
    "excluded_gate_manifest_sha256": digest(gate_manifest), "data_manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
    "evaluation_manifest_sha256": hashlib.sha256(evaluation_text.encode()).hexdigest(),
    "source_sha256": {str(path): digest(path) for path in sources}, "counts": counts,
    "packages": {item.metadata["Name"]: item.version for item in importlib.metadata.distributions()},
    "seed": 1337, "steps": 200, "gradient_accumulation": 4, "microbatch_size": 1,
    "learning_rate": 0.00005, "weight_decay": 0.01, "clip_norm": 1.0,
    "sampling": "uniform_six_categories_then_uniform_query", "objective": "mean_answer_token_ce_including_eot",
    "optimizer_reset": True, "trainable_parameters": sum(parameter.numel() for parameter in parameters),
    "checkpoint_selection": "final_step_only_no_development_selection", "cache_during_training": False,
    "generation": {"batch_size": 4, "max_new_tokens": 16, "do_sample": False, "enable_thinking": False, "kv_scope": "one_generate_call_no_persistent_stream"},
    "reader_acceptance": {"full_context_categories": ["babi_qa1", "correction_changed", "correction_unchanged", "exact_copy", "randomized_bindings"],
                          "partial_context_categories": ["babi_qa1", "missing_entity"], "minimum_accuracy_each": 0.95, "question_only_unknown": 1.0},
    "evaluation_split": "previously_consumed_development_gate_plus_new_questions_on_its_excluded_histories",
    "external_or_final_test_used": False, "memory_measurement": "step_boundary_mps_allocation_not_peak",
}
protocol_text = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
(args.output / "protocol.json").write_text(protocol_text)
for path in sources:
    (args.output / path.name).write_bytes(path.read_bytes())
initial = args.output / "initial_adapter"
reader.model.save_pretrained(initial, save_embedding_layers=False)
by_category = defaultdict(list)
for case, example in zip(train, encoded["train"], strict=True):
    by_category[case.category].append(example)
categories = sorted(by_category)
assert len(categories) == 6
generator = random.Random(1337)
optimizer = torch.optim.AdamW(parameters, lr=0.00005, weight_decay=0.01)
started = time.perf_counter()
metrics = []
print(f"artifacts: {args.output}", flush=True)
with (args.output / "metrics.jsonl").open("x") as handle:
    for step in range(1, 201):
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        losses, tokens, targets, selected = [], 0, 0, []
        for _ in range(4):
            category = generator.choice(categories)
            example = generator.choice(by_category[category])
            loss = reader_answer_loss(reader, example)
            (loss / 4).backward()
            losses.append(float(loss.detach()))
            tokens += len(example.prompt_ids) + len(example.answer_ids) - 1
            targets += len(example.answer_ids)
            selected.append(example.case_id)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.mps.synchronize()
        metric = {"step": step, "answer_ce": sum(losses) / 4, "case_ids": selected, "gradient_norm": float(norm),
                  "forward_tokens": tokens, "supervised_tokens": targets, "seconds": time.perf_counter() - step_started,
                  "mps_allocated_bytes": torch.mps.current_allocated_memory(), "mps_driver_bytes": torch.mps.driver_allocated_memory()}
        metrics.append(metric)
        handle.write(json.dumps(metric, sort_keys=True) + "\n")
        handle.flush()
        if step % 10 == 0:
            print(json.dumps(metric), flush=True)
training_seconds = time.perf_counter() - started
checkpoint = args.output / "step_000200"
reader.model.eval().save_pretrained(checkpoint, save_embedding_layers=False)
(checkpoint / "reader_adapter_protocol.json").write_text(json.dumps({
    "protocol": "visible_reader_adapter_v1", "excluded_gate_manifest_sha256": digest(gate_manifest),
    "training_protocol_sha256": hashlib.sha256(protocol_text.encode()).hexdigest(),
    "training_manifest_sha256": protocol["data_manifest_sha256"],
}, indent=2, sort_keys=True) + "\n")
reader.model.requires_grad_(False)
reader.model.gradient_checkpointing_disable()
development_losses = defaultdict(list)
with torch.no_grad():
    for case, example in zip(development, encoded["development"], strict=True):
        development_losses[case.category].append(float(reader_answer_loss(reader, example)))
predictions = []
started = time.perf_counter()
with (args.output / "predictions.jsonl").open("x") as handle:
    for start in range(0, len(evaluation), 4):
        batch = evaluation[start:start + 4]
        generated = reader.generate([row["prompt"] for row in batch], max_new_tokens=16)
        for prepared, output in zip(batch, generated, strict=True):
            case = ReaderCase(**prepared["case"])
            row = {"condition": prepared["condition"], **asdict(case), **output,
                   "exact_match": reader_exact_match(output["prediction"], case.answer, case.category)}
            predictions.append(row)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        if (start + len(batch)) % 100 == 0:
            print(json.dumps({"evaluation_done": len(predictions), "total": len(evaluation)}), flush=True)
summary = summarize_reader_predictions(predictions)
full = summary["by_condition"]["full_context"]
partial = summary["by_condition"]["partial_context"]
empty = [row for row in predictions if row["condition"] == "question_only"]
empty_unknown = sum(reader_exact_match(row["prediction"], "unknown", row["category"]) for row in empty)
passed = all(full[name]["exact_accuracy"] >= 0.95 for name in ("babi_qa1", "correction_changed", "correction_unchanged", "exact_copy", "randomized_bindings"))
passed = passed and all(partial[name]["exact_accuracy"] >= 0.95 for name in ("babi_qa1", "missing_entity")) and empty_unknown == len(empty)
result = {"reader_accepted": passed, **summary, "empty_context_unknown": {"count": len(empty), "correct": empty_unknown},
          "development_category_ce": {key: sum(value) / len(value) for key, value in development_losses.items()},
          "training_seconds": training_seconds, "generation_seconds": time.perf_counter() - started,
          "training_forward_tokens": sum(row["forward_tokens"] for row in metrics), "supervised_tokens": sum(row["supervised_tokens"] for row in metrics),
          "final_checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint / "adapter_model.safetensors")}
(args.output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2), flush=True)
