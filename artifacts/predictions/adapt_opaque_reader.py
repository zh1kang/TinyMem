#!/usr/bin/env python3
"""One fixed exact-format reader continuation using training worlds only."""

import argparse
import hashlib
import importlib.metadata
import json
import random
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
from peft import PeftModel

from tinymem.data.reader_gate import ReaderCase
from tinymem.evaluation.reader_gate import reader_exact_match, reader_messages, summarize_reader_predictions
from tinymem.research.pretrained import load_qwen_reader, verify_qwen_snapshot
from tinymem.research.reader_adaptation import encode_reader_answer, reader_answer_loss
from tinymem.utils.experiment import current_git_source_state
from tinymem.utils.seed import seed_everything


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-gate", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()
source_protocol = json.loads((args.source_gate / "protocol.json").read_text())
source_result = json.loads((args.source_gate / "results.json").read_text())
assert source_result["reader_accepted"] is False
assert digest(args.source_gate / "predictions.jsonl") == source_result["predictions_sha256"]
data = Path(source_protocol["data"])
assert digest(data / "protocol.json") == source_protocol["data_protocol_sha256"]
data_protocol = json.loads((data / "protocol.json").read_text())
assert digest(data / "train.json") == data_protocol["data_sha256"]["train.json"]
worlds = json.loads((data / "train.json").read_text())
assert len(worlds) == 256
evaluation_path = args.source_gate / "evaluation_manifest.json"
assert digest(evaluation_path) == source_protocol["evaluation_manifest_sha256"]
evaluation = json.loads(evaluation_path.read_text())
assert len(evaluation) == 864
source_groups = json.loads((data / "source_selection.json").read_text())
assert digest(data / "source_selection.json") == data_protocol["selection_sha256"]
assert {row["group_id"] for row in source_groups["train"]}.isdisjoint(row["group_id"] for row in source_groups["development"])
adapter = Path(source_protocol["adapter"])
assert all(digest(adapter / name) == expected for name, expected in source_protocol["adapter_sha256"].items())
categories = ("opaque_known", "opaque_missing", "short_known", "short_missing")
cases = defaultdict(list)
for row in worlds:
    for variant in ("opaque", "short"):
        for item in row[variant]["queries"]:
            case = ReaderCase(**item)
            category = variant + ("_missing" if case.answer == "unknown" else "_known")
            cases[category].append(case)
assert {category: len(cases[category]) for category in categories} == {
    "opaque_known": 2048, "opaque_missing": 256, "short_known": 2048, "short_missing": 256}
if args.dry_run:
    print(json.dumps({"counts": {category: len(cases[category]) for category in categories}, "steps": 100,
                      "microbatches": 400, "learning_rate": 0.00002, "qualification_predictions": len(evaluation),
                      "confirmation_used": False}))
    raise SystemExit(0)
args.output.mkdir(parents=True, exist_ok=False)
model_dir = Path("data/raw/pretrained/qwen3-1.7b")
snapshot = verify_qwen_snapshot(model_dir)
assert snapshot == source_protocol["snapshot"]
reader = load_qwen_reader(model_dir, device=torch.device("mps"), dtype=torch.bfloat16)
reader.model = PeftModel.from_pretrained(reader.model, adapter, is_trainable=True, local_files_only=True, use_safetensors=True)
reader.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
reader.model.train()
seed_everything(1337)
encoded = {category: [encode_reader_answer(reader, case) for case in cases[category]] for category in categories}
for item in evaluation:
    case = ReaderCase(**item["case"])
    prompt = reader.tokenizer.apply_chat_template(reader_messages(case, condition="question_only" if item["condition"] == "question_only" else "full_context"),
                                                  tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert prompt == item["prompt"]
    assert reader.tokenizer.encode(prompt, add_special_tokens=False) == item["prompt_ids"]
parameters = [value for value in reader.model.parameters() if value.requires_grad]
assert parameters and all("lora_" in name for name, value in reader.model.named_parameters() if value.requires_grad)
assert sum(parameter.numel() for parameter in parameters) == 1605632
reader.model.save_pretrained(args.output / "initial_adapter", save_embedding_layers=False)
training_manifest = {category: [{"case": asdict(case), "tokens": asdict(tokens)}
    for case, tokens in zip(cases[category], encoded[category], strict=True)] for category in categories}
(args.output / "training_manifest.json").write_text(json.dumps(training_manifest, indent=2, sort_keys=True) + "\n")
(args.output / "evaluation_manifest.json").write_bytes(evaluation_path.read_bytes())
sources = [Path(__file__), *(Path("src/tinymem") / name for name in (
    "research/reader_adaptation.py", "research/pretrained.py", "evaluation/reader_gate.py", "data/opaque_qa1.py"))]
protocol = {
    "protocol": "opaque_qa1_reader_continuation_v1", "claim": "development_reader_qualification_not_compression",
    "source": current_git_source_state(Path.cwd()).to_dict(), "snapshot": snapshot, "data": str(data),
    "data_protocol_sha256": digest(data / "protocol.json"), "development_sha256": source_protocol["development_sha256"],
    "source_gate": str(args.source_gate), "source_gate_protocol_sha256": digest(args.source_gate / "protocol.json"),
    "source_gate_results_sha256": digest(args.source_gate / "results.json"),
    "source_adapter": str(adapter), "source_adapter_sha256": source_protocol["adapter_sha256"],
    "training_manifest_sha256": digest(args.output / "training_manifest.json"),
    "evaluation_manifest_sha256": digest(args.output / "evaluation_manifest.json"),
    "source_sha256": {str(path): digest(path) for path in sources},
    "packages": {item.metadata["Name"]: item.version for item in importlib.metadata.distributions()},
    "steps": 100, "gradient_accumulation": 4, "microbatch_size": 1, "seed": 1337,
    "learning_rate": 0.00002, "weight_decay": 0.01, "clip_norm": 1.0, "optimizer_reset": True,
    "sampling": "one_uniform_query_per_each_of_four_variant_answerability_categories_per_update",
    "loss": "mean_answer_token_CE_including_EOT", "checkpoint_selection": "final_step_only",
    "trainable_parameters": 1605632, "training_cache": False, "generation": source_protocol["generation"],
    "acceptance": source_protocol["acceptance"], "confirmation_or_original_reserve_evaluated": False,
    "memory_measurement": "step_boundary_allocations_not_peak",
}
(args.output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
for path in sources:
    (args.output / path.name).write_bytes(path.read_bytes())
sampler = random.Random(1337)
optimizer = torch.optim.AdamW(parameters, lr=0.00002, weight_decay=0.01)
metrics = []
started = time.perf_counter()
print(f"artifacts: {args.output}", flush=True)
with (args.output / "metrics.jsonl").open("x") as handle:
    for step in range(1, 101):
        optimizer.zero_grad(set_to_none=True)
        step_started = time.perf_counter()
        losses, selected, tokens, targets = [], [], 0, 0
        for category in categories:
            example = sampler.choice(encoded[category])
            loss = reader_answer_loss(reader, example)
            (loss / 4).backward()
            losses.append(float(loss.detach()))
            selected.append(example.case_id)
            tokens += len(example.prompt_ids) + len(example.answer_ids) - 1
            targets += len(example.answer_ids)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.mps.synchronize()
        metric = {"step": step, "category_ce": dict(zip(categories, losses, strict=True)), "mean_ce": sum(losses) / 4,
                  "case_ids": selected, "gradient_norm": float(norm), "forward_tokens": tokens, "supervised_tokens": targets,
                  "seconds": time.perf_counter() - step_started, "mps_allocated_bytes": torch.mps.current_allocated_memory(),
                  "mps_driver_bytes": torch.mps.driver_allocated_memory()}
        metrics.append(metric)
        handle.write(json.dumps(metric, sort_keys=True) + "\n")
        handle.flush()
        if step % 10 == 0:
            print(json.dumps(metric), flush=True)
training_seconds = time.perf_counter() - started
checkpoint = args.output / "step_000100"
reader.model.eval().save_pretrained(checkpoint, save_embedding_layers=False)
(checkpoint / "reader_adapter_protocol.json").write_text(json.dumps({
    "protocol": "opaque_visible_reader_adapter_v1", "training_protocol_sha256": digest(args.output / "protocol.json"),
    "training_manifest_sha256": digest(args.output / "training_manifest.json"),
    "excluded_evaluation_manifest_sha256": digest(args.output / "evaluation_manifest.json"),
    "data_protocol_sha256": digest(data / "protocol.json"),
}, indent=2, sort_keys=True) + "\n")
reader.model.requires_grad_(False)
reader.model.gradient_checkpointing_disable()
predictions = []
started = time.perf_counter()
with (args.output / "predictions.jsonl").open("x") as handle:
    for item in evaluation:
        case = ReaderCase(**item["case"])
        generated = reader.generate([item["prompt"]], max_new_tokens=8)[0]
        row = {"condition": item["condition"], "world_id": item["world_id"], **asdict(case), **generated,
               "exact_match": reader_exact_match(generated["prediction"], case.answer, case.category)}
        predictions.append(row)
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        if len(predictions) % 100 == 0:
            print(json.dumps({"evaluation_done": len(predictions), "total": len(evaluation)}), flush=True)
summary = summarize_reader_predictions(predictions)
passed = all(summary["by_condition"][condition][category]["exact_accuracy"] >= 0.95
             for condition in ("full_opaque", "full_short") for category in ("opaque_qa1_known", "opaque_qa1_missing"))
empty = [row for row in predictions if row["condition"] == "question_only"]
empty_unknown = sum(reader_exact_match(row["prediction"], "unknown", row["category"]) for row in empty)
result = {"reader_accepted": passed and empty_unknown == len(empty), "by_condition": summary["by_condition"],
          "question_only_unknown": {"correct": empty_unknown, "count": len(empty)},
          "training_seconds": training_seconds, "generation_seconds": time.perf_counter() - started,
          "training_forward_tokens": sum(row["forward_tokens"] for row in metrics),
          "supervised_tokens": sum(row["supervised_tokens"] for row in metrics),
          "predictions_sha256": digest(args.output / "predictions.jsonl"), "final_adapter": str(checkpoint),
          "adapter_sha256": {name: digest(checkpoint / name) for name in
                             ("adapter_model.safetensors", "adapter_config.json", "reader_adapter_protocol.json")}}
(args.output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2), flush=True)
