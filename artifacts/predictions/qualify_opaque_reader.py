#!/usr/bin/env python3
"""Qualify visible-evidence and absent-entity answers before memory training."""

import argparse
import hashlib
import importlib.metadata
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from peft import PeftModel

from tinymem.data.reader_gate import ReaderCase
from tinymem.evaluation.reader_gate import reader_exact_match, reader_messages, summarize_reader_predictions
from tinymem.research.memory_prompt import encode_history_chunks, encode_memory_example
from tinymem.research.pretrained import load_qwen_reader, verify_qwen_snapshot
from tinymem.utils.experiment import current_git_source_state
from tinymem.utils.seed import seed_everything


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--adapter", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
data_protocol = json.loads((args.data / "protocol.json").read_text())
assert digest(args.data / "development.json") == data_protocol["data_sha256"]["development.json"]
assert data_protocol["original_reserved_groups_used"] is False
assert data_protocol["prior_native_consumed_confirmation_groups"] == 0
worlds = json.loads((args.data / "development.json").read_text())
assert len(worlds) == 32
assert digest(args.adapter / "adapter_model.safetensors") == "1c4402c1c808474f7f964e4e492215f98aac3998ea9c0b40fe2a81d6b26ad9f0"
assert digest(args.adapter / "adapter_config.json") == "7615a7007fac4f5ca871db58bd7fede2d552dfe67eb0524011da8574a0b2cd80"
assert digest(args.adapter / "reader_adapter_protocol.json") == "1c5f228fae1df6503f6fccbf5c0ff503d8927ad8f50e02ca0d422d5cf91475fa"
args.output.mkdir(parents=True, exist_ok=False)
model_dir = Path("data/raw/pretrained/qwen3-1.7b")
snapshot = verify_qwen_snapshot(model_dir)
reader = load_qwen_reader(model_dir, device=torch.device("mps"), dtype=torch.bfloat16)
reader.model = PeftModel.from_pretrained(reader.model, args.adapter, is_trainable=False, local_files_only=True, use_safetensors=True)
reader.model.eval().requires_grad_(False)
seed_everything(1337)
evaluation = []
encodings = []
for variant, condition in (("opaque", "full_opaque"), ("short", "full_short"), ("opaque", "question_only")):
    for row in worlds:
        world = row[variant]
        for item in world["queries"]:
            case = ReaderCase(**item)
            messages = reader_messages(case, condition="question_only" if condition == "question_only" else "full_context")
            prompt = reader.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            prompt_ids = reader.tokenizer.encode(prompt, add_special_tokens=False)
            assert len(prompt_ids) + 8 <= reader.model.config.max_position_embeddings
            evaluation.append({"condition": condition, "world_id": world["world_id"], "case": item, "prompt": prompt, "prompt_ids": prompt_ids})
            if condition != "question_only":
                tokens = encode_memory_example(reader, case)
                chunks = encode_history_chunks(reader, case, world["chunks"])
                assert len(chunks) == 4
                assert tuple(token for chunk in chunks for token in chunk) == tokens.history_ids
                encodings.append({"variant": variant, "world_id": world["world_id"], "tokens": asdict(tokens), "history_chunks": chunks})
assert len(evaluation) == 864 and len(encodings) == 576
assert all(sum(row["condition"] == condition for row in evaluation) == 288
           for condition in ("full_opaque", "full_short", "question_only"))
(args.output / "evaluation_manifest.json").write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n")
(args.output / "native_encodings.json").write_text(json.dumps(encodings, indent=2, sort_keys=True) + "\n")
sources = [Path(__file__), *(Path("src/tinymem") / value for value in (
    "research/pretrained.py", "research/memory_prompt.py", "evaluation/reader_gate.py", "data/opaque_qa1.py"))]
protocol = {
    "protocol": "opaque_qa1_reader_qualification_v1", "claim": "development_reader_qualification_not_compression",
    "source": current_git_source_state(Path.cwd()).to_dict(), "snapshot": snapshot,
    "data": str(args.data), "data_protocol_sha256": digest(args.data / "protocol.json"),
    "development_sha256": digest(args.data / "development.json"),
    "adapter": str(args.adapter), "adapter_sha256": {name: digest(args.adapter / name) for name in
        ("adapter_model.safetensors", "adapter_config.json", "reader_adapter_protocol.json")},
    "source_sha256": {str(path): digest(path) for path in sources},
    "evaluation_manifest_sha256": digest(args.output / "evaluation_manifest.json"),
    "native_encodings_sha256": digest(args.output / "native_encodings.json"),
    "packages": {item.metadata["Name"]: item.version for item in importlib.metadata.distributions()},
    "generation": {"batch_size": 1, "max_new_tokens": 8, "do_sample": False, "enable_thinking": False,
                   "cache": "generation-local_only", "padding": "none"},
    "acceptance": {"full_opaque_known": 0.95, "full_opaque_missing": 0.95,
                   "full_short_known": 0.95, "full_short_missing": 0.95, "question_only_unknown": 1.0},
    "count": len(evaluation), "worlds": 32, "source_groups": 64,
    "confirmation_or_original_reserve_evaluated": False,
}
(args.output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
for path in sources:
    (args.output / path.name).write_bytes(path.read_bytes())
predictions = []
started = time.perf_counter()
with (args.output / "predictions.jsonl").open("x") as handle:
    for item in evaluation:
        case = ReaderCase(**item["case"])
        generated = reader.generate([item["prompt"]], max_new_tokens=8)[0]
        result = {"condition": item["condition"], "world_id": item["world_id"], **asdict(case), **generated,
                  "exact_match": reader_exact_match(generated["prediction"], case.answer, case.category)}
        predictions.append(result)
        handle.write(json.dumps(result, sort_keys=True) + "\n")
        handle.flush()
        if len(predictions) % 100 == 0:
            print(json.dumps({"completed": len(predictions), "total": len(evaluation), "seconds": time.perf_counter() - started}), flush=True)
summary = summarize_reader_predictions(predictions)
passed = all(summary["by_condition"][condition][category]["exact_accuracy"] >= 0.95
             for condition in ("full_opaque", "full_short") for category in ("opaque_qa1_known", "opaque_qa1_missing"))
empty = [row for row in predictions if row["condition"] == "question_only"]
empty_unknown = sum(reader_exact_match(row["prediction"], "unknown", row["category"]) for row in empty)
result = {"reader_accepted": passed and empty_unknown == len(empty), "by_condition": summary["by_condition"],
          "question_only_unknown": {"correct": empty_unknown, "count": len(empty)},
          "seconds": time.perf_counter() - started, "predictions_sha256": digest(args.output / "predictions.jsonl")}
(args.output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2), flush=True)
