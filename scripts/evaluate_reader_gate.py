#!/usr/bin/env python3
"""Screen pinned Qwen on visible development evidence before memory training."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from tinymem.data.reader_gate import BABI_GATE_FILES, make_reader_gate_cases
from tinymem.evaluation.reader_gate import reader_exact_match, reader_messages, summarize_reader_predictions
from tinymem.research.pretrained import load_qwen_reader, verify_qwen_snapshot
from tinymem.utils.device import select_device
from tinymem.utils.experiment import current_git_source_state
from tinymem.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("data/raw/pretrained/qwen3-1.7b"))
    parser.add_argument("--babi-root", type=Path, default=Path("data/raw/tasks_1-20_v1-2/en-10k"))
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--examples-per-task", type=int, default=100)
    parser.add_argument("--replacement-examples", type=int, default=200)
    parser.add_argument("--copy-examples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="mps")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/predictions/reader_gate"))
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise ValueError("batch size and max_new_tokens must be positive")
    adapter_fingerprint = None
    excluded_gate_hash = None
    if args.adapter is not None:
        adapter_fingerprint = {
            name: hashlib.sha256((args.adapter / name).read_bytes()).hexdigest()
            for name in ("adapter_config.json", "adapter_model.safetensors", "reader_adapter_protocol.json")
        }
        adapter_protocol = json.loads((args.adapter / "reader_adapter_protocol.json").read_text())
        excluded_gate_hash = adapter_protocol["excluded_gate_manifest_sha256"]
        if adapter_protocol["protocol"] != "visible_reader_adapter_v1" or not excluded_gate_hash:
            raise ValueError("adapter must record the gate excluded from its training")
    device = select_device(args.device)
    seed_everything(args.seed)
    cases = make_reader_gate_cases(
        args.babi_root, count=args.examples_per_task, replacement_count=args.replacement_examples,
        copy_count=args.copy_examples, seed=args.seed,
    )
    snapshot = verify_qwen_snapshot(args.model_dir)
    source = current_git_source_state(Path(__file__).resolve().parents[1])
    started = time.perf_counter()
    reader = load_qwen_reader(args.model_dir, device=device, dtype=getattr(torch, args.dtype))
    if args.adapter is not None:
        from peft import PeftModel

        reader.model = PeftModel.from_pretrained(reader.model, args.adapter, is_trainable=False, local_files_only=True, use_safetensors=True)
        reader.model.eval().requires_grad_(False)
    load_seconds = time.perf_counter() - started
    conditions = ("full_context", "question_only")
    prepared = []
    for condition in conditions:
        for case in cases:
            messages = reader_messages(case, condition=condition)
            prompt = reader.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            ids = reader.tokenizer.encode(prompt, add_special_tokens=False)
            if len(ids) + args.max_new_tokens > reader.model.config.max_position_embeddings:
                raise ValueError(f"case {case.case_id} exceeds the reader context")
            prepared.append({"case": asdict(case), "condition": condition, "messages": messages, "prompt": prompt, "prompt_ids": ids})
    manifest = json.dumps({
        "split": "development_only", "cases": prepared,
        "babi_source_sha256": {name: hashlib.sha256((args.babi_root / name).read_bytes()).hexdigest() for name in BABI_GATE_FILES.values()},
        "replacement_protocol": "history_disjoint_v2", "replacement_split": "validation",
    }, sort_keys=True, indent=2) + "\n"
    manifest_hash = hashlib.sha256(manifest.encode()).hexdigest()
    if excluded_gate_hash is not None and manifest_hash != excluded_gate_hash:
        raise ValueError("adapter evaluation must use the exact gate excluded from training")
    run = args.artifact_root / f"{datetime.now(UTC):%Y%m%dT%H%M%S.%fZ}-{uuid.uuid4().hex[:8]}"
    run.mkdir(parents=True, exist_ok=False)
    (run / "data_manifest.json").write_text(manifest, encoding="utf-8")
    protocol = {
        "protocol": "visible_reader_gate_v1", "source": source.to_dict(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "snapshot": snapshot, "data_manifest_sha256": manifest_hash,
        "adapter_sha256": adapter_fingerprint,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": {item.metadata["Name"]: item.version for item in importlib.metadata.distributions()}},
        "model_config": reader.model.config.to_dict(), "reader_parameters": sum(p.numel() for p in reader.model.parameters()),
        "generation": {"enable_thinking": False, "do_sample": False, "max_new_tokens": args.max_new_tokens, "cache_scope": "one_generate_call"},
        "gate": {"categories": ["babi_qa1", "correction_changed", "correction_unchanged"], "minimum_count_each": 100, "threshold_each": 0.95},
        "load_seconds": load_seconds,
    }
    (run / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"artifacts: {run}", flush=True)
    predictions = []
    started = time.perf_counter()
    with (run / "predictions.jsonl").open("x", encoding="utf-8") as handle:
        for start in range(0, len(prepared), args.batch_size):
            batch = prepared[start:start + args.batch_size]
            generated = reader.generate([row["prompt"] for row in batch], max_new_tokens=args.max_new_tokens)
            for row, output in zip(batch, generated, strict=True):
                case = row["case"]
                prediction = {"condition": row["condition"], **case, **output,
                              "exact_match": reader_exact_match(output["prediction"], case["answer"], case["category"])}
                predictions.append(prediction)
                handle.write(json.dumps(prediction, sort_keys=True) + "\n")
            handle.flush()
            print(json.dumps({"done": len(predictions), "total": len(prepared), "seconds": time.perf_counter() - started}), flush=True)
    result = {
        **summarize_reader_predictions(predictions), "inference_seconds": time.perf_counter() - started,
        "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024),
        "memory_measurement": "process_lifetime_max_rss_not_persistent_stream_bytes_or_mps_peak",
    }
    if device.type == "mps":
        result["mps_end_allocated_bytes"] = torch.mps.current_allocated_memory()
        result["mps_end_driver_bytes"] = torch.mps.driver_allocated_memory()
    (run / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
