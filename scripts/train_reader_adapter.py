#!/usr/bin/env python3
"""Adapt the shared reader on visible evidence, with an isolated internal split."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import time
import uuid
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from tinymem.data.reader_adaptation import make_reader_adaptation_data
from tinymem.data.reader_gate import BABI_GATE_FILES, ReaderCase
from tinymem.research.pretrained import load_qwen_reader, verify_qwen_snapshot
from tinymem.research.reader_adaptation import attach_reader_lora, encode_reader_answer, reader_answer_loss
from tinymem.utils.device import select_device
from tinymem.utils.experiment import current_git_source_state
from tinymem.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-run", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("data/raw/pretrained/qwen3-1.7b"))
    parser.add_argument("--babi-root", type=Path, default=Path("data/raw/tasks_1-20_v1-2/en-10k"))
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--train-per-task", type=int, default=400)
    parser.add_argument("--development-per-task", type=int, default=50)
    parser.add_argument("--replacement-examples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--evaluate-every", type=int, default=100)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/predictions/reader_adaptation"))
    args = parser.parse_args()
    if min(args.steps, args.gradient_accumulation, args.rank, args.evaluate_every) <= 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("training counts, rank, and learning rate must be positive")
    gate_path = args.gate_run / "data_manifest.json"
    gate_protocol = json.loads((args.gate_run / "protocol.json").read_text())
    gate_text = gate_path.read_bytes()
    gate_hash = hashlib.sha256(gate_text).hexdigest()
    if gate_hash != gate_protocol["data_manifest_sha256"]:
        raise ValueError("gate manifest hash does not match its protocol")
    gate = json.loads(gate_text)
    source_hashes = {name: hashlib.sha256((args.babi_root / name).read_bytes()).hexdigest() for name in BABI_GATE_FILES.values()}
    if source_hashes != gate["babi_source_sha256"]:
        raise ValueError("bAbI sources differ from the excluded reader gate")
    cases = [ReaderCase(**row["case"]) for row in gate["cases"] if row["condition"] == "full_context"]
    data = make_reader_adaptation_data(
        args.babi_root, cases, train_per_task=args.train_per_task,
        development_per_task=args.development_per_task,
        replacement_examples=args.replacement_examples, seed=args.data_seed,
    )
    device = select_device(args.device)
    seed_everything(args.seed)
    snapshot = verify_qwen_snapshot(args.model_dir)
    source = current_git_source_state(Path(__file__).resolve().parents[1])
    reader = load_qwen_reader(args.model_dir, device=device, dtype=torch.bfloat16 if device.type != "cpu" else torch.float32)
    train = [encode_reader_answer(reader, case) for case in data.train]
    development = [encode_reader_answer(reader, case) for case in data.development]
    ordered = sorted(train, key=lambda row: len(row.prompt_ids) + len(row.answer_ids))
    profile_examples = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    steps = 3 if args.profile_only else args.steps
    accumulation = 1 if args.profile_only else args.gradient_accumulation
    attach_reader_lora(reader, rank=args.rank, checkpointing=True)
    parameters = [value for value in reader.model.parameters() if value.requires_grad]
    run = args.artifact_root / f"{datetime.now(UTC):%Y%m%dT%H%M%S.%fZ}-{uuid.uuid4().hex[:8]}"
    run.mkdir(parents=True, exist_ok=False)
    manifest = json.dumps({
        "train": [{"case": asdict(case), "tokens": asdict(tokens)} for case, tokens in zip(data.train, train, strict=True)],
        "development": [{"case": asdict(case), "tokens": asdict(tokens)} for case, tokens in zip(data.development, development, strict=True)],
        "excluded_episodes": data.excluded_episodes,
        "gate_manifest_sha256": gate_hash,
        "source_sha256": source_hashes,
    }, indent=2, sort_keys=True) + "\n"
    (run / "data_manifest.json").write_text(manifest, encoding="utf-8")
    protocol = {
        "protocol": "visible_reader_lora_v1", "status": "profiling_only" if args.profile_only else "reader_adaptation_development",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "snapshot": snapshot, "source": source.to_dict(),
        "data_manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
        "packages": {item.metadata["Name"]: item.version for item in importlib.metadata.distributions()},
        "trainable_parameters": sum(value.numel() for value in parameters),
        "reader_parameters": sum(value.numel() for value in reader.model.parameters()),
        "objective": "mean_answer_token_ce_per_example",
        "sampling": "shortest_median_longest" if args.profile_only else "uniform_category_then_uniform_example",
        "effective_optimizer_steps": steps, "effective_gradient_accumulation": accumulation, "microbatch_size": 1,
        "profile_case_ids": [row.case_id for row in profile_examples] if args.profile_only else None,
        "checkpoint_selection": "lowest_internal_macro_category_answer_ce",
        "weight_decay": 0.01, "gradient_clip_norm": 1.0,
        "lora": {"rank": args.rank, "alpha": 2 * args.rank, "dropout": 0.0, "targets": ["q_proj", "v_proj"]},
        "checkpointing": "non_reentrant_training_mode", "cache": False,
        "profile_memory_measurement": "after_backward_and_step_boundaries_not_peak",
    }
    protocol_text = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    (run / "protocol.json").write_text(protocol_text, encoding="utf-8")
    adapter_protocol = {
        "protocol": "visible_reader_adapter_v1", "excluded_gate_manifest_sha256": gate_hash,
        "training_protocol_sha256": hashlib.sha256(protocol_text.encode()).hexdigest(),
        "training_manifest_sha256": protocol["data_manifest_sha256"],
    }
    print(f"artifacts: {run}", flush=True)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01)
    by_category = defaultdict(list)
    for case, tokens in zip(data.train, train, strict=True):
        by_category[case.category].append(tokens)
    categories = sorted(by_category)
    generator = random.Random(args.seed)
    best_loss, best_checkpoint = float("inf"), None
    started = time.perf_counter()
    history = []
    with (run / "metrics.jsonl").open("x", encoding="utf-8") as handle:
        for step in range(1, steps + 1):
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            losses, token_count, answer_count = [], 0, 0
            for _ in range(accumulation):
                example = profile_examples[step - 1] if args.profile_only else generator.choice(by_category[generator.choice(categories)])
                loss = reader_answer_loss(reader, example)
                (loss / accumulation).backward()
                losses.append(float(loss.detach()))
                token_count += len(example.prompt_ids) + len(example.answer_ids) - 1
                answer_count += len(example.answer_ids)
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            metric = {"step": step, "loss": sum(losses) / len(losses), "forward_tokens": token_count, "supervised_answer_tokens": answer_count,
                      "gradient_norm": float(norm), "step_seconds": time.perf_counter() - step_started}
            if device.type == "mps":
                metric.update(mps_allocated_bytes=torch.mps.current_allocated_memory(), mps_driver_bytes=torch.mps.driver_allocated_memory())
            if not args.profile_only and (step % args.evaluate_every == 0 or step == steps):
                reader.model.eval()
                values = defaultdict(list)
                with torch.no_grad():
                    for case, tokens in zip(data.development, development, strict=True):
                        values[case.category].append(float(reader_answer_loss(reader, tokens)))
                category_losses = {key: sum(value) / len(value) for key, value in values.items()}
                macro_loss = sum(category_losses.values()) / len(category_losses)
                if not math.isfinite(macro_loss):
                    raise ValueError("internal development loss is not finite")
                checkpoint = run / f"step_{step:06d}"
                reader.model.save_pretrained(checkpoint, save_embedding_layers=False)
                (checkpoint / "reader_adapter_protocol.json").write_text(json.dumps(adapter_protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                metric.update(development_category_ce=category_losses, development_macro_ce=macro_loss, checkpoint=str(checkpoint))
                if macro_loss < best_loss:
                    best_loss, best_checkpoint = macro_loss, str(checkpoint)
                reader.model.train()
            history.append(metric)
            handle.write(json.dumps(metric, sort_keys=True) + "\n")
            handle.flush()
            if args.profile_only or step % 10 == 0 or step == steps:
                print(json.dumps(metric), flush=True)
    result = {
        "status": protocol["status"], "steps": steps, "seconds": time.perf_counter() - started,
        "best_checkpoint": best_checkpoint, "best_internal_macro_ce": None if args.profile_only else best_loss,
        "gate_evaluated": False, "metrics": history,
    }
    (run / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "metrics"}), flush=True)


if __name__ == "__main__":
    main()
