#!/usr/bin/env python3
"""Fit six training-only memory codes; do not interpret this as compression."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch

from tinymem.data.reader_gate import ReaderCase
from tinymem.evaluation.reader_gate import reader_exact_match
from tinymem.research.memory_prompt import encode_memory_example
from tinymem.research.native_memory_oracle import NativeMemoryOracle, select_oracle_cases
from tinymem.research.prefix_reader import generate_prefix_answer, prefix_answer_loss
from tinymem.research.pretrained import load_qwen_reader, verify_qwen_snapshot
from tinymem.utils.device import select_device
from tinymem.utils.experiment import current_git_source_state
from tinymem.utils.seed import seed_everything


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    from peft import PeftModel
    from safetensors.torch import save_file

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptation-run", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("data/raw/pretrained/qwen3-1.7b"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    args = parser.parse_args()
    if args.steps <= 0 or args.seed < 0:
        raise ValueError("steps must be positive and seed nonnegative")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite run: {args.output}")
    manifest_path = args.adaptation_run / "data_manifest.json"
    source_hash = sha256(manifest_path)
    source_protocol = json.loads((args.adaptation_run / "protocol.json").read_text())
    if source_hash != source_protocol["data_manifest_sha256"]:
        raise ValueError("adaptation manifest hash mismatch")
    source_data = json.loads(manifest_path.read_text())
    adapter = Path(json.loads((args.adaptation_run / "results.json").read_text())["best_checkpoint"])
    if adapter.parent.resolve() != args.adaptation_run.resolve():
        raise ValueError("selected adapter must belong to the adaptation run")
    adapter_protocol = json.loads((adapter / "reader_adapter_protocol.json").read_text())
    if adapter_protocol["training_manifest_sha256"] != source_hash or adapter_protocol["training_protocol_sha256"] != sha256(args.adaptation_run / "protocol.json"):
        raise ValueError("adapter provenance does not match adaptation data and protocol")
    snapshot = verify_qwen_snapshot(args.model_dir)
    device = select_device(args.device)
    reader = load_qwen_reader(args.model_dir, device=device, dtype=torch.float32 if device.type == "cpu" else torch.bfloat16)
    reader.model = PeftModel.from_pretrained(reader.model, adapter, is_trainable=False, local_files_only=True, use_safetensors=True)
    reader.model.requires_grad_(False)
    reader.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reader.model.train()
    cases = [ReaderCase(**row["case"]) for row in source_data["train"] if row["case"]["category"] == "babi_qa1"]
    examples = [encode_memory_example(reader, case) for case in cases]
    indices = select_oracle_cases(cases, examples)
    cases, examples = [cases[index] for index in indices], [examples[index] for index in indices]
    unique_queries = {}
    for index, example in enumerate(examples):
        unique_queries.setdefault((example.before_ids, example.after_ids), index)
    query_representatives = list(unique_queries.values())
    unique_grid_count = len(query_representatives) * len(cases)
    minimum_grid_matches = math.ceil(5 * unique_grid_count / 6)
    seed_everything(args.seed)
    oracle = NativeMemoryOracle(len(cases), reader.model.config.hidden_size).to(device)
    inputs = [tuple(torch.tensor(ids, device=device) for ids in (example.before_ids, example.after_ids, example.answer_ids)) for example in examples]
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.dumps({"training_only": [{"case": asdict(case), "tokens": asdict(example)} for case, example in zip(cases, examples, strict=True)]}, indent=2, sort_keys=True) + "\n"
    (args.output / "data_manifest.json").write_text(manifest)
    root = Path(__file__).resolve().parents[1]
    implementation_paths = [Path(__file__).resolve(), *(root / "src/tinymem/research" / name for name in ("native_memory_oracle.py", "prefix_reader.py", "memory_prompt.py", "pretrained.py"))]
    protocol = {
        "claim": "privileged_training_only_readout_fit_not_query_blind_compression",
        "source": current_git_source_state(root).to_dict(), "snapshot": snapshot,
        "source_data_manifest_sha256": source_hash, "data_manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
        "source_sha256": {str(path.relative_to(root)): sha256(path) for path in implementation_paths},
        "adapter": str(adapter), "adapter_sha256": {name: sha256(adapter / name) for name in ("adapter_config.json", "adapter_model.safetensors", "reader_adapter_protocol.json")},
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "packages": {item.metadata["Name"]: item.version for item in importlib.metadata.distributions()},
        "platform": platform.platform(), "python": platform.python_version(),
        "selection": "one_shortest_native_training_history_per_qa1_answer_then_case_id",
        "steps": args.steps, "microbatch_size": 1, "gradient_accumulation": 6,
        "sampling": "all_six_cases_in_sorted_answer_order_each_step", "seed_set_after_reader_load": True,
        "learning_rate": 0.001, "weight_decay": 0.01, "clip_norm": 1.0,
        "objective": "mean_example_answer_ce_including_eot", "checkpoint_selection": "final_step_only",
        "reader_frozen": True, "reader_dtype": str(reader.model.dtype), "oracle_dtype": "float32",
        "slots": 2, "memory_width": 8, "code_parameterization": "tanh_of_independent_normal_std_0.1_parameters",
        "state_tensor_bytes_per_materialized_code": oracle.state(0).nbytes,
        "training_codebank_parameters": oracle.codes.numel(),
        "shared_projection_parameters": oracle.read_projection.weight.numel(),
        "checkpointing": "non_reentrant", "cache": False, "history_in_training_forward": False,
        "initialization_reuses_prior_memory_weights": False, "max_new_tokens": 8,
        "evaluation": "same_six_fitted_training_cases_all_36_code_query_pairs_plus_drop_zero_full_history",
        "unique_query_representatives": query_representatives,
        "strict_success": {"normal_correct": 6, "drop_unknown": 6, "minimum_unique_grid_donor_answer_matches": minimum_grid_matches},
        "limits": "oracle_codes_are_label_supervised_and_case_indexed_no_generalization_or_fixed_byte_frontier_claim",
        "memory_measurement": "tensor_payload_and_step_boundary_device_allocations_not_peak",
        "development_or_test_or_external_evaluated": False,
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    for path in implementation_paths:
        (args.output / path.name).write_bytes(path.read_bytes())
    save_file({key: value.detach().cpu().contiguous() for key, value in oracle.state_dict().items()}, args.output / "initial_oracle.safetensors")

    def check_cache(module, call_args, kwargs, output):
        if kwargs.get("use_cache") is not False or kwargs.get("past_key_values") is not None or output.past_key_values is not None:
            raise RuntimeError("oracle diagnostic must not use a KV cache")

    hook = reader.model.get_decoder().register_forward_hook(check_cache, with_kwargs=True)
    print(f"artifacts: {args.output}", flush=True)
    optimizer = torch.optim.AdamW(oracle.parameters(), lr=0.001, weight_decay=0.01)
    started = time.perf_counter()
    try:
        with (args.output / "metrics.jsonl").open("x") as log:
            for step in range(1, args.steps + 1):
                step_started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for index, (before, after, answer) in enumerate(inputs):
                    loss = prefix_answer_loss(reader, before, oracle(index), after, answer)
                    (loss / len(inputs)).backward()
                    losses.append(float(loss.detach()))
                norm = torch.nn.utils.clip_grad_norm_(oracle.parameters(), 1.0, error_if_nonfinite=True)
                if any(parameter.grad is not None for parameter in reader.model.parameters()):
                    raise RuntimeError("frozen reader accumulated a gradient")
                optimizer.step()
                if device.type == "mps":
                    torch.mps.synchronize()
                elif device.type == "cuda":
                    torch.cuda.synchronize()
                metric = {"step": step, "mean_answer_ce": sum(losses) / len(losses), "case_answer_ce": losses,
                          "gradient_norm": float(norm), "seconds": time.perf_counter() - step_started}
                if device.type == "mps":
                    metric.update(mps_allocated_bytes=torch.mps.current_allocated_memory(), mps_driver_bytes=torch.mps.driver_allocated_memory())
                log.write(json.dumps(metric, sort_keys=True) + "\n")
                log.flush()
                if step % 10 == 0 or step == args.steps:
                    print(json.dumps(metric), flush=True)
        training_seconds = time.perf_counter() - started
        checkpoint = args.output / f"step_{args.steps:06d}.safetensors"
        save_file({key: value.detach().cpu().contiguous() for key, value in oracle.state_dict().items()}, checkpoint)
        reader.model.eval()
        oracle.eval()
        predictions = []
        evaluation_started = time.perf_counter()
        with (args.output / "predictions.jsonl").open("x") as log, torch.no_grad():
            for index, (case, example, (before, after, answer)) in enumerate(zip(cases, examples, inputs, strict=True)):
                vectors = [("code", donor, oracle(donor)) for donor in range(len(cases))]
                vectors.extend([
                    ("drop", None, oracle.read_projection.weight.new_empty(0, reader.model.config.hidden_size)),
                    ("zero", None, oracle.read_projection(torch.zeros_like(oracle.state(index).values[0]))),
                    ("full_history", None, reader.model.get_input_embeddings()(torch.tensor(example.history_ids, device=device))),
                ])
                for condition, donor, memory in vectors:
                    loss = prefix_answer_loss(reader, before, memory, after, answer)
                    result = generate_prefix_answer(reader, before, memory, after, max_new_tokens=8)
                    donor_answer = cases[donor].answer if donor is not None else None
                    row = {"case_id": case.case_id, "query_index": index, "condition": condition, "code_index": donor,
                           "answer": case.answer, "donor_answer": donor_answer,
                           "correct": reader_exact_match(result["prediction"], case.answer, case.category),
                           "unknown": reader_exact_match(result["prediction"], "unknown", case.category),
                           "follows_donor": reader_exact_match(result["prediction"], donor_answer, case.category) if donor_answer is not None else None,
                           "answer_ce": float(loss), **result}
                    predictions.append(row)
                    log.write(json.dumps(row, sort_keys=True) + "\n")
                    log.flush()
        by_condition = defaultdict(list)
        for row in predictions:
            condition = "normal" if row["condition"] == "code" and row["code_index"] == row["query_index"] else row["condition"]
            by_condition[condition].append(row)
        summary = {name: {"count": len(rows), "correct": sum(row["correct"] for row in rows),
                          "unknown": sum(row["unknown"] for row in rows),
                          "mean_answer_ce": sum(row["answer_ce"] for row in rows) / len(rows)} for name, rows in by_condition.items()}
        grid_matches = sum(row["follows_donor"] for row in predictions if row["condition"] == "code")
        unique_grid_matches = sum(row["follows_donor"] for row in predictions if row["condition"] == "code" and row["query_index"] in query_representatives)
        result = {"claim": protocol["claim"], "by_condition": summary, "grid_donor_matches": grid_matches, "grid_count": 36,
                  "unique_grid_donor_matches": unique_grid_matches, "unique_grid_count": unique_grid_count,
                  "per_unique_query_donor_matches": {str(index): sum(row["follows_donor"] for row in predictions if row["condition"] == "code" and row["query_index"] == index) for index in query_representatives},
                  "strict_fit_success": summary["normal"]["correct"] == 6 and summary["drop"]["unknown"] == 6 and unique_grid_matches >= minimum_grid_matches,
                  "training_seconds": training_seconds, "evaluation_seconds": time.perf_counter() - evaluation_started,
                  "checkpoint_sha256": sha256(checkpoint), "steps": args.steps,
                  "logical_training_forward_tokens": args.steps * sum(len(example.before_ids) + 2 + len(example.after_ids) + len(example.answer_ids) - 1 for example in examples),
                  "supervised_answer_tokens": args.steps * sum(len(example.answer_ids) for example in examples),
                  "checkpoint_selected_on_development": False, "development_or_test_or_external_evaluated": False}
        (args.output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        hook.remove()


if __name__ == "__main__":
    main()
