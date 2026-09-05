#!/usr/bin/env python3
"""Fit four privileged training codes through an unchanged native readout."""

import argparse
import importlib.metadata
import json
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file, save_file

from tinymem.data.reader_gate import ReaderCase
from tinymem.research.study_runtime import (
    allocation_metrics, attach_execution, check_repository, execution_record,
    prepare_device, repository_path, sha256, synchronize, validate_execution,
)

from tinymem.evaluation.reader_gate import reader_exact_match
from tinymem.research.memory_prompt import encode_memory_example
from tinymem.research.native_memory_oracle import FixedProjectionMemoryOracle
from tinymem.research.prefix_reader import generate_prefix_answer, prefix_answer_loss
from tinymem.research.pretrained import load_qwen_reader, verify_qwen_snapshot
from tinymem.utils.experiment import current_git_source_state
from tinymem.utils.seed import seed_everything


WORLD_IDS = tuple(f"opaque-qa1-v1:train:{index:04d}" for index in range(4))
CONDITIONS = ("normal", "cyclic_donor", "drop", "zero", "full_context")


def digest(path):
    return sha256(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def oracle_from_states(states, projection):
    values = torch.tensor([row["values"] for row in states], dtype=torch.float32)
    valid = torch.tensor([row["valid"] for row in states])
    require(values.shape == (4, 1, 2, 8) and valid.shape == (4, 1, 2) and valid.dtype == torch.bool,
            "each saved state must have values [1,2,8] and boolean validity [1,2]")
    require(bool(valid.all()), "each selected state must have two valid slots")
    oracle = FixedProjectionMemoryOracle(values[:, 0], projection)
    require(all(oracle.state(index).nbytes == 66 for index in range(4)), "selected state storage is not 66 bytes")
    return oracle


def load_inputs(training_run, training_fit, study_path):
    study = json.loads(study_path.read_text())
    require(all(digest(Path(path)) == expected for path, expected in study["source_sha256"].items()),
            "frozen study sources changed")
    require(str(training_run) in study["runs"], "training run is outside the frozen study")
    training = json.loads((training_run / "protocol.json").read_text())
    result = json.loads((training_run / "results.json").read_text())
    require(training["study_protocol_sha256"] == digest(study_path), "training study hash mismatch")
    require(training["seed"] == 1337 and training["writer_kind"] == "query_pool" and training["steps"] == 1000,
            "use the predetermined final query-pool seed-1337 checkpoint")
    require(result["profile"] is False, "a profile is not a completed training run")
    checkpoint_path = training_run / "step_001000.safetensors"
    require(digest(checkpoint_path) == result["checkpoint_sha256"], "checkpoint hash mismatch")
    require(digest(training_run / "metrics.jsonl") == result["metrics_sha256"], "training metrics hash mismatch")
    inputs_path = training_run / "input_artifact_hashes.json"
    require(digest(inputs_path) == result["input_artifact_hashes_sha256"], "training input hash mismatch")
    inputs = json.loads(inputs_path.read_text())
    require(digest(training_run / "protocol.json") == inputs["protocol_sha256"], "training protocol hash mismatch")
    encoding_path = training_run / "native_encodings.json"
    require(digest(encoding_path) == inputs["native_encodings_sha256"], "native encoding hash mismatch")
    fit = json.loads((training_fit / "protocol.json").read_text())
    fit_result = json.loads((training_fit / "results.json").read_text())
    require(fit["protocol"] == "opaque_qa1_final_training_fit_v1", "unsupported training-fit protocol")
    require(fit["training_run"] == str(training_run), "training fit is from a different run")
    require(fit["study_protocol_sha256"] == digest(study_path), "training fit study mismatch")
    require(fit["training_protocol_sha256"] == digest(training_run / "protocol.json"), "training fit protocol mismatch")
    require(fit["training_results_sha256"] == digest(training_run / "results.json"), "training fit results mismatch")
    require(fit["checkpoint_sha256"] == digest(checkpoint_path), "training fit checkpoint mismatch")
    for filename in ("protocol", "states", "predictions"):
        suffix = "jsonl" if filename == "predictions" else "json"
        require(digest(training_fit / f"{filename}.{suffix}") == fit_result[f"{filename}_sha256"],
                f"training fit {filename} hash mismatch")
    data_path = repository_path(training["arguments"]["data"])
    require(digest(data_path / "protocol.json") == study["data_protocol_sha256"], "data protocol mismatch")
    require(digest(data_path / "train.json") == training["training_sha256"] == fit["training_sha256"],
            "training data mismatch")
    adapter = repository_path(training["adapter"])
    require(training["adapter_sha256"] == study["adapter_sha256"] == fit["adapter_sha256"], "reader adapter mismatch")
    require(all(digest(adapter / name) == expected for name, expected in training["adapter_sha256"].items()),
            "reader adapter file changed")
    rows = json.loads((data_path / "train.json").read_text())
    encodings = json.loads(encoding_path.read_text())["train"]
    states = json.loads((training_fit / "states.json").read_text())
    predictions = [json.loads(line) for line in (training_fit / "predictions.jsonl").read_text().splitlines()]
    require(len(rows) == len(states) == len(encodings) == 256 and len(predictions) == 2304,
            "training fit must cover all 256 worlds and 2304 queries")
    ids = [row["opaque"]["world_id"] for row in rows]
    require(len(set(ids)) == 256 and ids == [row["world_id"] for row in states] == [row["world_id"] for row in encodings],
            "training worlds, encodings, and states do not align")
    expected_cases = {case["case_id"]: case for row in rows for case in row["opaque"]["queries"]}
    require(len(expected_cases) == len(predictions) and Counter(row["case_id"] for row in predictions)
            == Counter({case_id: 1 for case_id in expected_cases}), "training-fit case coverage mismatch")
    for prediction in predictions:
        require(all(prediction[key] == value for key, value in expected_cases[prediction["case_id"]].items()),
                "training-fit prediction metadata mismatch")
    selected = [row["opaque"] for row in rows[:4]]
    require(tuple(world["world_id"] for world in selected) == WORLD_IDS, "selected training IDs changed")
    require(all(len(world["queries"]) == 9 and Counter(case["category"] for case in world["queries"])
                == {"opaque_qa1_known": 8, "opaque_qa1_missing": 1} for world in selected),
            "each selected world requires eight known and one absent query")
    entities = [set(world["entities"]) for world in selected]
    require(all(len(values) == 9 for values in entities) and len(set.union(*entities)) == 36,
            "selected worlds must have disjoint known and absent entities")
    weights = load_file(checkpoint_path)
    projection = weights["read_projection.weight"]
    require(projection.shape == (2048, 8) and projection.dtype == torch.float32, "read projection contract changed")
    oracle = oracle_from_states(states[:4], projection)
    selected_predictions = {row["case_id"]: row for row in predictions if row["world_id"] in WORLD_IDS}
    require(len(selected_predictions) == 36, "selected training predictions are incomplete")
    return study, training, fit, selected, encodings[:4], selected_predictions, oracle


def fit_step(reader, oracle, examples, optimizer):
    """Accumulate all query losses at one weight state, then update only codes."""
    require(len(examples) == len(oracle.codes) and all(group for group in examples), "oracle query groups do not align")
    require(not any(parameter.requires_grad for parameter in reader.model.parameters()), "reader must remain frozen")
    optimizer.zero_grad(set_to_none=True)
    means, counts = [], []
    for index, group in enumerate(examples):
        losses = []
        for before, after, answer in group:
            loss = prefix_answer_loss(reader, before, oracle(index), after, answer)
            require(bool(torch.isfinite(loss)), "oracle loss is nonfinite")
            (loss / (len(examples) * len(group))).backward()
            losses.append(float(loss.detach()))
        means.append(statistics.mean(losses))
        counts.append(len(group))
    require(oracle.codes.grad is not None, "oracle codes received no gradient")
    code_gradient_norms = oracle.codes.grad.flatten(1).norm(dim=1).tolist()
    norm = torch.nn.utils.clip_grad_norm_(oracle.parameters(), 1.0, error_if_nonfinite=True)
    require(all(parameter.grad is None for parameter in reader.model.parameters()), "reader received a gradient")
    require(oracle.projection.grad is None, "fixed projection received a gradient")
    optimizer.step()
    oracle.project_codes_()
    return {"answer_ce": statistics.mean(means), "world_answer_ce": means, "queries_per_world": counts,
            "gradient_norm": float(norm), "code_gradient_norms": code_gradient_norms,
            "codes_at_box_boundary": int((oracle.codes.detach().abs() == 1).sum())}


@torch.inference_mode()
def evaluate(reader, oracle, worlds, examples, phase, output):
    reader.model.eval()
    predictions = []
    device = reader.model.device
    with output.open("x") as handle:
        for index, (world, group) in enumerate(zip(worlds, examples, strict=True)):
            for condition in CONDITIONS:
                donor = (index + 1) % len(worlds)
                for raw, example in zip(world["queries"], group, strict=True):
                    case = ReaderCase(**raw)
                    if condition == "full_context":
                        memory = reader.model.get_input_embeddings()(torch.tensor(example.history_ids, device=device))
                    elif condition == "drop":
                        memory = oracle.projection.new_empty(0, oracle.projection.shape[0])
                    elif condition == "zero":
                        memory = torch.zeros_like(oracle(index))
                    else:
                        memory = oracle(donor if condition == "cyclic_donor" else index)
                    before, after, answer = (torch.tensor(ids, device=device) for ids in
                                             (example.before_ids, example.after_ids, example.answer_ids))
                    generated = generate_prefix_answer(reader, before, memory, after, max_new_tokens=8)
                    expected = "unknown" if condition in ("cyclic_donor", "drop", "zero") else case.answer
                    row = {"phase": phase, "condition": condition, "world_id": world["world_id"],
                           "donor_world_id": worlds[donor]["world_id"] if condition == "cyclic_donor" else None,
                           **asdict(case), **generated,
                           "recipient_answer_ce": float(prefix_answer_loss(reader, before, memory, after, answer)),
                           "recipient_exact_match": reader_exact_match(generated["prediction"], case.answer, case.category),
                           "expected_answer": expected,
                           "expected_exact_match": reader_exact_match(generated["prediction"], expected, case.category)}
                    predictions.append(row)
                    handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
            print(json.dumps({"phase": phase, "worlds_complete": index + 1}), flush=True)
    return predictions


def summarize(predictions):
    grouped = defaultdict(list)
    for row in predictions:
        grouped[f"{row['condition']}:{row['category']}"].append(row)
    return {key: {"count": len(rows), "recipient_correct": sum(row["recipient_exact_match"] for row in rows),
                  "expected_correct": sum(row["expected_exact_match"] for row in rows),
                  "mean_recipient_answer_ce": statistics.mean(row["recipient_answer_ce"] for row in rows),
                  "prediction_counts": dict(Counter(row["prediction"].strip().lower() for row in rows))}
            for key, rows in grouped.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=repository_path, required=True)
    parser.add_argument("--training-fit", type=repository_path, required=True)
    parser.add_argument("--study-protocol", type=repository_path, required=True)
    parser.add_argument("--output", type=repository_path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    args = parser.parse_args()
    check_repository()
    study, training, fit, worlds, saved, fit_predictions, oracle = load_inputs(
        args.training_run, args.training_fit, args.study_protocol)
    if args.dry_run:
        print(json.dumps({"claim": "preflight_only_no_inference", "world_ids": WORLD_IDS, "queries": 36,
                          "stored_state_bytes": 66, "trainable_scalars": oracle.codes.numel()}))
        return
    device = prepare_device(args.device)
    validate_execution(fit.get("execution", {}), execution_record(device))
    snapshot_path = Path("data/raw/pretrained/qwen3-1.7b")
    snapshot = verify_qwen_snapshot(snapshot_path)
    require(snapshot == training["snapshot"] == study["snapshot"] == fit["snapshot"], "reader snapshot mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    seed_everything(1337)
    sources = [Path(__file__), *(Path("src/tinymem") / name for name in (
        "research/native_memory_oracle.py", "research/memory_prompt.py", "research/prefix_reader.py",
        "research/pretrained.py", "memory/recurrent_slots.py", "evaluation/reader_gate.py",
        "evaluation/longmemeval.py", "utils/seed.py"))]
    protocol = {
        "protocol": "opaque_qa1_fixed_projection_oracle_v1", "claim": "privileged_training_fit_not_a_history_encoder",
        "source": current_git_source_state(Path.cwd()).to_dict(), "snapshot": snapshot,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "source_sha256": {str(path): digest(path) for path in sources},
        "study_protocol_sha256": digest(args.study_protocol), "training_protocol_sha256": digest(args.training_run / "protocol.json"),
        "checkpoint_sha256": digest(args.training_run / "step_001000.safetensors"),
        "training_fit_protocol_sha256": digest(args.training_fit / "protocol.json"),
        "training_fit_results_sha256": digest(args.training_fit / "results.json"),
        "training_fit_states_sha256": digest(args.training_fit / "states.json"),
        "training_fit_predictions_sha256": digest(args.training_fit / "predictions.jsonl"),
        "training_sha256": training["training_sha256"], "adapter_sha256": training["adapter_sha256"],
        "world_ids": WORLD_IDS, "selection": "first_four_training_worlds_all_nine_queries_no_replacement",
        "initialization": "exact_final_query_seed1337_codes_and_fixed_read_projection",
        "stored_state_bytes": 66, "slots": 2, "memory_width": 8, "trainable_scalars": 64,
        "reader_frozen": True, "projection_frozen": True, "trainable": ["codes"],
        "seed": 1337, "steps": 200, "learning_rate": 0.001, "weight_decay": 0.01, "clip_norm": 1.0,
        "batch": "all_four_worlds_nine_serial_queries_each_one_optimizer_update",
        "objective": "mean_world_mean_query_token_mean_answer_CE_including_EOT",
        "code_constraint": "closed_box_minus_one_to_one_projected_after_every_update_no_tanh",
        "recurrent_writes": 0, "checkpoint_selection": "final_only", "checkpoint_every": 50,
        "evaluation": {"phases": ["initial", "final"], "conditions": CONDITIONS, "greedy": True,
                       "max_new_tokens": 8, "batch_size": 1, "cache": False, "padding": False},
        "fit_gate": "final_normal_32_of_32_known_and_4_of_4_absent_no_replacement_if_reader_fails",
        "donor_expected_answer": "unknown_for_all_queries_because_all_nine_entities_are_disjoint",
        "development_confirmation_or_external_used": False,
        "packages": {item.metadata["Name"]: item.version for item in importlib.metadata.distributions()},
        "numerics": {"reader_dtype": "bfloat16", "code_dtype": "float32", "device": device.type,
                     "checkpointing": "non_reentrant", "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                     "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                     "float32_matmul_precision": torch.get_float32_matmul_precision()},
    }
    attach_execution(protocol, sources, device)
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    for path in sources:
        (args.output / path.name).write_bytes(path.read_bytes())
    reader = load_qwen_reader(snapshot_path, device=device, dtype=torch.bfloat16)
    reader.model = PeftModel.from_pretrained(reader.model, repository_path(training["adapter"]), is_trainable=False,
                                           local_files_only=True, use_safetensors=True)
    reader.model.eval().requires_grad_(False)
    require(all(module.p == 0 for module in reader.model.modules() if isinstance(module, torch.nn.Dropout)),
            "this fixed-reader diagnostic requires zero dropout")
    require(reader.model.config.attention_dropout == 0, "attention dropout must be zero")
    oracle = oracle.to(device)
    projection = oracle.projection.detach().clone()
    examples = [[encode_memory_example(reader, ReaderCase(**case)) for case in world["queries"]] for world in worlds]
    require(all(json.loads(json.dumps([asdict(example) for example in group])) == row["queries"]
                for group, row in zip(examples, saved, strict=True)), "selected native encodings changed")
    (args.output / "native_encodings.json").write_text(json.dumps([
        {"world_id": world["world_id"], "queries": [asdict(example) for example in group]}
        for world, group in zip(worlds, examples, strict=True)], indent=2, sort_keys=True) + "\n")
    save_file({name: value.detach().cpu().contiguous() for name, value in oracle.state_dict().items()}, args.output / "initial.safetensors")
    initial_started = time.perf_counter()
    initial = evaluate(reader, oracle, worlds, examples, "initial", args.output / "initial_predictions.jsonl")
    initial_seconds = time.perf_counter() - initial_started
    for row in initial:
        if row["condition"] == "normal":
            expected = fit_predictions[row["case_id"]]
            require(row["generated_ids"] == expected["generated_ids"] and row["prediction"] == expected["prediction"],
                    "initial oracle generation differs from the exact saved training state")
            require(row["recipient_answer_ce"] == expected["answer_ce"], "initial oracle loss differs from training fit")
    reader.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reader.model.train()
    tensor_examples = [[tuple(torch.tensor(ids, device=device) for ids in
                              (example.before_ids, example.after_ids, example.answer_ids)) for example in group] for group in examples]
    forward_tokens = sum(len(example.before_ids) + 2 + len(example.after_ids) + len(example.answer_ids) - 1
                         for group in examples for example in group)
    target_tokens = sum(len(example.answer_ids) for group in examples for example in group)
    require(forward_tokens == 3433 and target_tokens == 99, "declared training token budget changed")
    optimizer = torch.optim.AdamW(oracle.parameters(), lr=0.001, weight_decay=0.01)
    metrics = []
    started = time.perf_counter()
    with (args.output / "metrics.jsonl").open("x") as handle:
        for step in range(1, 201):
            synchronize(device)
            step_started = time.perf_counter()
            metric = fit_step(reader, oracle, tensor_examples, optimizer)
            synchronize(device)
            metric.update(step=step, seconds=time.perf_counter() - step_started,
                          forward_tokens=forward_tokens, supervised_tokens=target_tokens,
                          **allocation_metrics(device))
            require(torch.equal(oracle.projection, projection), "fixed read projection changed")
            metrics.append(metric)
            handle.write(json.dumps(metric, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            if step % 10 == 0:
                print(json.dumps(metric), flush=True)
            if step % 50 == 0:
                save_file({name: value.detach().cpu().contiguous() for name, value in oracle.state_dict().items()},
                          args.output / f"step_{step:06d}.safetensors")
                torch.save({"step": step, "optimizer": optimizer.state_dict(), "protocol_sha256": digest(args.output / "protocol.json")},
                           args.output / f"optimizer_{step:06d}.pt")
    training_seconds = time.perf_counter() - started
    reader.model.eval()
    reader.model.gradient_checkpointing_disable()
    final_started = time.perf_counter()
    final = evaluate(reader, oracle, worlds, examples, "final", args.output / "final_predictions.jsonl")
    final_seconds = time.perf_counter() - final_started
    for before, after in zip(initial, final, strict=True):
        if before["condition"] in ("drop", "zero", "full_context"):
            require(before["generated_ids"] == after["generated_ids"] and before["recipient_answer_ce"] == after["recipient_answer_ce"],
                    "a code-independent control changed while fitting codes")
    summary = summarize(final)
    exact_fit = all(summary[f"normal:{category}"]["recipient_correct"] == count
                    for category, count in (("opaque_qa1_known", 32), ("opaque_qa1_missing", 4)))
    full_context_exact = all(summary[f"full_context:{category}"]["recipient_correct"] == count
                            for category, count in (("opaque_qa1_known", 32), ("opaque_qa1_missing", 4)))
    result = {"claim": protocol["claim"], "exact_fitted_readout": exact_fit, "full_context_exact": full_context_exact,
              "initial": summarize(initial), "final": summary, "training_seconds": training_seconds,
              "initial_evaluation_seconds": initial_seconds, "final_evaluation_seconds": final_seconds,
              "median_step_seconds_after_first": statistics.median(row["seconds"] for row in metrics[1:]),
              "forward_tokens": sum(row["forward_tokens"] for row in metrics),
              "supervised_tokens": sum(row["supervised_tokens"] for row in metrics),
              "token_accounting": "logical_training_inputs_excluding_controls_and_checkpoint_recomputation",
              "protocol_sha256": digest(args.output / "protocol.json"), "metrics_sha256": digest(args.output / "metrics.jsonl"),
              "initial_checkpoint_sha256": digest(args.output / "initial.safetensors"),
              "final_checkpoint_sha256": digest(args.output / "step_000200.safetensors"),
              "native_encodings_sha256": digest(args.output / "native_encodings.json"),
              "initial_predictions_sha256": digest(args.output / "initial_predictions.jsonl"),
              "final_predictions_sha256": digest(args.output / "final_predictions.jsonl")}
    (args.output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
