"""Check transferred inputs, then exercise real Qwen writes, reads, and BPTT."""

import argparse
import json
import math
from pathlib import Path
import time

import torch

from tinymem.data.reader_gate import ReaderCase
from tinymem.research.memory_prompt import encode_history_chunks, encode_memory_example
from tinymem.research.prefix_reader import generate_prefix_answer, prefix_answer_loss
from tinymem.research.pretrained import load_qwen_reader, verify_qwen_snapshot
from tinymem.research.recurrent_memory import NativeRecurrentMemory
from tinymem.research.study_runtime import (
    allocation_metrics, check_repository, execution_record, prepare_device,
    repository_path, sha256, synchronize,
)


STUDY = Path("artifacts/predictions/opaque_memory_study_20260905/protocol.json")


def check_inputs(study_path: Path) -> dict:
    study = json.loads(study_path.read_text())
    for name, expected in study["source_sha256"].items():
        if sha256(name) != expected:
            raise ValueError(f"frozen source changed: {name}")
    if len(study["runs"]) != 6 or len(set(study["runs"])) != 6:
        raise ValueError("expected six distinct completed training runs")
    if sha256(study["vocabulary"]) != study["vocabulary_sha256"]:
        raise ValueError("training vocabulary changed")
    pairs = set()
    for name in study["runs"]:
        run = repository_path(name)
        training = json.loads((run / "protocol.json").read_text())
        result = json.loads((run / "results.json").read_text())
        inputs = json.loads((run / "input_artifact_hashes.json").read_text())
        if result["profile"] is not False or training["steps"] != 1000:
            raise ValueError(f"incomplete training: {run}")
        if training["study_protocol_sha256"] != sha256(study_path):
            raise ValueError(f"training study mismatch: {run}")
        expected_files = {
            "step_001000.safetensors": result["checkpoint_sha256"],
            "metrics.jsonl": result["metrics_sha256"],
            "input_artifact_hashes.json": result["input_artifact_hashes_sha256"],
            "protocol.json": inputs["protocol_sha256"],
            "native_encodings.json": inputs["native_encodings_sha256"],
            "schedule.json": inputs["schedule_sha256"],
            "initial_writer.safetensors": training["initial_writer_sha256"],
        }
        for filename, expected in expected_files.items():
            if sha256(run / filename) != expected:
                raise ValueError(f"training input changed: {run / filename}")
        metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        if [row["step"] for row in metrics] != list(range(1, 1001)):
            raise ValueError(f"incomplete metrics: {run}")
        pairs.add((training["writer_kind"], training["seed"]))
        data = repository_path(training["arguments"]["data"])
        gate = repository_path(training["arguments"]["reader_gate"])
        if sha256(data / "protocol.json") != study["data_protocol_sha256"]:
            raise ValueError("data protocol changed")
        if sha256(data / "train.json") != training["training_sha256"]:
            raise ValueError("training data changed")
        if sha256(data / "development.json") != training["development_sha256"]:
            raise ValueError("development data changed")
        if sha256(gate / "protocol.json") != study["reader_gate_protocol_sha256"] or sha256(gate / "results.json") != study["reader_gate_results_sha256"]:
            raise ValueError("reader qualification changed")
        if training["adapter_sha256"] != study["adapter_sha256"]:
            raise ValueError("training adapter mismatch")
        for filename, expected in training["adapter_sha256"].items():
            if sha256(repository_path(training["adapter"]) / filename) != expected:
                raise ValueError("reader adapter changed")
    if pairs != {(writer, seed) for writer in study["writers"] for seed in study["seeds"]}:
        raise ValueError("training method/seed coverage is incomplete")
    if verify_qwen_snapshot(Path("data/raw/pretrained/qwen3-1.7b")) != study["snapshot"]:
        raise ValueError("base model snapshot changed")
    return study


def backward_check(reader, writer, examples, chunks) -> dict:
    """Expose all write-state gradients while preserving the production loss."""
    device = reader.model.device
    state = writer.writer.empty(1)
    states = []
    for chunk in chunks:
        state = writer.write(reader, state, torch.tensor(chunk, device=device))
        state.values.retain_grad()
        states.append(state)
    memory = writer.memory_vectors(state)
    losses = [prefix_answer_loss(reader, torch.tensor(example.before_ids, device=device), memory,
                                 torch.tensor(example.after_ids, device=device),
                                 torch.tensor(example.answer_ids, device=device)) for example in examples]
    loss = torch.stack(losses).mean()
    loss.backward()
    gradients = [float(state.values.grad.norm()) for state in states]
    if not math.isfinite(float(loss.detach())) or not all(math.isfinite(value) and value > 0 for value in gradients):
        raise RuntimeError("answer loss did not reach every recurrent write with a finite nonzero gradient")
    if any(parameter.grad is not None for parameter in reader.model.parameters()):
        raise RuntimeError("the frozen reader received gradients")
    norm = torch.nn.utils.clip_grad_norm_(writer.parameters(), 1.0, error_if_nonfinite=True)
    return {"answer_ce": float(loss.detach()), "state_gradient_norms": gradients,
            "writer_gradient_norm": float(norm), "state_bytes": state.nbytes}


def main() -> None:
    from peft import PeftModel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-protocol", type=repository_path, default=STUDY)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    parser.add_argument("--output", type=repository_path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    check_repository()
    study = check_inputs(args.study_protocol)
    if args.check_only:
        print(json.dumps({"input_check": "passed", "training_runs": 6, "frozen_sources": len(study["source_sha256"]),
                          "model_loaded": False, "confirmation_answers_read": False}))
        return
    if args.output is None:
        parser.error("--output is required for a model smoke test")
    device = prepare_device(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    execution = execution_record(device)
    (args.output / "protocol.json").write_text(json.dumps({"execution": execution,
        "study_protocol_sha256": sha256(args.study_protocol), "scope": "first_training_world_only_not_a_scientific_result"}, indent=2) + "\n")
    first_run = repository_path(study["runs"][0])
    training = json.loads((first_run / "protocol.json").read_text())
    data = repository_path(training["arguments"]["data"])
    world = json.loads((data / "train.json").read_text())[0]["opaque"]
    reader = load_qwen_reader(Path("data/raw/pretrained/qwen3-1.7b"), device=device, dtype=torch.bfloat16)
    reader.model = PeftModel.from_pretrained(reader.model, repository_path(training["adapter"]),
        is_trainable=False, local_files_only=True, use_safetensors=True)
    reader.model.requires_grad_(False)
    reader.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reader.model.train()
    cases = [ReaderCase(**case) for case in world["queries"]]
    examples = [encode_memory_example(reader, case) for case in cases]
    chunks = encode_history_chunks(reader, cases[0], world["chunks"])
    results = {}
    for kind, width in (("query_pool", 64), ("mean_pool", 82)):
        writer = NativeRecurrentMemory(2048, memory_width=8, slots=2, segment_length=512,
                                      writer_kind=kind, aggregation_width=width).to(device)
        synchronize(device)
        started = time.perf_counter()
        result = backward_check(reader, writer, examples, chunks)
        torch.optim.AdamW(writer.parameters(), lr=0.001, weight_decay=0.01).step()
        synchronize(device)
        result.update(seconds=time.perf_counter() - started, **allocation_metrics(device))
        if result["state_bytes"] != 66 or len(result["state_gradient_norms"]) != 4:
            raise RuntimeError("memory state or recurrent write count changed")
        results[kind] = result
        del writer
    reader.model.eval()
    reader.model.gradient_checkpointing_disable()
    with torch.inference_mode():
        example = examples[0]
        memory = reader.model.get_input_embeddings()(torch.tensor(example.history_ids, device=device))
        generated = generate_prefix_answer(reader, torch.tensor(example.before_ids, device=device), memory,
                                           torch.tensor(example.after_ids, device=device), max_new_tokens=8)
    results["full_context_generation"] = generated
    results["execution"] = execution
    (args.output / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
