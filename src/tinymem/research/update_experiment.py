"""Fresh-output experiment execution around the paired-update core.

No implicit resume, checkpoint selection, or reader adaptation. Partial outputs
remain evidence of interruption and cannot satisfy completion checks.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import torch
from safetensors.torch import load_file, save_file

from tinymem.research.study_runtime import (
    REPOSITORY, allocation_metrics, execution_record, repository_path, synchronize,
)
from tinymem.research.update_protocol import (
    DATA_SOURCES, DevelopmentData, _load_split, _selection_metadata,
    file_sha256, read_json, verify_hashes,
)
from tinymem.research.update_runner import (
    BASELINES, CONTROLS, NEURAL_METHODS, competence_gate, encode_update,
    evaluate_update_episode, new_update_writer, retention_policies,
    train_update_step, training_schedule, training_vocabulary,
)


SOURCES = tuple(dict.fromkeys((*DATA_SOURCES,
    "src/tinymem/research/update_runner.py", "src/tinymem/research/update_protocol.py",
    "src/tinymem/research/update_experiment.py", "scripts/run_memory_updates.py", "src/tinymem/utils/seed.py",
    "src/tinymem/evaluation/memory_updates.py", "src/tinymem/evaluation/longmemeval.py",
    "src/tinymem/evaluation/reader_gate.py", "src/tinymem/research/memory_prompt.py",
    "src/tinymem/research/prefix_reader.py", "src/tinymem/research/recurrent_memory.py",
    *(f"src/tinymem/memory/{name}.py" for name in ("query_pool_slots", "mean_pool_slots", "recurrent_slots",
      "fingerprint_facts", "latest_fact_tokens", "packed_tokens", "storage", "template_facts", "vocabulary_tokens")),
)))
OBJECTIVE = "mean_four_states_of_mean_ten_queries_of_token_mean_answer_CE_including_EOT"
OPTIMIZER = {"name": "AdamW", "lr": 0.001, "weight_decay": 0.01, "clip_norm": 1.0}
SEEDS = (1337, 2027, 4099)


def write_json(path: Path, value) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def relative(path: Path) -> str:
    return str(repository_path(path, root=REPOSITORY))


def execution(reader) -> dict:
    record = execution_record(reader.model.device)
    # Keep actual tiny-test FP32 separate from production BF16 evidence.
    record["runtime"]["reader_dtype"] = str(reader.model.get_input_embeddings().weight.dtype).removeprefix("torch.")
    return {"version": "memory_update_execution_v1", "runtime": record["runtime"],
            "source_sha256": {name: file_sha256(REPOSITORY / name) for name in SOURCES}}


def provenance(data: DevelopmentData, identity: dict, reader) -> dict:
    synthetic = data.protocol["design"]["status"] == "synthetic_fixture_not_scientific_data"
    if not synthetic and data.protocol["design"]["status"] != "data_and_measurement_design_not_a_training_launch_protocol":
        raise ValueError("unsupported study design status")
    if not synthetic and reader.model.get_input_embeddings().weight.dtype != torch.bfloat16:
        raise ValueError("production study requires the qualified BF16 reader")
    return {"evidence_kind": "synthetic_fixture" if synthetic else "model_experiment",
            "data": relative(data.directory), "data_protocol_sha256": data.protocol_sha256,
            "reader": identity, "execution": execution(reader)}


def seal(directory: Path, files: list[str], **metadata) -> dict:
    """Write completion last; a crash before this never becomes a completed run."""
    result = {**metadata, "files": {name: file_sha256(directory / name) for name in files}}
    write_json(directory / "complete.json", result)
    return result


def verify_complete(directory: Path, required: set[str]) -> dict:
    result = read_json(directory / "complete.json")
    if set(result["files"]) != required:
        raise ValueError("completion artifact coverage changed")
    verify_hashes(result["files"], directory)
    return result


def evaluate_rows(reader, encoded, method, *, writer=None, policy=None, output: Path) -> list[dict]:
    reader.model.eval()
    reader.model.gradient_checkpointing_disable()
    if writer is not None:
        writer.eval()
    records = []
    with output.open("x") as handle:
        for row in encoded:
            result = evaluate_update_episode(reader, row, method, writer=writer, policy=policy)
            handle.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            records.append(result)
    return records


def read_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def qualify(reader, data: DevelopmentData, identity: dict, output: Path) -> dict:
    base = provenance(data, identity, reader)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", {"kind": "reader_qualification", **base})
    encoded = [encode_update(reader, row) for row in data.development]
    records = evaluate_rows(reader, encoded, "full_context", output=output / "predictions.jsonl")
    gate = competence_gate(records, data.development, reader_gate=True)
    write_json(output / "gate.json", gate)
    seal(output, ["protocol.json", "predictions.jsonl", "gate.json"], kind="reader_qualification")
    return gate


def verify_qualification(directory: Path, data: DevelopmentData, expected: dict) -> dict:
    complete = verify_complete(directory, {"protocol.json", "predictions.jsonl", "gate.json"})
    if complete["kind"] != "reader_qualification" or read_json(directory / "protocol.json") != {"kind": "reader_qualification", **expected}:
        raise ValueError("qualification provenance differs from this launch")
    gate = competence_gate(read_rows(directory / "predictions.jsonl"), data.development, reader_gate=True)
    if read_json(directory / "gate.json") != gate or gate["passed"] is not True:
        raise ValueError("new-task full-context reader qualification failed")
    return gate


def freeze_launch(reader, data: DevelopmentData, identity: dict, qualification: Path, output: Path, *, steps: int) -> dict:
    """Freeze exact schedules and encodings before any new training checkpoint."""
    if type(steps) is not int or steps <= 0:
        raise ValueError("positive integer training steps required")
    base = provenance(data, identity, reader)
    verify_qualification(qualification, data, base)
    if (data.protocol["design"]["optimization_seeds"] != list(SEEDS)
            or data.protocol["design"]["neural_methods"] != list(NEURAL_METHODS)):
        raise ValueError("unsupported seeds or writers")
    encoded = [encode_update(reader, row) for row in data.train]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "training_encodings.json", [row.token_record() for row in encoded])
    write_json(output / "vocabulary.json", training_vocabulary(encoded))
    write_json(output / "schedules.json", {str(seed): training_schedule(len(encoded), steps, seed) for seed in SEEDS})
    launch = {"kind": "memory_update_launch_v1", **base, "steps": steps,
              "objective": OBJECTIVE, "optimizer": OPTIMIZER, "checkpoint_every": 100,
              "checkpoint_selection": "final_only", "gradient_checkpointing": True,
              "persistent_bytes": 66, "max_new_tokens": 8,
              "qualification": relative(qualification), "qualification_complete_sha256": file_sha256(qualification / "complete.json"),
              "runs": [{"method": method, "seed": seed, "directory": f"runs/{method}_seed_{seed}"}
                       for seed in SEEDS for method in NEURAL_METHODS],
              "claim": "descriptive_updates_report_competence_limited_if_before_gate_fails",
              "shared_costs": {"reader_parameters": sum(p.numel() for p in reader.model.parameters()),
                  "reader_parameter_bytes": sum(p.numel() * p.element_size() for p in reader.model.parameters()),
                  "training_vocabulary_entries": len(training_vocabulary(encoded)),
                  "training_vocabulary_serialized_bytes": 4 * len(training_vocabulary(encoded)),
                  "scope": "shared_model_and_dictionary_costs_not_per_stream_state"},
              "confirmation_opened": False}
    write_json(output / "protocol.json", launch)
    seal(output, ["protocol.json", "training_encodings.json", "vocabulary.json", "schedules.json"], kind="launch")
    return launch


def verify_launch(directory: Path, data: DevelopmentData, expected: dict | None = None) -> dict:
    complete = verify_complete(directory, {"protocol.json", "training_encodings.json", "vocabulary.json", "schedules.json"})
    launch = read_json(directory / "protocol.json")
    if complete["kind"] != "launch" or launch["kind"] != "memory_update_launch_v1":
        raise ValueError("expected a frozen update launch")
    base = {key: launch[key] for key in ("evidence_kind", "data", "data_protocol_sha256", "reader", "execution")}
    if expected is not None and base != expected:
        raise ValueError("launch differs from current data, reader, sources, or runtime")
    if launch["data_protocol_sha256"] != data.protocol_sha256 or launch["data"] != relative(data.directory):
        raise ValueError("launch data identity changed")
    if launch["execution"]["source_sha256"] != {name: file_sha256(REPOSITORY / name) for name in SOURCES}:
        raise ValueError("launch execution sources changed")
    qualification = REPOSITORY / repository_path(launch["qualification"], root=REPOSITORY)
    if file_sha256(qualification / "complete.json") != launch["qualification_complete_sha256"]:
        raise ValueError("qualification completion changed")
    verify_qualification(qualification, data, base)
    if (launch["objective"] != OBJECTIVE or launch["optimizer"] != OPTIMIZER or launch["persistent_bytes"] != 66
            or launch["checkpoint_every"] != 100 or launch["checkpoint_selection"] != "final_only"
            or launch["gradient_checkpointing"] is not True or launch["max_new_tokens"] != 8):
        raise ValueError("unsupported training or read contract")
    expected_runs = [{"method": method, "seed": seed, "directory": f"runs/{method}_seed_{seed}"}
                     for seed in SEEDS for method in NEURAL_METHODS]
    if launch["runs"] != expected_runs:
        raise ValueError("launch must declare exactly six fresh runs")
    if read_json(directory / "schedules.json") != {str(seed): training_schedule(len(data.train), launch["steps"], seed) for seed in SEEDS}:
        raise ValueError("frozen schedule differs from declared training")
    return launch


def _save_writer(path, writer):
    save_file({key: value.detach().cpu().contiguous() for key, value in writer.state_dict().items()}, path)


def train(reader, data: DevelopmentData, identity: dict, launch_dir: Path, method: str, seed: int) -> dict:
    launch = verify_launch(launch_dir, data, provenance(data, identity, reader))
    if method not in NEURAL_METHODS or seed not in SEEDS:
        raise ValueError("undeclared method/seed")
    encoded = [encode_update(reader, row) for row in data.train]
    if (json.loads(json.dumps([row.token_record() for row in encoded])) != read_json(launch_dir / "training_encodings.json")
            or training_vocabulary(encoded) != read_json(launch_dir / "vocabulary.json")):
        raise ValueError("training tokenization changed")
    output = launch_dir / f"runs/{method}_seed_{seed}"
    output.mkdir(parents=True, exist_ok=False)
    writer = new_update_writer(reader, method, seed)
    optimizer = torch.optim.AdamW(writer.parameters(), lr=OPTIMIZER["lr"], weight_decay=OPTIMIZER["weight_decay"])
    write_json(output / "protocol.json", {"kind": "training", "launch_complete_sha256": file_sha256(launch_dir / "complete.json"),
               "method": method, "seed": seed, "steps": launch["steps"], "evidence_kind": launch["evidence_kind"]})
    _save_writer(output / "initial.safetensors", writer)
    reader.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reader.model.train()
    writer.train()
    schedule = read_json(launch_dir / "schedules.json")[str(seed)]
    files = ["protocol.json", "initial.safetensors", "metrics.jsonl"]
    started = time.perf_counter()
    with (output / "metrics.jsonl").open("x") as handle:
        for step, index in enumerate(schedule, 1):
            synchronize(reader.model.device)
            start = time.perf_counter()
            metric = train_update_step(reader, writer, encoded[index], optimizer, trace_gradients=step == 1)
            synchronize(reader.model.device)
            metric.update(step=step, episode_id=encoded[index].episode.episode_id, seconds=time.perf_counter() - start,
                          **allocation_metrics(reader.model.device))
            handle.write(json.dumps(metric, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            if step % 10 == 0 or step == 1:
                print(json.dumps(metric), flush=True)
            if step % 100 == 0 or step == launch["steps"]:
                checkpoint, opt = f"step_{step:06d}.safetensors", f"optimizer_{step:06d}.pt"
                _save_writer(output / checkpoint, writer)
                torch.save({"step": step, "optimizer": optimizer.state_dict(),
                            "launch_complete_sha256": file_sha256(launch_dir / "complete.json")}, output / opt)
                files.extend((checkpoint, opt))
    seconds = time.perf_counter() - started
    development = [encode_update(reader, row) for row in data.development]
    records = evaluate_rows(reader, development, method, writer=writer, output=output / "development.jsonl")
    gate = competence_gate(records, data.development, reader_gate=False)
    write_json(output / "gate.json", gate)
    files.extend(("development.jsonl", "gate.json"))
    return seal(output, files, kind="training", training_seconds=seconds, completed_steps=launch["steps"],
                evidence_kind=launch["evidence_kind"], final_checkpoint=f"step_{launch['steps']:06d}.safetensors")


def profile(reader, data: DevelopmentData, identity: dict, output: Path, method: str) -> dict:
    """Ten training-only steps, never a qualification or a reusable checkpoint."""
    if method not in NEURAL_METHODS:
        raise ValueError("profiling requires a learned method")
    base = provenance(data, identity, reader)
    output.mkdir(parents=True, exist_ok=False)
    encoded = [encode_update(reader, row) for row in data.train]
    writer = new_update_writer(reader, method, SEEDS[0])
    optimizer = torch.optim.AdamW(writer.parameters(), lr=OPTIMIZER["lr"], weight_decay=OPTIMIZER["weight_decay"])
    reader.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reader.model.train()
    write_json(output / "protocol.json", {"kind": "compute_profile_not_training", **base, "method": method,
               "steps": 10, "seed": SEEDS[0], "objective": OBJECTIVE, "optimizer": OPTIMIZER,
               "checkpoint_reuse": False, "confirmation_opened": False})
    schedule = training_schedule(len(encoded), 10, SEEDS[0])
    write_json(output / "schedule.json", schedule)
    with (output / "metrics.jsonl").open("x") as handle:
        for step, index in enumerate(schedule, 1):
            synchronize(reader.model.device)
            started = time.perf_counter()
            metric = train_update_step(reader, writer, encoded[index], optimizer, trace_gradients=True)
            synchronize(reader.model.device)
            metric.update(step=step, episode_id=encoded[index].episode.episode_id,
                          seconds=time.perf_counter() - started, **allocation_metrics(reader.model.device))
            handle.write(json.dumps(metric, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
    reader.model.eval()
    reader.model.gradient_checkpointing_disable()
    return seal(output, ["protocol.json", "schedule.json", "metrics.jsonl"], kind="compute_profile_not_training",
                checkpoint_reuse=False, token_accounting="logical_forward_inputs_excluding_checkpoint_recomputation")


def verify_training(launch_dir: Path, launch: dict, data: DevelopmentData, method: str, seed: int) -> Path:
    if method not in NEURAL_METHODS or seed not in SEEDS:
        raise ValueError("undeclared method/seed")
    directory = launch_dir / f"runs/{method}_seed_{seed}"
    steps = launch["steps"]
    checkpoints = sorted(set(range(100, steps + 1, 100)) | {steps})
    files = {"protocol.json", "initial.safetensors", "metrics.jsonl", "development.jsonl", "gate.json",
             *(f"step_{step:06d}.safetensors" for step in checkpoints), *(f"optimizer_{step:06d}.pt" for step in checkpoints)}
    complete = verify_complete(directory, files)
    expected = {"kind": "training", "launch_complete_sha256": file_sha256(launch_dir / "complete.json"),
                "method": method, "seed": seed, "steps": steps, "evidence_kind": launch["evidence_kind"]}
    if (read_json(directory / "protocol.json") != expected or complete["kind"] != "training"
            or complete["completed_steps"] != steps or complete["evidence_kind"] != launch["evidence_kind"]
            or complete["final_checkpoint"] != f"step_{steps:06d}.safetensors"):
        raise ValueError("training completion differs from frozen launch")
    metrics = read_rows(directory / "metrics.jsonl")
    schedule = read_json(launch_dir / "schedules.json")[str(seed)]
    if len(metrics) != steps or any(row["step"] != i + 1 or row["episode_id"] != data.train[index].episode_id
                                  for i, (row, index) in enumerate(zip(metrics, schedule, strict=True))):
        raise ValueError("training log does not cover the declared schedule")
    records = read_rows(directory / "development.jsonl")
    if any(row["method"] != method for row in records):
        raise ValueError("development predictions belong to another method")
    if read_json(directory / "gate.json") != competence_gate(records, data.development, reader_gate=False):
        raise ValueError("stored memory competence differs from authoritative scores")
    return directory / complete["final_checkpoint"]


def confirmation_episodes(launch_dir: Path, data: DevelopmentData, launch: dict):
    """Only open confirmation after every declared training run is complete."""
    verify_launch(launch_dir, data)
    if launch != read_json(launch_dir / "protocol.json"):
        raise ValueError("unexpected launch supplied to confirmation")
    for run in launch["runs"]:
        verify_training(launch_dir, launch, data, run["method"], run["seed"])
    selection = _selection_metadata(data.directory, data.protocol, REPOSITORY)
    return _load_split(data.directory, data.protocol, "confirmation", selection)


def evaluate(reader, data: DevelopmentData, identity: dict, launch_dir: Path, method: str, *, seed: int | None = None, split: str = "confirmation") -> dict:
    launch = verify_launch(launch_dir, data, provenance(data, identity, reader))
    if method not in (*NEURAL_METHODS, *BASELINES, *CONTROLS) or split not in ("development", "confirmation"):
        raise ValueError("unsupported evaluation method or split")
    if (method in NEURAL_METHODS and seed not in SEEDS) or (method not in NEURAL_METHODS and seed is not None):
        raise ValueError("only a learned method takes a declared training seed")
    writer, policy = None, None
    if method in NEURAL_METHODS:
        checkpoint = verify_training(launch_dir, launch, data, method, seed)
        writer = new_update_writer(reader, method, seed)
        writer.load_state_dict(load_file(checkpoint, device=str(reader.model.device)), strict=True)
    else:
        checkpoint = None
        policy = retention_policies(reader, read_json(launch_dir / "vocabulary.json")).get(method)
    episodes = confirmation_episodes(launch_dir, data, launch) if split == "confirmation" else data.development
    encoded = [encode_update(reader, row) for row in episodes]
    output = launch_dir / "evaluations" / split / (method if seed is None else f"{method}_seed_{seed}")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", {"kind": "evaluation", "method": method, "seed": seed, "split": split,
               "launch_complete_sha256": file_sha256(launch_dir / "complete.json"), "evidence_kind": launch["evidence_kind"],
               "checkpoint_sha256": None if checkpoint is None else file_sha256(checkpoint),
               "shared_writer_parameters": 0 if writer is None else sum(p.numel() for p in writer.parameters()),
               "shared_writer_parameter_bytes": 0 if writer is None else sum(p.numel() * p.element_size() for p in writer.parameters()),
               "shared_dictionary_serialized_bytes": getattr(policy, "dictionary_serialized_bytes", None),
               "shared_dictionary_scope": "training_vocabulary_or_fixed_grammar_not_per_stream; fingerprint_fixed_code_not_serialized_here"})
    records = evaluate_rows(reader, encoded, method, writer=writer, policy=policy, output=output / "predictions.jsonl")
    write_json(output / "encodings.json", [row.token_record() for row in encoded])
    return seal(output, ["protocol.json", "predictions.jsonl", "encodings.json"], kind="evaluation", histories=len(records))
