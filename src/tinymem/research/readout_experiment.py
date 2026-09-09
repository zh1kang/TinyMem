"""One explicit readout run: no implicit resume or checkpoint selection."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from tinymem.data.opaque_qa1 import ROOMS
from tinymem.memory.readout_interface import OneShotEncoder, ReadoutBridge, ReadoutKind, STATE_BYTES
from tinymem.research.pretrained import PretrainedReader
from tinymem.research.readout_checkpoint import load_checkpoint, save_checkpoint
from tinymem.research.readout_evaluation import evaluate_full_text, evaluate_readout
from tinymem.research.readout_runner import EncodedBefore, train_readout_step
from tinymem.research.study_runtime import synchronize as _synchronize
from tinymem.research.update_protocol import file_sha256, read_json
from tinymem.research.update_runner import training_schedule


SPLITS = ("train", "development")
FILES = ("protocol.json", "encodings.json", "initial.safetensors", "final.safetensors",
         "metrics.jsonl", "predictions.jsonl")
# Keep the executed scientific path bound without altering the closed study.
SOURCE_FILES = (
    "memory/readout_interface.py", "memory/recurrent_slots.py",
    "research/readout_interface.py", "research/readout_controls.py", "research/readout_read.py",
    "research/readout_runner.py", "research/readout_evaluation.py", "research/readout_checkpoint.py",
    "research/readout_experiment.py", "research/prefix_reader.py", "research/memory_prompt.py",
    "research/update_encoding.py", "research/update_runner.py", "research/update_protocol.py",
    "research/pretrained.py", "research/study_runtime.py", "data/memory_updates.py",
    "data/opaque_qa1.py", "data/reader_gate.py", "data/symbolic_world.py",
    "evaluation/reader_gate.py", "evaluation/longmemeval.py",
)


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {name: file_sha256(root / name) for name in SOURCE_FILES}


def _reader_hash(reader: PretrainedReader) -> str:
    """Stream parameters and buffers, including shape and dtype, into one digest."""
    digest = hashlib.sha256()
    for name, value in sorted(reader.model.state_dict().items()):
        header = json.dumps([name, str(value.dtype), list(value.shape)]).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _write_json(path: Path, value) -> None:
    with path.open("x") as handle:
        handle.write(_json(value) + "\n")


def _validate_inputs(reader, splits, *, kind, seed, steps, learning_rate, weight_decay,
                     max_new_tokens, input_identity, required_splits=SPLITS) -> None:
    if kind not in ("affine", "gelu"):
        raise ValueError("kind must be affine or gelu")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if any(type(n) is not int or n <= 0 for n in (steps, max_new_tokens)):
        raise ValueError("steps and max_new_tokens must be positive integers")
    for name, value in (("learning_rate", learning_rate), ("weight_decay", weight_decay)):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if learning_rate == 0:
        raise ValueError("learning_rate must be positive")
    if not isinstance(input_identity, dict) or not input_identity:
        raise ValueError("nonempty input identity is required")
    _json(input_identity)
    if any(m.training for m in reader.model.modules()):
        raise ValueError("reader must be in evaluation mode")
    if any(p.requires_grad or p.grad is not None for p in reader.model.parameters()):
        raise ValueError("reader must be frozen with no parameter gradients")
    if not isinstance(splits, Mapping) or set(splits) != set(required_splits):
        raise ValueError(f"required splits: {required_splits}")
    histories, cases = set(), set()
    source_seen = {key: set() for key in ("source_group_ids", "source_case_ids", "source_context_sha256")}
    vocab, context = reader.model.config.vocab_size, reader.model.config.max_position_embeddings
    for split in required_splits:
        rows = splits[split]
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or len(rows) < 2:
            raise ValueError("each split requires at least two histories for shuffling")
        for row in rows:
            if not isinstance(row, EncodedBefore):
                raise ValueError("expected encoded before histories")
            if not isinstance(row.history_id, str) or not row.history_id.strip() or row.history_id in histories:
                raise ValueError("history identities must be unique and nonempty")
            histories.add(row.history_id)
            for key, seen in source_seen.items():
                values = getattr(row, key)
                if (not isinstance(values, tuple) or len(values) != 2
                        or any(not isinstance(v, str) or not v.strip() for v in values)):
                    raise ValueError("source overlap or invalid source identity")
                if key == "source_case_ids":
                    values = tuple(v.rsplit(":question-", 1)[0] for v in values)
                if any(not v.strip() for v in values) or len(set(values)) != 2 or seen.intersection(values):
                    raise ValueError("source overlap or invalid source identity")
                if key == "source_context_sha256" and any(
                    len(v) != 64 or any(c not in "0123456789abcdef" for c in v) for v in values
                ):
                    raise ValueError("source context identity must be SHA-256")
                seen.update(values)
            if len(row.queries) != 10 or sum(q.category == "update_known" for q in row.queries) != 8:
                raise ValueError("each history requires eight known and two missing queries")
            for query in row.queries:
                if query.category not in ("update_known", "update_missing"):
                    raise ValueError("invalid query category")
                allowed = ROOMS if query.category == "update_known" else ("unknown",)
                if query.answer not in allowed:
                    raise ValueError("answer disagrees with category")
                if not isinstance(query.case_id, str) or not query.case_id.strip() or query.case_id in cases:
                    raise ValueError("query identities must be unique and nonempty")
                cases.add(query.case_id)
                for ids in (row.before_ids, row.history_ids, query.after_ids, query.answer_ids):
                    if not isinstance(ids, tuple) or not ids or any(type(t) is not int or not 0 <= t < vocab for t in ids):
                        raise ValueError("invalid native token fragment")
                if len(query.answer_ids) < 2:
                    raise ValueError("answer and stopping tokens are required")
                if len(row.before_ids) + max(2, len(row.history_ids)) + len(query.after_ids) + max(max_new_tokens, len(query.answer_ids) - 1) > context:
                    raise ValueError("reader context exceeded; no truncation or filtering")


def run_arm(
    reader: PretrainedReader, splits: Mapping[str, Sequence[EncodedBefore]], output: Path, *,
    kind: ReadoutKind, seed: int, steps: int, learning_rate: float, weight_decay: float,
    max_new_tokens: int, input_identity: dict,
) -> dict:
    """Run one arm with caller-declared input provenance and measured reader identity.

    Production callers must use the verified development loader before encoding.
    No malformed input creates an output directory. A failed run stays unsealed;
    recovery requires a new output directory, not reuse of partial results.
    """
    _validate_inputs(reader, splits, kind=kind, seed=seed, steps=steps, learning_rate=learning_rate,
                     weight_decay=weight_decay, max_new_tokens=max_new_tokens, input_identity=input_identity)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    sources = _source_hashes()
    reader_hash = _reader_hash(reader)
    width = reader.model.get_input_embeddings().embedding_dim
    # CPU initialization gives paired arms the same tensors before device transfer.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        encoder, bridge = OneShotEncoder(width), ReadoutBridge(width, kind)
    encoder.to(reader.model.device)
    bridge.to(reader.model.device)
    schedule = training_schedule(len(splits["train"]), steps, seed)
    protocol = {
        "kind": "readout_arm_protocol_v1", "arm": kind, "seed": seed, "steps": steps,
        "schedule": schedule, "splits": list(SPLITS), "checkpoint_selection": "final_only",
        "optimizer": {"kind": "AdamW", "lr": learning_rate, "weight_decay": weight_decay,
                      "betas": [0.9, 0.999], "eps": 1e-8, "clip_norm": 1.0},
        "max_new_tokens": max_new_tokens, "persistent_bytes": STATE_BYTES,
        "input_identity": input_identity, "input_identity_verification": "caller_declared",
        "reader_parameters_sha256": reader_hash, "reader_config": reader.model.config.to_dict(),
        "reader_width": width, "device": str(reader.model.device), "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "device_name": (torch.cuda.get_device_name(reader.model.device)
                        if reader.model.device.type == "cuda" else str(reader.model.device)),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "reader_dtype": str(reader.model.get_input_embeddings().weight.dtype),
        "source_sha256": sources,
        "shared_parameters": {"encoder": sum(p.numel() for p in encoder.parameters()),
                              "bridge": sum(p.numel() for p in bridge.parameters())},
    }
    # Serialize before claiming the output directory, including caller metadata.
    _json(protocol)
    encodings = {split: [asdict(row) for row in splits[split]] for split in SPLITS}
    _json(encodings)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "protocol.json", protocol)
    _write_json(output / "encodings.json", encodings)
    save_checkpoint(output / "initial.safetensors", encoder, bridge)
    prediction_count = 0
    with (output / "predictions.jsonl").open("x") as predictions:
        def record(records, split, phase):
            nonlocal prediction_count
            for row in records:
                predictions.write(_json({**row, "split": split, "phase": phase}) + "\n")
                prediction_count += 1
            predictions.flush()

        for split in SPLITS:
            record(evaluate_readout(reader, encoder, bridge, splits[split], max_new_tokens=max_new_tokens), split, "initial")
            record(evaluate_full_text(reader, splits[split], max_new_tokens=max_new_tokens), split, "reference")
        optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(bridge.parameters()),
                                      lr=learning_rate, weight_decay=weight_decay)
        device = reader.model.device
        _synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with (output / "metrics.jsonl").open("x") as metrics:
            for step, index in enumerate(schedule, 1):
                _synchronize(device)
                step_started = time.perf_counter()
                result = train_readout_step(reader, encoder, bridge, splits["train"][index], optimizer)
                _synchronize(device)
                metrics.write(_json({**result, "step": step, "history_id": splits["train"][index].history_id,
                                     "seconds": time.perf_counter() - step_started}) + "\n")
                metrics.flush()
        training_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        digest = save_checkpoint(output / "final.safetensors", encoder, bridge)
        del optimizer, encoder, bridge
        encoder, bridge = load_checkpoint(output / "final.safetensors", expected_sha256=digest,
                                           reader_width=width, kind=kind)
        encoder.to(reader.model.device)
        bridge.to(reader.model.device)
        for split in SPLITS:
            record(evaluate_readout(reader, encoder, bridge, splits[split], max_new_tokens=max_new_tokens), split, "final")
    if _reader_hash(reader) != reader_hash or any(p.requires_grad or p.grad is not None for p in reader.model.parameters()):
        raise ValueError("reader changed during the run")
    if _source_hashes() != sources:
        raise ValueError("execution source identity changed during the run")
    complete = {"kind": "readout_arm_complete_v1", "files": {name: file_sha256(output / name) for name in FILES},
                "reader_parameters_sha256": reader_hash, "prediction_count": prediction_count,
                "elapsed_seconds": time.perf_counter() - started,
                "training_peak_memory_bytes": training_peak}
    _write_json(output / "complete.json", complete)
    return complete


PROFILE_FILES = ("protocol.json", "schedule.json", "metrics.jsonl", "evaluation.json")


def profile_arm(
    reader: PretrainedReader, rows: Sequence[EncodedBefore], output: Path, *,
    kind: ReadoutKind, seed: int, steps: int, learning_rate: float, weight_decay: float,
    max_new_tokens: int, input_identity: dict,
) -> dict:
    """Measure disposable training and controlled evaluation on training rows only.

    The caller supplies training provenance; no development or confirmation loader
    is invoked here. Every step is recorded, including cold-start optimizer setup.
    CUDA peaks include resident reader weights, not just incremental allocations.
    CPU and MPS peak memory are unavailable, never reported as zero.
    """
    _validate_inputs(reader, {"train": rows}, kind=kind, seed=seed, steps=steps,
                     learning_rate=learning_rate, weight_decay=weight_decay,
                     max_new_tokens=max_new_tokens, input_identity=input_identity,
                     required_splits=("train",))
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    device = reader.model.device
    if device.type not in ("cpu", "cuda", "mps"):
        raise ValueError("profiling supports CPU, CUDA, or MPS")
    sources, reader_hash = _source_hashes(), _reader_hash(reader)
    width = reader.model.get_input_embeddings().embedding_dim
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        encoder, bridge = OneShotEncoder(width), ReadoutBridge(width, kind)
    encoder.to(device)
    bridge.to(device)
    schedule = training_schedule(len(rows), steps, seed)
    protocol = {
        "kind": "readout_profile_protocol_v1", "arm": kind, "seed": seed,
        "training_steps": steps, "data_role": "training_only",
        "input_identity": input_identity, "input_identity_verification": "caller_declared",
        "encodings_sha256": hashlib.sha256(_json([asdict(row) for row in rows]).encode()).hexdigest(),
        "reader_parameters_sha256": reader_hash, "reader_config": reader.model.config.to_dict(),
        "source_sha256": sources, "device": str(device), "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "optimizer": {"kind": "AdamW", "lr": learning_rate, "weight_decay": weight_decay,
                      "betas": [0.9, 0.999], "eps": 1e-8, "clip_norm": 1.0},
        "max_new_tokens": max_new_tokens, "persistent_bytes": STATE_BYTES,
        "feature_cache": False, "checkpoint_reuse": False, "warmup_steps_excluded": 0,
        "memory_measurement": "cuda_max_memory_allocated_including_reader" if device.type == "cuda" else "unavailable",
        "history_tokens": [len(row.history_ids) for row in rows],
        "history_ids": [row.history_id for row in rows],
    }
    _json(protocol)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "protocol.json", protocol)
    _write_json(output / "schedule.json", schedule)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(bridge.parameters()),
                                  lr=learning_rate, weight_decay=weight_decay)
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with (output / "metrics.jsonl").open("x") as metrics:
        for step, index in enumerate(schedule, 1):
            _synchronize(device)
            started = time.perf_counter()
            result = train_readout_step(reader, encoder, bridge, rows[index], optimizer)
            _synchronize(device)
            seconds = time.perf_counter() - started
            metrics.write(_json({**result, "step": step, "history_id": rows[index].history_id,
                                 "seconds": seconds, "cold_start": step == 1}) + "\n")
            metrics.flush()
    training_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    # Match post-checkpoint evaluation ownership: no optimizer or retained gradients.
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    readout_count = len(evaluate_readout(reader, encoder, bridge, rows, max_new_tokens=max_new_tokens))
    _synchronize(device)
    readout_seconds = time.perf_counter() - started
    started = time.perf_counter()
    full_text_count = len(evaluate_full_text(reader, rows, max_new_tokens=max_new_tokens))
    _synchronize(device)
    full_text_seconds = time.perf_counter() - started
    evaluation_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    _write_json(output / "evaluation.json", {
        "data_role": "training_only", "readout_predictions": readout_count,
        "full_text_predictions": full_text_count,
        "readout_seconds": readout_seconds, "full_text_seconds": full_text_seconds,
        "seconds": readout_seconds + full_text_seconds, "peak_memory_bytes": evaluation_peak,
    })
    if _reader_hash(reader) != reader_hash or any(p.requires_grad or p.grad is not None for p in reader.model.parameters()):
        raise ValueError("reader changed during profiling")
    if _source_hashes() != sources:
        raise ValueError("execution source identity changed during profiling")
    complete = {
        "kind": "readout_compute_profile_v1", "checkpoint_reuse": False,
        "training_steps": steps, "evaluation_histories": len(rows),
        "training_peak_memory_bytes": training_peak,
        "files": {name: file_sha256(output / name) for name in PROFILE_FILES},
    }
    _write_json(output / "complete.json", complete)
    return complete


def verify_profile(output: Path) -> dict:
    """Detect damaged profile artifacts; this is not scientific run completion."""
    output = Path(output)
    complete = read_json(output / "complete.json")
    if (complete.get("kind") != "readout_compute_profile_v1"
            or complete.get("checkpoint_reuse") is not False
            or set(complete.get("files", {})) != set(PROFILE_FILES)):
        raise ValueError("invalid profile completion identity")
    for name in PROFILE_FILES:
        if file_sha256(output / name) != complete["files"][name]:
            raise ValueError(f"profile artifact identity changed: {name}")
    return complete


def verify_run(output: Path) -> dict:
    """Check the completion seal against its exact local artifact set.

    This detects damage, not malicious replacement of both seal and artifacts.
    A publication or multi-arm manifest must bind the seal independently.
    """
    output = Path(output)
    complete = read_json(output / "complete.json")
    if complete.get("kind") != "readout_arm_complete_v1" or set(complete.get("files", {})) != set(FILES):
        raise ValueError("invalid run completion identity")
    for name in FILES:
        if file_sha256(output / name) != complete["files"][name]:
            raise ValueError(f"artifact identity changed: {name}")
    protocol = read_json(output / "protocol.json")
    if protocol.get("reader_parameters_sha256") != complete.get("reader_parameters_sha256"):
        raise ValueError("reader completion identity mismatch")
    with (output / "predictions.jsonl").open() as handle:
        count = sum(1 for line in handle if line.strip())
    if count != complete.get("prediction_count"):
        raise ValueError("prediction count identity mismatch")
    return complete
