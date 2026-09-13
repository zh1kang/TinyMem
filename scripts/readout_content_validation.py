"""Validate content-probe exports against declared original sources before fitting."""

import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from tinymem.data.opaque_qa1 import ROOMS
from tinymem.data.symbolic_world import parse_qa1_movement
from tinymem.research.readout_experiment import verify_run
from tinymem.research.update_protocol import file_sha256

SPLIT_COUNTS = {"train": 256, "development": 32}

def replay_targets(tokenizer, row):
    ordered, final = [], {}
    text = tokenizer.decode(row["history_ids"], skip_special_tokens=False)
    for i, line in enumerate(text.splitlines(), 1):
        if not line:
            continue
        entity, room = parse_qa1_movement(line, i)
        if entity not in final:
            ordered.append(entity)
        final[entity] = room.value
    if len(ordered) != 8:
        raise ValueError("independent history replay requires eight entities")
    return [ROOMS.index(final[entity]) for entity in ordered]


def validate_exports(directories, source_root, declaration_path, tokenizer_dir):
    declaration = json.loads(declaration_path.read_text())
    if declaration["gold_ce_tolerance"] != 3e-6:
        raise ValueError("gold replay tolerance differs from fixed protocol")
    expected = {f"{arm}_seed_{seed}" for arm in ("affine", "gelu") for seed in (1337, 2027, 4099)}
    if len(directories) != 6 or {p.name for p in directories} != expected:
        raise ValueError("all six distinct checkpoint exports are required before fitting")
    for name in ("readout_content_probe.py", "readout_content_validation.py", "export_readout_content.py"):
        if file_sha256(Path(__file__).with_name(name)) != declaration["source_sha256"][name]:
            raise ValueError("content-probe code differs from declaration")
    first_protocol = json.loads((source_root / "affine_seed_1337" / "protocol.json").read_text())
    snapshot = first_protocol["input_identity"]["reader"]["snapshot"]
    for name in ("tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json", "config.json"):
        if file_sha256(tokenizer_dir / name) != snapshot["files"][name]["sha256"]:
            raise ValueError("tokenizer differs from original reader")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    loaded, common = [], None
    for directory in directories:
        source = source_root / directory.name
        source_seal = verify_run(source)
        if file_sha256(source / "complete.json") != declaration["source_complete_sha256"][directory.name]:
            raise ValueError("original checkpoint is outside declaration")
        original = json.loads((source / "protocol.json").read_text())
        seal = json.loads((directory / "complete.json").read_text())
        if seal["kind"] != "readout_content_export_v1" or set(seal["files"]) != {"states.safetensors", "metadata.json", "replay.jsonl"}:
            raise ValueError("invalid export completion seal")
        for name, digest in seal["files"].items():
            if file_sha256(directory / name) != digest:
                raise ValueError("export artifact changed")
        meta = json.loads((directory / "metadata.json").read_text())
        if (meta["source_complete_sha256"] != file_sha256(source / "complete.json")
                or meta["source_seal"] != source_seal or (meta["arm"], meta["seed"]) != (original["arm"], original["seed"])
                or meta["declaration_sha256"] != file_sha256(declaration_path)
                or meta["export_script_sha256"] != declaration["source_sha256"]["export_readout_content.py"]
                or meta["state_encoding_policy"] != declaration["state_encoding_policy"]
                or meta["reader_parameters_sha256"] != original["reader_parameters_sha256"]
                or any(original[k] != v for k, v in meta["runtime"].items())):
            raise ValueError("export provenance differs from original or declaration")
        if set(meta["runtime"]) != {"device", "torch_version", "cuda_version", "device_name", "reader_dtype", "deterministic_algorithms"}:
            raise ValueError("runtime coverage differs")
        encodings = json.loads((source / "encodings.json").read_text())
        states = load_file(directory / "states.safetensors")
        if set(states) != {"train", "development"} or set(meta["rows"]) != set(states) or set(encodings) != set(states):
            raise ValueError("unexpected export splits")
        ids, source_ids, cases = set(), set(), {}
        for split, count in SPLIT_COUNTS.items():
            if len(encodings[split]) != count or len(meta["rows"][split]) != count:
                raise ValueError("unexpected split counts")
            if states[split].dtype != torch.float32 or states[split].shape != (count, 16):
                raise ValueError("state dtype or shape differs")
            if not torch.isfinite(states[split]).all() or (states[split].abs() > 1).any():
                raise ValueError("invalid state values")
            for row, record in zip(encodings[split], meta["rows"][split], strict=True):
                targets = replay_targets(tokenizer, row)
                expected_record = {"history_id": row["history_id"], "source_group_ids": row["source_group_ids"], "targets": targets}
                if record != expected_record:
                    raise ValueError("export rows or first-appearance labels differ from original history")
                if row["history_id"] in ids or source_ids.intersection(row["source_group_ids"]):
                    raise ValueError("overlapping histories or source groups")
                ids.add(row["history_id"])
                source_ids.update(row["source_group_ids"])
                if len(row["queries"]) != 10:
                    raise ValueError("unexpected query count")
                for query in row["queries"]:
                    key = split, row["history_id"], query["case_id"]
                    if key in cases:
                        raise ValueError("duplicate source query")
                    cases[key] = query
        saved = {}
        for line in (source / "predictions.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["phase"] == "final" and r["condition"] == "normal":
                key = r["split"], r["history_id"], r["case_id"]
                if key in saved:
                    raise ValueError("duplicate saved replay target")
                saved[key] = r
        records = [json.loads(line) for line in (directory / "replay.jsonl").read_text().splitlines()]
        observed = set()
        for record in records:
            key = record["split"], record["history_id"], record["case_id"]
            if key in observed or key not in cases or key not in saved:
                raise ValueError("invalid or duplicate replay record")
            observed.add(key)
            error = abs(record["answer_ce"] - saved[key]["answer_ce"])
            if (not math.isfinite(error) or error > declaration["gold_ce_tolerance"]
                    or record["error"] != error or record["saved_answer_ce"] != saved[key]["answer_ce"]
                    or record["answer"] != cases[key]["answer"] or record["answer"] != saved[key]["answer"]):
                raise ValueError("gold replay differs from original")
        expected_records = sum(SPLIT_COUNTS.values()) * 10
        if (len(records) != expected_records or observed != cases.keys() or observed != saved.keys()
                or seal["records"] != expected_records or seal["max_replay_error"] != max(r["error"] for r in records)):
            raise ValueError("replay coverage or maximum differs")
        shared = (meta["rows"], meta["runtime"], meta["reader_parameters_sha256"], original["source_sha256"], original["input_identity"]["reader"])
        if common is None:
            common = shared
        elif common != shared:
            raise ValueError("cross-run rows, labels, runtime, or source identity differ")
        loaded.append((meta, states, seal))
    return loaded
