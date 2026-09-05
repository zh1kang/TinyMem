"""Artifact boundaries for the separate paired-update study.

Training/development verification never opens confirmation history or predictions.
The frozen old association study remains an input, not an editable launch protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from collections.abc import Mapping

from tinymem.data.memory_updates import UpdateEpisode, update_episode_from_dict
from tinymem.research.study_runtime import REPOSITORY, repository_path


DESIGN = "configs/memory_update_study.json"
OLD_STUDY = "artifacts/predictions/opaque_memory_study_20260905/protocol.json"
REFERENCE = "artifacts/predictions/opaque_qa1_data_20260905"
RESERVE = "artifacts/predictions/native_holdout_reserve_20260904"
SPLITS = ("train", "development", "confirmation")
RAW_SOURCE = "data/raw/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt"
DATA_SOURCES = (
    "scripts/prepare_memory_updates.py", "src/tinymem/data/memory_updates.py",
    "src/tinymem/data/opaque_qa1.py", "src/tinymem/data/symbolic_world.py",
    "src/tinymem/data/reader_gate.py", "src/tinymem/data/babi.py",
    "src/tinymem/research/study_runtime.py", "src/tinymem/research/pretrained.py",
    "src/tinymem/research/update_encoding.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def verify_hashes(hashes: Mapping[str, str], root: Path) -> None:
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("nonempty artifact hashes are required")
    for name, expected in hashes.items():
        path = root / repository_path(name, root=root)
        if file_sha256(path) != expected:
            raise ValueError(f"artifact identity changed: {name}")


@dataclass(frozen=True)
class DevelopmentData:
    directory: Path
    protocol_sha256: str
    protocol: dict
    train: tuple[UpdateEpisode, ...]
    development: tuple[UpdateEpisode, ...]


def _selection_metadata(directory: Path, protocol: dict, root: Path) -> dict:
    names = ("source_selection.json", "source_groups.json")
    verify_hashes({name: protocol["data_sha256"][name] for name in names}, directory)
    selection, groups = (read_json(directory / name) for name in names)
    previous = read_json(root / REFERENCE / "source_selection.json")
    reserve = read_json(root / RESERVE / "data_manifest.json")
    original_groups = {item["group_id"]: item for item in reserve["all_source_groups"]}
    blocked = {item["group_id"] for item in reserve["selected"]} | {item["group_id"] for item in previous["confirmation"]}
    if set(selection) != set(SPLITS):
        raise ValueError("selection must cover the three declared splits")
    old_groups = {item["group_id"] for rows in previous.values() for item in rows}
    seen_groups, seen_episodes, seen_contexts = set(), set(), set()
    for split in SPLITS:
        rows = selection[split]
        if len(rows) != 2 * protocol["design"]["worlds"][split]:
            raise ValueError("selection count differs from the declared design")
        if split != "confirmation" and rows != previous[split]:
            raise ValueError("training/development must preserve the original pairings")
        for row in rows:
            group, episode, context = (row[key] for key in ("group_id", "episode", "context_sha256"))
            if group in seen_groups or episode in seen_episodes or context in seen_contexts:
                raise ValueError("source/history overlap in selection metadata")
            metadata = groups.get(group)
            if (group in blocked or metadata is None or metadata != original_groups.get(group)
                    or episode not in metadata["episodes"] or context not in metadata["context_sha256"]
                    or row["source_example_id"].rsplit(":question-", 1)[0] != episode):
                raise ValueError("selected representative disagrees with source metadata")
            if split == "confirmation" and (metadata["excluded"] is not False
                    or group in old_groups):
                raise ValueError("confirmation selection reuses a consumed source")
            seen_groups.add(group)
            seen_episodes.add(episode)
            seen_contexts.add(context)
    if set(groups) != seen_groups:
        raise ValueError("source-group coverage differs from selection")
    return selection


def _load_split(directory: Path, protocol: dict, split: str, selection: dict) -> tuple[UpdateEpisode, ...]:
    name = f"{split}.json"
    verify_hashes({name: protocol["data_sha256"][name]}, directory)
    rows = tuple(update_episode_from_dict(row) for row in read_json(directory / name))
    if len(rows) != protocol["design"]["worlds"][split]:
        raise ValueError("history count differs from the declared design")
    selected = selection[split]
    for index, row in enumerate(rows):
        pair = selected[index * 2:index * 2 + 2]
        if (row.episode_id != f"memory-update-v1:{split}:{index:04d}"
                or row.source_group_ids != tuple(item["group_id"] for item in pair)
                or row.source_case_ids != tuple(item["source_example_id"] for item in pair)
                or row.source_context_sha256 != tuple(item["context_sha256"] for item in pair)):
            raise ValueError("history does not match the selected source pairing")
    return rows


def load_development_data(directory: Path, *, root: Path = REPOSITORY) -> DevelopmentData:
    """Check the complete input provenance, but read only train/dev histories."""
    root = root.resolve()
    directory = root / repository_path(directory, root=root)
    protocol = read_json(directory / "protocol.json")
    design = read_json(root / DESIGN)
    # Exact JSON comparison avoids accepting bool/float substitutes for integers.
    if (file_sha256(root / DESIGN) != protocol["design_sha256"]
            or json.dumps(design, sort_keys=True) != json.dumps(protocol["design"], sort_keys=True)):
        raise ValueError("update design identity changed")
    if file_sha256(root / OLD_STUDY) != design["old_study_protocol_sha256"]:
        raise ValueError("frozen old-study trust anchor changed")
    if protocol["input_sha256"].get(OLD_STUDY) != design["old_study_protocol_sha256"]:
        raise ValueError("data provenance lacks the old-study trust anchor")
    if set(protocol["data_sha256"]) != {"source_selection.json", "source_groups.json", *(f"{s}.json" for s in SPLITS)}:
        raise ValueError("data hash coverage differs from the declared splits")
    if set(protocol["source_sha256"]) != set(DATA_SOURCES):
        raise ValueError("data source coverage changed")
    verify_hashes(protocol["source_sha256"], root)
    verify_hashes(protocol["input_sha256"], root)
    # Required transitive identities cannot be deleted from the supplied map.
    old = read_json(root / OLD_STUDY)
    reference = read_json(root / REFERENCE / "protocol.json")
    reserve = read_json(root / RESERVE / "protocol.json")
    required = {OLD_STUDY: design["old_study_protocol_sha256"],
                f"{REFERENCE}/protocol.json": old["data_protocol_sha256"],
                f"{REFERENCE}/source_selection.json": reference["selection_sha256"],
                f"{REFERENCE}/source_groups.json": reference["source_groups_sha256"],
                f"{RESERVE}/data_manifest.json": reference["reserve_manifest_sha256"],
                f"{RESERVE}/protocol.json": reference["reserve_protocol_sha256"]}
    for mapping in ({RAW_SOURCE: reference["source_sha256"][RAW_SOURCE]}, reserve["exclusion_sha256"]):
        for name, digest in mapping.items():
            relative = str(repository_path(name, root=root))
            if relative in required and required[relative] != digest:
                raise ValueError("conflicting provenance hash declarations")
            required[relative] = digest
    if protocol["input_sha256"] != required:
        raise ValueError("input provenance coverage changed")
    selection = _selection_metadata(directory, protocol, root)
    # Metadata proves cross-split disjointness; exact pairing connects every
    # loaded episode to it without opening confirmation histories.
    splits = {split: _load_split(directory, protocol, split, selection) for split in SPLITS[:2]}
    return DevelopmentData(directory, file_sha256(directory / "protocol.json"), protocol,
                           splits["train"], splits["development"])


def shared_reader_identity(*, root: Path = REPOSITORY) -> dict:
    """Resolve the already-qualified adapter without loading model weights."""
    root = root.resolve()
    design = read_json(root / DESIGN)
    if file_sha256(root / OLD_STUDY) != design["old_study_protocol_sha256"]:
        raise ValueError("frozen old-study trust anchor changed")
    old = read_json(root / OLD_STUDY)
    gate = root / repository_path(old["reader_gate"], root=root)
    verify_hashes({"protocol.json": old["reader_gate_protocol_sha256"],
                   "results.json": old["reader_gate_results_sha256"]}, gate)
    protocol, results = read_json(gate / "protocol.json"), read_json(gate / "results.json")
    if protocol["protocol"] == "opaque_qa1_reader_continuation_v1":
        adapter = repository_path(results["final_adapter"], root=root)
        hashes = results["adapter_sha256"]
    elif protocol["protocol"] == "opaque_qa1_reader_qualification_v1":
        adapter = repository_path(protocol["adapter"], root=root)
        hashes = protocol["adapter_sha256"]
    else:
        raise ValueError("unsupported existing reader qualification")
    if results["reader_accepted"] is not True or hashes != old["adapter_sha256"] or protocol["snapshot"] != old["snapshot"]:
        raise ValueError("existing qualified reader differs from frozen study")
    verify_hashes(hashes, root / adapter)
    return {"snapshot": old["snapshot"], "adapter": str(adapter), "adapter_sha256": hashes,
            "old_study_protocol_sha256": design["old_study_protocol_sha256"],
            "reader_gate_protocol_sha256": old["reader_gate_protocol_sha256"],
            "reader_gate_results_sha256": old["reader_gate_results_sha256"]}


def load_shared_reader(identity: dict, device, *, root: Path = REPOSITORY):
    """Only production Qwen loading; tests use the core reader functions directly."""
    import torch
    from peft import PeftModel
    from tinymem.research.pretrained import load_qwen_reader, verify_qwen_snapshot

    if identity != shared_reader_identity(root=root):
        raise ValueError("reader identity changed before model loading")
    snapshot = root / "data/raw/pretrained/qwen3-1.7b"
    if verify_qwen_snapshot(snapshot) != identity["snapshot"]:
        raise ValueError("Qwen snapshot differs from frozen reader")
    reader = load_qwen_reader(snapshot, device=device, dtype=torch.bfloat16)
    reader.model = PeftModel.from_pretrained(reader.model, root / identity["adapter"],
        is_trainable=False, local_files_only=True, use_safetensors=True)
    reader.model.requires_grad_(False).eval()
    return reader
