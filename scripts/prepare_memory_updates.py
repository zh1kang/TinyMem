"""Freeze paired-update data without reading the old confirmation dataset."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

from tinymem.data.babi import parse_question_payload, split_numbered_line
from tinymem.data.memory_updates import make_update_episode, text_sha256, validate_update_splits
from tinymem.data.opaque_qa1 import PEOPLE
from tinymem.data.reader_gate import ReaderCase
from tinymem.data.symbolic_world import parse_qa1_movement
from tinymem.research.pretrained import QWEN_MODEL_ID, QWEN_REVISION
from tinymem.research.study_runtime import repository_path


ROOT = Path(__file__).resolve().parents[1]
DESIGN = Path("configs/memory_update_study.json")
REFERENCE = Path("artifacts/predictions/opaque_qa1_data_20260905")
STUDY = Path("artifacts/predictions/opaque_memory_study_20260905/protocol.json")
RESERVE = Path("artifacts/predictions/native_holdout_reserve_20260904")
SOURCE = Path("data/raw/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt")
CODE = ("scripts/prepare_memory_updates.py", "src/tinymem/data/memory_updates.py",
        "src/tinymem/data/opaque_qa1.py", "src/tinymem/data/symbolic_world.py",
        "src/tinymem/data/reader_gate.py", "src/tinymem/data/babi.py",
        "src/tinymem/research/study_runtime.py", "src/tinymem/research/pretrained.py",
        "src/tinymem/research/update_encoding.py")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def _final_contexts(source: Path, episode_group: dict, allowed: set[str]) -> dict:
    """Input-only pass: question/answer fields are not parsed for selection."""
    finals, facts, episode_number = {}, [], 0
    with source.open() as handle:
        for raw in handle:
            line_id, payload = split_numbered_line(raw)
            if line_id == 1:
                episode_number += 1
                facts = []
            episode = f"{source.name}:episode-{episode_number:06d}"
            if episode_group[episode] not in allowed:
                continue
            if "\t" not in payload:
                facts.append(payload)
            else:
                finals[episode] = {"source_example_id": f"{episode}:question-{line_id}",
                                   "context": "\n".join(facts)}
    return finals


def prepare_memory_updates(output: Path, *, root: Path = ROOT) -> dict:
    if output.exists():
        raise FileExistsError(f"output must be fresh: {output}")
    design = load(root / DESIGN)
    counts = design["worlds"]
    if (set(counts) != {"train", "development", "confirmation"}
            or any(type(n) is not int or n <= 0 for n in counts.values())
            or type(design["seed"]) is not int or design["seed"] < 0):
        raise ValueError("positive split sizes and a nonnegative integer seed are required")
    fixed = {
        "protocol": "memory_update_study_v1", "persistent_bytes": 66,
        "state": {"slots": 2, "width": 8, "dtype": "float32", "validity_dtype": "bool"},
        "events": ["addition", "repetition", "correction"], "events_per_branch": 1,
        "chunk_separator": "\n\n", "initial_chunks": 4, "initial_known_entities": 8,
        "query_entities": 10, "always_absent_entities": 1, "addition_live_entities": 9,
        "reader": QWEN_MODEL_ID.removeprefix("Qwen/"), "reader_revision": QWEN_REVISION,
    }
    # JSON comparison also rejects bool/int and float/int substitutions.
    if json.dumps({key: design.get(key) for key in fixed}, sort_keys=True) != json.dumps(fixed, sort_keys=True):
        raise ValueError("unsupported update design: declare a separate experiment instead")
    if digest(root / STUDY) != design.get("old_study_protocol_sha256"):
        raise ValueError(f"input identity changed: {STUDY}")
    old_study = load(root / STUDY)
    reference = load(root / REFERENCE / "protocol.json")
    reserve = load(root / RESERVE / "protocol.json")
    expected = {
        STUDY: design["old_study_protocol_sha256"],
        REFERENCE / "protocol.json": old_study["data_protocol_sha256"],
        REFERENCE / "source_selection.json": reference["selection_sha256"],
        REFERENCE / "source_groups.json": reference["source_groups_sha256"],
        RESERVE / "data_manifest.json": reference["reserve_manifest_sha256"],
        RESERVE / "protocol.json": reference["reserve_protocol_sha256"],
        SOURCE: reference["source_sha256"][str(SOURCE)],
    }
    for name, sha in reserve["exclusion_sha256"].items():
        path = repository_path(name, root=root)
        if path in expected and expected[path] != sha:
            raise ValueError(f"conflicting input hashes: {path}")
        expected[path] = sha
    for path, sha in expected.items():
        if digest(root / path) != sha:
            raise ValueError(f"input identity changed: {path}")
    manifest = load(root / RESERVE / "data_manifest.json")
    groups = {group["group_id"]: group for group in manifest["all_source_groups"]}
    episode_group = {episode: group_id for group_id, group in groups.items() for episode in group["episodes"]}
    previous = load(root / REFERENCE / "source_selection.json")
    previous_groups = load(root / REFERENCE / "source_groups.json")
    if set(previous) != set(counts):
        raise ValueError("reference split coverage changed")
    used = [row["group_id"] for rows in previous.values() for row in rows]
    if len(set(used)) != len(used) or any(previous_groups[group] != groups[group] for group in used):
        raise ValueError("reference selection repeats or misidentifies source groups")
    reserved = {row["group_id"] for row in manifest["selected"]}
    old_confirmation = {row["group_id"] for row in previous["confirmation"]}
    blocked = reserved | old_confirmation
    fresh = {group_id for group_id, group in groups.items()
             if group["excluded"] is False} - reserved - set(used)
    reused = {row["group_id"] for split in ("train", "development") for row in previous[split]}
    finals = _final_contexts(root / SOURCE, episode_group, reused | fresh)
    eligible = {}
    for episode, row in finals.items():
        if episode_group[episode] not in fresh:
            continue
        people = {parse_qa1_movement(line, index)[0] for index, line in enumerate(row["context"].splitlines(), 1)}
        if people == set(PEOPLE):
            eligible.setdefault(episode_group[episode], []).append(episode)

    def rank(tag, values):
        return sorted(values, key=lambda value: text_sha256(f"memory-update-v1:{design['seed']}:{tag}:{value}"))

    selection = {}
    for split in ("train", "development"):
        # Preserve original pairing and data roles, not merely the same set.
        if len(previous[split]) != 2 * counts[split]:
            raise ValueError("reused source counts must match the frozen train/development selections")
        selection[split] = previous[split]
    required = 2 * counts["confirmation"]
    if len(eligible) < required:
        raise ValueError("not enough fresh eligible source groups; do not reuse old confirmation")
    selection["confirmation"] = []
    for group in rank("confirmation", eligible)[:required]:
        episode = rank("representative", eligible[group])[0]
        row = finals[episode]
        selection["confirmation"].append({"group_id": group, "episode": episode,
            "source_example_id": row["source_example_id"], "context_sha256": text_sha256(row["context"])})
    selected = {}
    for rows in selection.values():
        for row in rows:
            group, episode = row["group_id"], row["episode"]
            if (group in blocked or episode_group[episode] != group
                    or finals[episode]["source_example_id"] != row["source_example_id"]
                    or text_sha256(finals[episode]["context"]) != row["context_sha256"]):
                raise ValueError("selected representative disagrees with original input history")
            selected[row["source_example_id"]] = (group, finals[episode]["context"])
    # Parse answers only for the already selected training-source representatives.
    # None belong to the old confirmation or original holdout groups.
    cases, episode_number = {}, 0
    with (root / SOURCE).open() as handle:
        for raw in handle:
            line_id, payload = split_numbered_line(raw)
            if line_id == 1:
                episode_number += 1
            case_id = f"{SOURCE.name}:episode-{episode_number:06d}:question-{line_id}"
            if case_id in selected:
                question, answer, _ = parse_question_payload(payload)
                group, context = selected[case_id]
                cases[case_id] = ReaderCase(case_id, "babi_qa1", group, context, question, answer)
    worlds = {}
    for split, rows in selection.items():
        worlds[split] = tuple(make_update_episode(
            tuple(cases[row["source_example_id"]] for row in rows[index:index + 2]),
            tuple(row["group_id"] for row in rows[index:index + 2]),
            episode_id=f"memory-update-v1:{split}:{index // 2:04d}", seed=design["seed"],
        ) for index in range(0, len(rows), 2))
    validate_update_splits(worlds, source_groups=groups, blocked_groups=blocked)
    output.mkdir(parents=True, exist_ok=False)
    records = {f"{split}.json": [asdict(world) for world in rows] for split, rows in worlds.items()}
    records["source_selection.json"] = selection
    records["source_groups.json"] = {row["group_id"]: groups[row["group_id"]] for rows in selection.values() for row in rows}
    for name, content in records.items():
        (output / name).write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")
    protocol = {
        "design": design, "design_sha256": digest(root / DESIGN),
        "generator_runtime": {"python": platform.python_version(), "implementation": platform.python_implementation(),
                              "rng": "random.Random_string_seed_version_2"},
        "input_sha256": {str(path): digest(root / path) for path in expected},
        "source_sha256": {path: digest(ROOT / path) for path in CODE},
        "data_sha256": {name: digest(output / name) for name in records},
        "counts": {split: {"worlds": len(rows), "source_groups": len(rows) * 2,
                           "questions_per_state": 10, "states_per_world": 4} for split, rows in worlds.items()},
        "eligible_fresh_confirmation_groups": len(eligible),
        "old_confirmation_groups_used": 0, "original_holdout_groups_used": 0,
        "prior_native_consumed_new_confirmation_groups": 0,
        "exclusion_files_verified": len(reserve["exclusion_sha256"]),
        "selection_uses_answers": False, "independent_symbolic_validation": "passed",
        "old_confirmation_data_read": False, "model_loaded": False,
        "scientific_results": "not_run", "tokenizer_parity": "required_before_training_not_checked_by_data_builder",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        parser.error("run from the repository root")
    result = prepare_memory_updates(args.output)
    # Never print the new confirmation histories or answers.
    print(json.dumps({key: result[key] for key in ("counts", "eligible_fresh_confirmation_groups",
          "exclusion_files_verified", "old_confirmation_data_read", "model_loaded", "scientific_results")}, indent=2))


if __name__ == "__main__":
    main()
