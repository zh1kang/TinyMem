#!/usr/bin/env python3
"""Reserve answer-independent native-study history groups before more training."""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from tinymem.data.babi import load_babi_file


SOURCES = (
    "qa1_single-supporting-fact_train.txt", "qa2_two-supporting-facts_train.txt",
    "qa3_three-supporting-facts_train.txt", "qa4_two-arg-relations_train.txt",
    "qa5_three-arg-relations_train.txt",
)
EXCLUSIONS = """
batched_prefix_profile_20260904/protocol.json
native_memory_oracle_20260904/data_manifest.json
native_memory_oracle_20260904/protocol.json
native_memory_pilot_20260904/data_manifest.json
native_memory_pilot_20260904/protocol.json
native_multiquery_gate_20260904/data_manifest.json
native_multiquery_gate_20260904/protocol.json
native_prefix_smoke_20260904/protocol.json
native_recurrent_profile_20260904/protocol.json
native_recurrent_profile_20260904_v2/protocol.json
native_recurrent_profile_20260904_v3/protocol.json
native_writer_gate_20260904/data_manifest.json
native_writer_gate_20260904/protocol.json
reader_adaptation/20260904T215534.309647Z-9ef7a86d/data_manifest.json
reader_adaptation/20260904T215534.309647Z-9ef7a86d/protocol.json
reader_adaptation_profile/20260904T215302.556334Z-561e7953/data_manifest.json
reader_adaptation_profile/20260904T215302.556334Z-561e7953/protocol.json
reader_gate/20260904T213639.047632Z-bffa79b4/data_manifest.json
reader_gate/20260904T213639.047632Z-bffa79b4/protocol.json
reader_gate_adapted/20260904T222016.495967Z-4da340a2/data_manifest.json
reader_gate_adapted/20260904T222016.495967Z-4da340a2/protocol.json
reader_gate_numerics/bfloat16_single/protocol.json
reader_gate_numerics/float32_fresh/protocol.json
reader_gate_numerics/float32_single/protocol.json
reader_gate_numerics/oracle_current_fact/protocol.json
reader_gate_numerics/oracle_fresh/protocol.json
reader_gate_smoke/20260904T213426.613434Z-7cbdece7/data_manifest.json
reader_gate_smoke/20260904T213426.613434Z-7cbdece7/protocol.json
reader_partial_evidence_20260904/data_manifest.json
reader_partial_evidence_20260904/evaluation_manifest.json
reader_partial_evidence_20260904/protocol.json
""".split()
SEED = 20260904
COUNTS = {"qa1": 512, "qa2": 256, "qa3": 256}


def text_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()
root = Path("data/raw/tasks_1-20_v1-2/en-10k")
episodes = defaultdict(list)
for filename in SOURCES:
    for row in load_babi_file(root / filename, task_id=filename.split("_", 1)[0], split="train"):
        episodes[row.source_example_id.rsplit(":question-", 1)[0]].append(row)
parents = {episode: episode for episode in episodes}


def component_root(episode):
    while parents[episode] != episode:
        parents[episode] = parents[parents[episode]]
        episode = parents[episode]
    return episode


owners = {}
for episode, rows in episodes.items():
    for row in rows:
        if row.context in owners:
            left, right = component_root(episode), component_root(owners[row.context])
            parents[max(left, right)] = min(left, right)
        else:
            owners[row.context] = episode
components = defaultdict(list)
for episode in episodes:
    components[component_root(episode)].append(episode)
used_contexts, used_histories, used_episodes = set(), set(), set()


def collect(value):
    if isinstance(value, dict):
        if isinstance(value.get("context"), str):
            used_contexts.add(value["context"])
        if isinstance(value.get("history_id"), str):
            used_histories.add(value["history_id"])
        for item in value.values():
            collect(item)
    elif isinstance(value, list):
        for item in value:
            collect(item)
    elif isinstance(value, str):
        match = re.match(r"^(qa[1-5]_[^:]+:episode-[0-9]+)", value)
        if match:
            used_episodes.add(match.group(1))


evidence = [Path("artifacts/predictions") / name for name in EXCLUSIONS]
assert len(evidence) == len(set(evidence)) == 31
for path in evidence:
    collect(json.loads(path.read_text()))
groups = []
for members in components.values():
    members = sorted(members)
    group_id = text_hash("\n".join(members))
    contexts = {row.context for episode in members for row in episodes[episode]}
    excluded = bool(set(members) & used_episodes or contexts & used_contexts or group_id in used_histories)
    groups.append({"group_id": group_id, "episodes": members,
                   "tasks": sorted({episode.split("_", 1)[0] for episode in members}),
                   "context_sha256": sorted(text_hash(context) for context in contexts), "excluded": excluded})
groups.sort(key=lambda group: group["group_id"])
remaining = {task: sum(not group["excluded"] and task in group["tasks"] for group in groups) for task in ("qa1", "qa2", "qa3", "qa4", "qa5")}
assert remaining == {"qa1": 1312, "qa2": 1448, "qa3": 1450, "qa4": 1861, "qa5": 1990}
reserved, selected = set(), []
for task, count in COUNTS.items():
    pool = [group for group in groups if not group["excluded"] and task in group["tasks"] and group["group_id"] not in reserved]
    pool.sort(key=lambda group: text_hash(f"native-heldout-reserve-v1:{SEED}:{task}:{group['group_id']}"))
    if len(pool) < count:
        raise ValueError(f"not enough unused groups for {task}")
    for group in pool[:count]:
        candidates = [episode for episode in group["episodes"] if episode.startswith(task + "_")]
        episode = min(candidates, key=lambda item: text_hash(f"native-heldout-reserve-v1:{SEED}:{item}"))
        row = max(episodes[episode], key=lambda item: int(item.source_example_id.rsplit(":question-", 1)[1]))
        selected.append({**group, "task": task, "representative_case_id": row.source_example_id,
                         "representative_context_sha256": text_hash(row.context)})
        reserved.add(group["group_id"])
summary = {"question_rows": sum(map(len, episodes.values())), "episodes": len(episodes), "groups": len(groups),
           "excluded_groups": sum(group["excluded"] for group in groups), "remaining_groups_by_task": remaining,
           "reserved_by_task": COUNTS, "reserved_groups": len(reserved)}
assert len(reserved) == len(selected) == sum(COUNTS.values())
print(json.dumps(summary, indent=2), flush=True)
if args.dry_run:
    raise SystemExit(0)
manifest = json.dumps({"selected": selected, "all_source_groups": groups}, indent=2, sort_keys=True) + "\n"
protocol = {
    "protocol": "native_heldout_history_reserve_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
    "claim": "reserved_native_study_histories_not_project_wide_untouched_or_pretraining_clean",
    "seed": SEED, "counts": summary, "source_sha256": {str(root / name): digest(root / name) for name in SOURCES},
    "exclusion_sha256": {str(path): digest(path) for path in evidence}, "runner_sha256": digest(Path(__file__)),
    "parser_source_sha256": {str(path): digest(path) for path in (Path("src/tinymem/data/babi.py"), Path("src/tinymem/data/schema.py"))},
    "manifest_sha256": text_hash(manifest),
    "grouping": "same_episode_or_transitive_exact_fact_context_overlap_across_qa1_to_qa5",
    "exclusion": "whole_group_if_any_prior_context_episode_or_component_history_id_matches",
    "selection": "sha256_rank_per_task_then_episode_hash_then_last_question_in_episode_no_answer_filter",
    "answers_used_for_selection": False, "model_evaluated": False, "official_test_or_external_read": False,
    "status": "reserved_exclude_from_all_future_training_and_development_until_frozen_confirmatory_evaluation",
    "not_yet_frozen": "methods_budgets_seeds_delays_and_derived_question_protocol",
    "limits": ["prior_byte_source_pools_overlap", "pretrained_Qwen_exposure_unknown",
               "audited_31_artifact_scope_not_proof_against_unsaved_ad_hoc_use", "semantic_combination_novelty_not_certified"],
    "linked_token_only_profiles": "independently_traced_to_included_source_manifests_no_additional_cases_found",
}
args.output.mkdir(parents=True, exist_ok=False)
(args.output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
(args.output / "data_manifest.json").write_text(manifest)
(args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
print(f"reserved: {args.output}; manifest SHA256: {protocol['manifest_sha256']}", flush=True)
