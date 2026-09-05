#!/usr/bin/env python3
"""Freeze source-group-disjoint bAbI-derived opaque association worlds."""

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from tinymem.data.babi import parse_question_payload, split_numbered_line
from tinymem.data.opaque_qa1 import PEOPLE, make_opaque_qa1_world
from tinymem.data.reader_gate import ReaderCase
from tinymem.data.symbolic_world import parse_qa1_movement
from tinymem.utils.experiment import current_git_source_state


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(tag, values):
    return sorted(values, key=lambda value: sha(f"opaque-qa1-v1:20260905:{tag}:{value}"))


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()
reserve = Path("artifacts/predictions/native_holdout_reserve_20260904")
reserve_protocol = json.loads((reserve / "protocol.json").read_text())
assert digest(reserve / "data_manifest.json") == "968a639f8cd0617120aab161ca44244d08e06278c85401db974a7ce6bf5db5f4"
for path, expected in reserve_protocol["exclusion_sha256"].items():
    assert digest(Path(path)) == expected
reserved = json.loads((reserve / "data_manifest.json").read_text())
blocked = {group["group_id"] for group in reserved["selected"]}
groups = {group["group_id"]: group for group in reserved["all_source_groups"]}
episode_group = {episode: group["group_id"] for group in groups.values() for episode in group["episodes"]}
manifests = [Path("artifacts/predictions") / name / "data_manifest.json" for name in (
    "reader_adaptation/20260904T215534.309647Z-9ef7a86d", "reader_partial_evidence_20260904", "native_memory_pilot_20260904")]
trained = set()
for path in manifests:
    for row in json.loads(path.read_text())["train"]:
        episode = row["case"]["case_id"].split(":question-", 1)[0]
        if episode in episode_group:
            trained.add(episode_group[episode])
source = Path("data/raw/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt")
assert digest(source) == reserve_protocol["source_sha256"][str(source)]
episode_number, facts, finals = 0, [], {}
with source.open() as handle:
    for raw_line in handle:
        line_id, payload = split_numbered_line(raw_line)
        if line_id == 1:
            episode_number += 1
            facts = []
        episode = f"{source.name}:episode-{episode_number:06d}"
        if episode_group[episode] in blocked:
            continue
        if "\t" not in payload:
            facts.append(payload)
        else:
            finals[episode] = {"context": "\n".join(facts), "source_example_id": f"{episode}:question-{line_id}"}
eligible = defaultdict(list)
for episode, row in finals.items():
    people = {parse_qa1_movement(sentence, index)[0] for index, sentence in enumerate(row["context"].splitlines(), 1)}
    if people == set(PEOPLE):
        eligible[episode_group[episode]].append(episode)
old_train = set(eligible) & trained
consumed_nontrain = {group for group in eligible if groups[group]["excluded"]} - trained
fresh = {group for group in eligible if not groups[group]["excluded"]}
assert (len(old_train), len(consumed_nontrain), len(fresh)) == (315, 114, 631)
confirmation = rank("confirmation", fresh)[:256]
development = rank("development", consumed_nontrain)[:64]
train = rank("train-existing", old_train) + rank("train-new", fresh - set(confirmation))[:197]
selection = {name: rank(name + "-pairing", values) for name, values in
             (("train", train), ("development", development), ("confirmation", confirmation))}
assert len(set(group for values in selection.values() for group in values)) == 832
assert all(group not in blocked for values in selection.values() for group in values)
expected_hashes = {"train": "c77ee230c1276fceab3e954409968003b12b2a92d92377a148cda941885c0dd5",
                   "development": "3203ef5088bc0de3be011688f96159610c3519f088a2373464d72fbf968c32dc",
                   "confirmation": "26dc1ac8169fc57c869a13c17e1dc838e586b7f8288c63c93500a95a5c3bacac"}
representatives, worlds = {}, {}
selected_rows = {}
for split, values in selection.items():
    chosen = []
    for group in values:
        episode = rank("representative", eligible[group])[0]
        row = finals[episode]
        chosen.append({"group_id": group, "episode": episode, "source_example_id": row["source_example_id"],
                       "context_sha256": sha(row["context"])})
        selected_rows[row["source_example_id"]] = (group, row["context"])
    assert sha(json.dumps(chosen, sort_keys=True, separators=(",", ":"))) == expected_hashes[split]
    representatives[split] = chosen

selected_cases, episode_number = {}, 0
with source.open() as handle:
    for raw_line in handle:
        line_id, payload = split_numbered_line(raw_line)
        if line_id == 1:
            episode_number += 1
        case_id = f"{source.name}:episode-{episode_number:06d}:question-{line_id}"
        if case_id in selected_rows:
            group, context = selected_rows[case_id]
            assert group not in blocked
            question, answer, _ = parse_question_payload(payload)
            selected_cases[case_id] = ReaderCase(case_id, "babi_qa1", group, context, question, answer)
assert len(selected_cases) == 832
for split, values in selection.items():
    cases = [selected_cases[row["source_example_id"]] for row in representatives[split]]
    worlds[split] = []
    for index in range(0, len(cases), 2):
        pair = (cases[index], cases[index + 1])
        pair_groups = (values[index], values[index + 1])
        variants = {name: asdict(make_opaque_qa1_world(pair, pair_groups, world_id=f"opaque-qa1-v1:{split}:{index // 2:04d}",
                                                     seed=20260905, **options))
                    for name, options in (("opaque", {}), ("short", {"opaque": False}),
                                          ("counterfactual", {"counterfactual": True}))}
        worlds[split].append(variants)
counts = {split: {"worlds": len(rows), "source_groups": len(selection[split]), "known_queries": 8 * len(rows),
                  "missing_queries": len(rows)} for split, rows in worlds.items()}
print(json.dumps(counts, indent=2), flush=True)
if args.dry_run:
    raise SystemExit(0)
args.output.mkdir(parents=True, exist_ok=False)
for split, rows in worlds.items():
    (args.output / f"{split}.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
(args.output / "source_selection.json").write_text(json.dumps(representatives, indent=2, sort_keys=True) + "\n")
(args.output / "source_groups.json").write_text(json.dumps({group: groups[group] for values in selection.values() for group in values}, indent=2, sort_keys=True) + "\n")
sources = [Path(__file__), Path("src/tinymem/data/opaque_qa1.py"), Path("src/tinymem/data/symbolic_world.py"), Path("src/tinymem/data/babi.py")]
protocol = {
    "protocol": "opaque_qa1_derived_association_v1", "seed": 20260905, "source": current_git_source_state(Path.cwd()).to_dict(),
    "claim": "babi_derived_association_stress_not_official_babi", "counts": counts,
    "source_sha256": {str(path): digest(path) for path in [source, *sources, *manifests]},
    "reserve_manifest_sha256": digest(reserve / "data_manifest.json"),
    "reserve_protocol_sha256": digest(reserve / "protocol.json"),
    "representative_fingerprints": expected_hashes, "data_sha256": {f"{split}.json": digest(args.output / f"{split}.json") for split in worlds},
    "selection_sha256": digest(args.output / "source_selection.json"),
    "source_groups_sha256": digest(args.output / "source_groups.json"),
    "selection": "input-only_all-four-person_final_context_then_fixed_hash_ranking_and_disjoint_pairing",
    "source_answers_used_for_selection": False, "source_answers_validated_after_selection": True,
    "original_reserved_groups_used": False, "prior_native_consumed_confirmation_groups": 0,
    "group_reuse_across_or_within_splits": False, "identity_bits": 80, "known_entities_per_world": 8,
    "missing_entities_per_world": 1, "chunks": "first_half_A_first_half_B_second_half_A_second_half_B_complete_sentences",
    "room_mapping": "per-world_random_permutation_of_six_original_room_words",
    "counterfactual": "same_entity_IDs_every_known_location_changed_by_cyclic_room_permutation",
    "short_control": "same_source_facts_room_mapping_and_queries_with_nine_fixed_short_names",
    "qualification": "development_only_full_context_at_least_0.95_known_and_0.95_missing_before_compressor_training",
    "confirmation_status": "sealed_from_model_evaluation_until_methods_training_and_compute_protocol_frozen",
    "limitations": ["native-study_unused_not_project-wide_untouched", "Qwen_pretraining_exposure_unknown",
                    "exact_identity_entropy_is_not_an_answer_accuracy_bound", "complete_sentence_boundaries_are_visible"],
}
(args.output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
for path in sources:
    (args.output / path.name).write_bytes(path.read_bytes())
print(f"frozen data: {args.output}", flush=True)
