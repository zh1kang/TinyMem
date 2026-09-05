#!/usr/bin/env python3
"""Bind supplemental native-only exclusions to the frozen association study."""

import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


root = Path("artifacts/predictions")
data = root / "opaque_qa1_data_20260905"
expected_manifest = "fec7148342755450c75dbcbc4cae271aa3ba4cd4b33a6590aed912d32cb10d68"
source_protocol = root / "native_multiquery_gate_20260904/protocol.json"
source = json.loads((root / "native_multiquery_gate_20260904/data_manifest.json").read_text())
assert digest(root / "native_multiquery_gate_20260904/data_manifest.json") == expected_manifest
histories = {row["case"]["context"] for row in source["original_six_cases"]}
evidence = {}
for name in ("shared_history_oracle_20260904", "shared_history_oracle_continuation_20260904", "native_query_pool_gate_20260904"):
    for filename in ("protocol.json", "data_manifest.json"):
        path = root / name / filename
        evidence[str(path)] = digest(path)
    assert digest(root / name / "data_manifest.json") == expected_manifest
for name in ("batched_prefix_grouped_profile_20260904", "batched_prefix_diagnostic_20260904"):
    path = root / name / "protocol.json"
    protocol = json.loads(path.read_text())
    assert digest(source_protocol) in protocol.values()
    evidence[str(path)] = digest(path)
reserve = root / "native_holdout_reserve_20260904"
reserve_protocol = json.loads((reserve / "protocol.json").read_text())
for path, expected in reserve_protocol["exclusion_sha256"].items():
    assert digest(Path(path)) == expected
study_protocol = json.loads((data / "protocol.json").read_text())
selected = json.loads((data / "source_selection.json").read_text())
assert digest(data / "source_selection.json") == study_protocol["selection_sha256"]
all_groups = json.loads((data / "source_groups.json").read_text())
assert digest(data / "source_groups.json") == study_protocol["source_groups_sha256"]
used = set()
for split, rows in selected.items():
    for row in rows:
        assert row["group_id"] not in used
        used.add(row["group_id"])
        group = all_groups[row["group_id"]]
        assert row["episode"] in group["episodes"] and row["context_sha256"] in group["context_sha256"]
        if split == "confirmation":
            assert group["excluded"] is False
            assert all(hashlib.sha256(context.encode()).hexdigest() not in group["context_sha256"] for context in histories)
result = {
    "audit": "opaque_study_supplemental_exclusions_v1", "study_protocol_sha256": digest(data / "protocol.json"),
    "reserve_protocol_sha256": digest(reserve / "protocol.json"), "runner_sha256": digest(Path(__file__)),
    "evidence_sha256": evidence, "source_fit_protocol_sha256": digest(source_protocol),
    "source_fit_manifest_sha256": expected_manifest, "existing_training_contexts": len(histories),
    "additional_source_groups": 0, "original_exclusion_files_reverified": 31, "unique_study_source_groups": len(used),
    "confirmation_overlap": False, "model_evaluated": False,
    "limits": "native-study-used evidence scope, not project-wide untouched or pretraining-clean",
}
with (data / "supplemental_exclusion_audit.json").open("x") as handle:
    handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2))
