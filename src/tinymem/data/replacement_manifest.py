"""Exact, validated input snapshots for replacement experiments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict

from tinymem.data.replacement_qa import (
    REPLACEMENT_PROTOCOLS,
    ReplacementQAExample,
    replacement_history_id,
    validate_replacement_example,
)


def replacement_manifest(
    train: Sequence[ReplacementQAExample],
    validation: Sequence[ReplacementQAExample],
    *,
    protocol: str,
    data_seed: int,
    validation_seed: int,
) -> dict[str, object]:
    """Snapshot exact tokens and check split boundaries before any training."""
    if protocol not in REPLACEMENT_PROTOCOLS:
        raise ValueError(f"unsupported replacement protocol {protocol!r}")
    partitions = {}
    history_sets = {}
    for name, rows in (("train", train), ("validation", validation)):
        if not rows or any(row.split != name for row in rows):
            raise ValueError(f"{name} must contain examples from that split")
        for row in rows:
            validate_replacement_example(row)
        histories = [replacement_history_id(row) for row in rows]
        records = [asdict(row) for row in rows]
        encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
        partitions[name] = {
            "count": len(rows),
            "unique_histories": len(set(histories)),
            "content_sha256": hashlib.sha256(encoded).hexdigest(),
            "history_ids": histories,
            "examples": records,
        }
        history_sets[name] = set(histories)
    overlap = len(history_sets["train"] & history_sets["validation"])
    if protocol == "history_disjoint_v2" and overlap:
        raise ValueError("forbidden training/validation history overlap")
    return {
        "schema_version": 1,
        "protocol": protocol,
        "data_seed": data_seed,
        "validation_seed": validation_seed,
        "symbolic_replay_validated": True,
        "train_validation_history_overlap": overlap,
        "partitions": partitions,
    }
