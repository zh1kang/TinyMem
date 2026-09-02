"""Validation and Pareto analysis for adaptive write-cost sweeps."""

from collections.abc import Mapping, Sequence
from numbers import Real


WRITE_COST_WEIGHTS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1)
MATCHED_CONTROLLER_FIELDS = (
    "seed",
    "base_checkpoint_sha256",
    "training_steps",
    "memory_warmup_steps",
    "training_examples",
    "validation_examples",
    "compressor",
    "memory_update",
    "write_gate",
    "segment_length",
    "capacity",
    "summaries_per_segment",
    "memory_bytes_per_example",
    "max_distributed_distractor_tokens",
    "validation_distributed_distractor_tokens",
    "controller_hidden_width",
    "controller_temperature_start",
    "controller_temperature_end",
    "controller_anneal_steps",
)


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be numeric")
    converted = float(value)
    if not -float("inf") < converted < float("inf"):
        raise ValueError(f"{field} must be finite")
    return converted


def _learned_policy(document: Mapping[str, object]) -> Mapping[str, object]:
    comparison = document.get("controller_comparison")
    if not isinstance(comparison, Mapping):
        raise ValueError("controller result must contain a policy comparison")
    policies = comparison.get("policies")
    if not isinstance(policies, Sequence) or isinstance(policies, (str, bytes)):
        raise ValueError("controller comparison policies must be a sequence")
    learned = [
        policy
        for policy in policies
        if isinstance(policy, Mapping) and policy.get("policy") == "learned"
    ]
    if len(learned) != 1:
        raise ValueError("controller comparison must contain one learned policy")
    return learned[0]


def _is_dominated(
    point: Mapping[str, object],
    points: Sequence[Mapping[str, object]],
) -> bool:
    accuracy = float(point["delayed_recall_accuracy"])
    writes = float(point["writes_per_1000_tokens"])
    return any(
        float(other["delayed_recall_accuracy"]) >= accuracy
        and float(other["writes_per_1000_tokens"]) <= writes
        and (
            float(other["delayed_recall_accuracy"]) > accuracy
            or float(other["writes_per_1000_tokens"]) < writes
        )
        for other in points
        if other is not point
    )


def aggregate_controller_sweep(
    documents: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Validate the fixed lambda sweep and return its Pareto frontier."""
    if not isinstance(documents, Sequence) or isinstance(documents, (str, bytes)):
        raise TypeError("documents must be a sequence")
    if len(documents) != len(WRITE_COST_WEIGHTS):
        raise ValueError("controller sweep requires exactly five result documents")
    if not all(isinstance(document, Mapping) for document in documents):
        raise TypeError("documents must contain mappings")

    reference = documents[0]
    mismatches = sorted(
        {
            field
            for document in documents[1:]
            for field in MATCHED_CONTROLLER_FIELDS
            if document.get(field) != reference.get(field)
        }
    )
    if mismatches:
        raise ValueError(
            "controller sweep uses different protocols: " + ", ".join(mismatches)
        )
    if reference.get("write_gate") != "adaptive":
        raise ValueError("controller sweep requires adaptive write results")

    points = []
    seen_weights = set()
    for document in documents:
        weight = _finite_float(
            document.get("write_cost_weight"),
            field="write_cost_weight",
        )
        if weight in seen_weights:
            raise ValueError("controller sweep contains a duplicate write cost")
        seen_weights.add(weight)
        learned = _learned_policy(document)
        correct = learned.get("correct")
        count = learned.get("count")
        if (
            isinstance(correct, bool)
            or not isinstance(correct, int)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or not 0 <= correct <= count
        ):
            raise ValueError("learned policy has invalid accuracy counts")
        writes = _finite_float(
            learned.get("writes_per_1000_tokens"),
            field="writes_per_1000_tokens",
        )
        if writes < 0:
            raise ValueError("writes_per_1000_tokens must be nonnegative")
        points.append(
            {
                "write_cost_weight": weight,
                "correct": correct,
                "count": count,
                "delayed_recall_accuracy": correct / count,
                "writes_per_1000_tokens": writes,
            }
        )

    if seen_weights != set(WRITE_COST_WEIGHTS):
        raise ValueError("controller sweep does not match the required write costs")
    points.sort(key=lambda point: float(point["write_cost_weight"]))
    for point in points:
        point["pareto_optimal"] = not _is_dominated(point, points)
    frontier = sorted(
        (point for point in points if bool(point["pareto_optimal"])),
        key=lambda point: float(point["writes_per_1000_tokens"]),
    )
    return {
        "status": "development_single_seed",
        "matched_protocol": {
            field: reference.get(field)
            for field in MATCHED_CONTROLLER_FIELDS
        },
        "points": points,
        "pareto_frontier": frontier,
    }
