"""Validation and Pareto analysis for adaptive write-cost sweeps."""

from collections.abc import Mapping, Sequence
from numbers import Real
from statistics import fmean, stdev


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
MATCHED_CONTROLLER_SEED_FIELDS = tuple(
    field
    for field in MATCHED_CONTROLLER_FIELDS
    if field not in {"seed", "base_checkpoint_sha256"}
) + (
    "task_id",
    "manifest_sha256",
    "write_cost_weight",
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


def aggregate_controller_seeds(
    documents: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Aggregate one selected adaptive-controller protocol across seeds."""
    if not isinstance(documents, Sequence) or isinstance(documents, (str, bytes)):
        raise TypeError("documents must be a sequence")
    if len(documents) < 2:
        raise ValueError("controller aggregation requires at least two seeds")
    if not all(isinstance(document, Mapping) for document in documents):
        raise TypeError("documents must contain mappings")

    reference = documents[0]
    mismatches = sorted(
        {
            field
            for document in documents[1:]
            for field in MATCHED_CONTROLLER_SEED_FIELDS
            if document.get(field) != reference.get(field)
        }
    )
    if mismatches:
        raise ValueError(
            "controller runs use different protocols: " + ", ".join(mismatches)
        )
    if reference.get("write_gate") != "adaptive":
        raise ValueError("controller aggregation requires adaptive write results")

    seeds = []
    checkpoint_hashes = []
    policy_rows: dict[str, list[Mapping[str, object]]] = {}
    reference_policies: tuple[str, ...] | None = None
    relevant_rates = []
    background_rates = []
    learned_random_differences = []
    significant_seed_count = 0
    for document in documents:
        seed = document.get("seed")
        checkpoint_hash = document.get("base_checkpoint_sha256")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("each controller run must have a nonnegative seed")
        if not isinstance(checkpoint_hash, str) or not checkpoint_hash:
            raise ValueError("each controller run must have a checkpoint hash")
        seeds.append(seed)
        checkpoint_hashes.append(checkpoint_hash)

        comparison = document.get("controller_comparison")
        if not isinstance(comparison, Mapping):
            raise ValueError("each controller run must contain a comparison")
        policies = comparison.get("policies")
        if not isinstance(policies, Sequence) or isinstance(policies, (str, bytes)):
            raise ValueError("controller policies must be a sequence")
        names = tuple(
            policy.get("policy")
            for policy in policies
            if isinstance(policy, Mapping)
            and isinstance(policy.get("policy"), str)
        )
        if len(names) != len(policies) or len(set(names)) != len(names):
            raise ValueError("controller policies must have unique names")
        if reference_policies is None:
            reference_policies = names
        elif names != reference_policies:
            raise ValueError("controller runs must contain the same policies")
        for name, policy in zip(names, policies, strict=True):
            assert isinstance(policy, Mapping)
            policy_rows.setdefault(name, []).append(policy)

        relevant_rates.append(
            _unit_metric(comparison, "relevant_write_rate")
        )
        background_rates.append(
            _unit_metric(comparison, "background_write_rate")
        )
        paired = comparison.get("learned_vs_random")
        if not isinstance(paired, Mapping):
            raise ValueError("controller comparison must contain paired results")
        learned_random_differences.append(
            _bounded_metric(paired, "accuracy_difference", minimum=-1.0, maximum=1.0)
        )
        significant = paired.get("significant")
        if not isinstance(significant, bool):
            raise ValueError("paired controller result must report significance")
        significant_seed_count += int(significant)

    if len(set(seeds)) != len(seeds):
        raise ValueError("controller seeds must be unique")
    if len(set(checkpoint_hashes)) != len(checkpoint_hashes):
        raise ValueError("controller seeds must use independently trained checkpoints")

    assert reference_policies is not None
    policies = []
    for name in reference_policies:
        rows = policy_rows[name]
        accuracies = [_unit_metric(row, "accuracy") for row in rows]
        writes = [
            _bounded_metric(
                row,
                "writes_per_1000_tokens",
                minimum=0.0,
                maximum=float("inf"),
            )
            for row in rows
        ]
        policies.append(
            {
                "policy": name,
                "accuracy_mean": fmean(accuracies),
                "accuracy_sample_std": stdev(accuracies),
                "writes_per_1000_tokens_mean": fmean(writes),
                "writes_per_1000_tokens_sample_std": stdev(writes),
                "seed_results": [
                    {
                        "seed": seed,
                        "accuracy": accuracy,
                        "writes_per_1000_tokens": write_rate,
                    }
                    for seed, accuracy, write_rate in zip(
                        seeds,
                        accuracies,
                        writes,
                        strict=True,
                    )
                ],
            }
        )

    return {
        "status": "development_multi_seed",
        "seeds": seeds,
        "seed_count": len(seeds),
        "base_checkpoint_sha256": checkpoint_hashes,
        "matched_protocol": {
            field: reference.get(field)
            for field in MATCHED_CONTROLLER_SEED_FIELDS
        },
        "policies": policies,
        "learned_vs_random_accuracy_difference_mean": fmean(
            learned_random_differences
        ),
        "learned_vs_random_accuracy_difference_sample_std": stdev(
            learned_random_differences
        ),
        "learned_beats_random_seed_count": sum(
            difference > 0 for difference in learned_random_differences
        ),
        "learned_vs_random_significant_seed_count": significant_seed_count,
        "relevant_write_rate_mean": fmean(relevant_rates),
        "relevant_write_rate_sample_std": stdev(relevant_rates),
        "background_write_rate_mean": fmean(background_rates),
        "background_write_rate_sample_std": stdev(background_rates),
    }


def _bounded_metric(
    document: Mapping[str, object],
    field: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    value = _finite_float(document.get(field), field=field)
    if not minimum <= value <= maximum:
        raise ValueError(f"{field} must be in [{minimum}, {maximum}]")
    return value


def _unit_metric(document: Mapping[str, object], field: str) -> float:
    return _bounded_metric(document, field, minimum=0.0, maximum=1.0)
