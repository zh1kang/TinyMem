"""Matched-protocol comparisons for continuous and discrete memory."""

from collections.abc import Mapping, Sequence


MATCHED_PROTOCOL_FIELDS = (
    "seed",
    "base_checkpoint_sha256",
    "training_steps",
    "memory_warmup_steps",
    "segment_length",
    "max_distributed_distractor_tokens",
    "validation_distributed_distractor_tokens",
    "write_gate_kernel_size",
    "training_examples",
    "validation_examples",
)


def _evaluation_by_name(
    document: Mapping[str, object],
    field: str,
) -> dict[str, Mapping[str, object]]:
    evaluations = document.get(field)
    if not isinstance(evaluations, Sequence) or isinstance(
        evaluations,
        (str, bytes),
    ):
        raise ValueError(f"{field} must be a sequence")
    indexed = {}
    for evaluation in evaluations:
        if not isinstance(evaluation, Mapping):
            raise ValueError(f"{field} must contain mappings")
        name = evaluation.get("intervention")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{field} has an invalid intervention")
        if name in indexed:
            raise ValueError(f"{field} contains duplicate interventions")
        indexed[name] = evaluation
    return indexed


def _normal_summary(
    document: Mapping[str, object],
    field: str,
) -> dict[str, object]:
    normal = _evaluation_by_name(document, field).get("normal")
    if normal is None:
        raise ValueError(f"{field} must contain normal memory")
    correct = normal.get("correct")
    count = normal.get("count")
    if (
        isinstance(correct, bool)
        or not isinstance(correct, int)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or not 0 <= correct <= count
    ):
        raise ValueError(f"{field} has invalid normal accuracy counts")
    return {
        "correct": correct,
        "count": count,
        "accuracy": correct / count,
    }


def compare_discrete_to_continuous(
    continuous: Mapping[str, object],
    discrete: Mapping[str, object],
) -> dict[str, object]:
    """Validate matched conditions and compare normal-memory accuracy."""
    if not isinstance(continuous, Mapping):
        raise TypeError("continuous must be a mapping")
    if not isinstance(discrete, Mapping):
        raise TypeError("discrete must be a mapping")
    if continuous.get("compressor") == "discrete":
        raise ValueError("continuous result must not use a discrete compressor")
    if discrete.get("compressor") != "discrete":
        raise ValueError("discrete result must use the discrete compressor")

    mismatches = [
        field
        for field in MATCHED_PROTOCOL_FIELDS
        if continuous.get(field) != discrete.get(field)
    ]
    if mismatches:
        raise ValueError(
            "memory results use different protocols: " + ", ".join(mismatches)
        )
    continuous_bytes = continuous.get("memory_bytes_per_example")
    discrete_bytes = discrete.get("memory_bytes_per_example")
    if (
        isinstance(continuous_bytes, bool)
        or not isinstance(continuous_bytes, int)
        or isinstance(discrete_bytes, bool)
        or not isinstance(discrete_bytes, int)
        or continuous_bytes <= 0
        or discrete_bytes <= 0
    ):
        raise ValueError("memory results must contain positive byte counts")
    if discrete_bytes > continuous_bytes:
        raise ValueError("discrete memory exceeds the continuous byte budget")
    if continuous_bytes - discrete_bytes >= 17:
        raise ValueError("discrete memory leaves at least one code slot unused")

    comparisons = {}
    for label, field in (
        ("standard", "validation_evaluations"),
        ("delayed", "delayed_validation_evaluations"),
    ):
        continuous_summary = _normal_summary(continuous, field)
        discrete_summary = _normal_summary(discrete, field)
        if continuous_summary["count"] != discrete_summary["count"]:
            raise ValueError(f"{field} evaluates different example counts")
        comparisons[label] = {
            "continuous": continuous_summary,
            "discrete": discrete_summary,
            "discrete_minus_continuous_accuracy": (
                discrete_summary["accuracy"]
                - continuous_summary["accuracy"]
            ),
        }

    return {
        "status": "development_single_seed",
        "matched_protocol": {
            field: continuous.get(field)
            for field in MATCHED_PROTOCOL_FIELDS
        },
        "continuous_memory_bytes_per_example": continuous_bytes,
        "discrete_memory_bytes_per_example": discrete_bytes,
        "unused_byte_difference": continuous_bytes - discrete_bytes,
        "comparisons": comparisons,
        "codebook_diagnostics": discrete.get("codebook_diagnostics"),
    }
