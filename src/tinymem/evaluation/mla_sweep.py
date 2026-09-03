"""Aggregate matched MHA and MLA-lite quality versus cache measurements."""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MLASweepPoint:
    """Store aggregate metrics for one attention configuration."""

    attention_type: str
    kv_latent_dim: int | None
    cache_bytes: int
    parameter_count: int
    seeds: tuple[int, ...]
    validation_accuracy_mean: float
    validation_accuracy_std: float
    validation_loss_mean: float
    validation_loss_std: float
    delayed_accuracy_mean: float
    delayed_accuracy_std: float
    training_seconds_mean: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _number(record: dict[str, object], name: str) -> float:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    return float(value)


def _delayed_accuracy(record: dict[str, object]) -> float:
    curve = record.get("curve")
    if not isinstance(curve, list) or not curve:
        raise ValueError("curve must be a nonempty list")
    correct = 0
    count = 0
    for bucket in curve:
        if not isinstance(bucket, dict):
            raise TypeError("curve entries must be dictionaries")
        bucket_correct = bucket.get("correct")
        bucket_count = bucket.get("count")
        if (
            isinstance(bucket_correct, bool)
            or not isinstance(bucket_correct, int)
            or isinstance(bucket_count, bool)
            or not isinstance(bucket_count, int)
        ):
            raise TypeError("curve counts must be integers")
        if bucket_correct < 0 or bucket_count < 0 or bucket_correct > bucket_count:
            raise ValueError("curve counts are invalid")
        correct += bucket_correct
        count += bucket_count
    if count == 0:
        raise ValueError("curve must contain evaluated examples")
    return correct / count


def aggregate_mla_sweep(
    records: Sequence[dict[str, object]],
) -> list[MLASweepPoint]:
    """Aggregate independent seeds for matched attention configurations."""
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("records must be a sequence")
    if not records:
        raise ValueError("records must be nonempty")

    manifests = {record.get("manifest_sha256") for record in records}
    if len(manifests) != 1 or not all(isinstance(value, str) for value in manifests):
        raise ValueError("all records must use one data manifest")

    groups: dict[tuple[str, int | None], list[dict[str, object]]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("records must contain dictionaries")
        attention_type = record.get("attention_type")
        latent_dim = record.get("kv_latent_dim")
        if attention_type not in {"mha", "mla_lite"}:
            raise ValueError("attention_type must be 'mha' or 'mla_lite'")
        if attention_type == "mha" and latent_dim is not None:
            raise ValueError("MHA records must not define kv_latent_dim")
        if attention_type == "mla_lite" and (
            isinstance(latent_dim, bool) or not isinstance(latent_dim, int)
        ):
            raise TypeError("MLA-lite records must define an integer latent dim")
        groups.setdefault((attention_type, latent_dim), []).append(record)

    points = []
    for (attention_type, latent_dim), group in groups.items():
        seeds = tuple(sorted(int(_number(record, "seed")) for record in group))
        if len(set(seeds)) != len(seeds):
            raise ValueError("each configuration must use unique seeds")
        cache_bytes = {
            int(_number(record, "cache_bytes_batch1_full_window"))
            for record in group
        }
        parameter_counts = {
            int(_number(record, "parameter_count")) for record in group
        }
        if len(cache_bytes) != 1 or min(cache_bytes) <= 0:
            raise ValueError("cache bytes must be one positive matched value")
        if len(parameter_counts) != 1 or min(parameter_counts) <= 0:
            raise ValueError("parameter count must be one positive matched value")

        validation_accuracies = [
            _number(record, "validation_accuracy") for record in group
        ]
        validation_losses = [
            _number(record, "validation_loss") for record in group
        ]
        delayed_accuracies = [_delayed_accuracy(record) for record in group]
        training_seconds = [
            _number(record, "training_seconds") for record in group
        ]
        points.append(
            MLASweepPoint(
                attention_type=attention_type,
                kv_latent_dim=latent_dim,
                cache_bytes=cache_bytes.pop(),
                parameter_count=parameter_counts.pop(),
                seeds=seeds,
                validation_accuracy_mean=statistics.fmean(validation_accuracies),
                validation_accuracy_std=statistics.pstdev(validation_accuracies),
                validation_loss_mean=statistics.fmean(validation_losses),
                validation_loss_std=statistics.pstdev(validation_losses),
                delayed_accuracy_mean=statistics.fmean(delayed_accuracies),
                delayed_accuracy_std=statistics.pstdev(delayed_accuracies),
                training_seconds_mean=statistics.fmean(training_seconds),
            )
        )
    return sorted(points, key=lambda point: point.cache_bytes)
