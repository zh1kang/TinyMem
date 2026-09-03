import pytest

from tinymem.evaluation.mla_sweep import aggregate_mla_sweep


def make_record(
    *,
    attention_type: str,
    latent_dim: int | None,
    seed: int,
    cache_bytes: int,
    validation_accuracy: float,
) -> dict[str, object]:
    return {
        "attention_type": attention_type,
        "kv_latent_dim": latent_dim,
        "seed": seed,
        "manifest_sha256": "manifest",
        "cache_bytes_batch1_full_window": cache_bytes,
        "parameter_count": 1_000,
        "validation_accuracy": validation_accuracy,
        "validation_loss": 1.0 - validation_accuracy,
        "training_seconds": 2.0,
        "curve": [
            {"correct": int(validation_accuracy * 10), "count": 10},
        ],
    }


def test_aggregate_mla_sweep_groups_seeds_and_sorts_by_cache() -> None:
    records = [
        make_record(
            attention_type="mha",
            latent_dim=None,
            seed=1,
            cache_bytes=4_096,
            validation_accuracy=0.8,
        ),
        make_record(
            attention_type="mla_lite",
            latent_dim=16,
            seed=1,
            cache_bytes=512,
            validation_accuracy=0.6,
        ),
        make_record(
            attention_type="mla_lite",
            latent_dim=16,
            seed=2,
            cache_bytes=512,
            validation_accuracy=0.8,
        ),
    ]

    points = aggregate_mla_sweep(records)

    assert [point.attention_type for point in points] == ["mla_lite", "mha"]
    assert points[0].seeds == (1, 2)
    assert points[0].validation_accuracy_mean == pytest.approx(0.7)
    assert points[0].validation_accuracy_std == pytest.approx(0.1)
    assert points[0].delayed_accuracy_mean == pytest.approx(0.7)


def test_aggregate_mla_sweep_rejects_duplicate_seed() -> None:
    record = make_record(
        attention_type="mla_lite",
        latent_dim=16,
        seed=1,
        cache_bytes=512,
        validation_accuracy=0.7,
    )

    with pytest.raises(ValueError, match="unique seeds"):
        aggregate_mla_sweep([record, dict(record)])


def test_aggregate_mla_sweep_rejects_mismatched_manifests() -> None:
    first = make_record(
        attention_type="mha",
        latent_dim=None,
        seed=1,
        cache_bytes=4_096,
        validation_accuracy=0.8,
    )
    second = dict(first, seed=2, manifest_sha256="other")

    with pytest.raises(ValueError, match="manifest"):
        aggregate_mla_sweep([first, second])
