from dataclasses import FrozenInstanceError
import json

import pytest

from tinymem.model.config import (
    ExperimentConfig,
    MTPConfig,
    MemoryConfig,
    ModelConfig,
    StreamConfig,
)


def test_mtp_config_defaults() -> None:
    config = MTPConfig()

    assert config.enabled is False
    assert config.horizons == (2, 3, 4)
    assert config.loss_weight == 0.2


def test_mtp_config_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        MTPConfig().enabled = True


def test_mtp_config_rejects_nonboolean_enabled() -> None:
    with pytest.raises(TypeError, match="enabled must be a boolean"):
        MTPConfig(enabled=1)


def test_mtp_config_rejects_non_tuple_horizons() -> None:
    with pytest.raises(TypeError, match="horizons must be a tuple"):
        MTPConfig(horizons=[2, 3])


@pytest.mark.parametrize(
    ("horizons", "error_type", "message"),
    [
        ((), ValueError, "must be nonempty"),
        ((2, True), TypeError, "must contain integers"),
        ((2, 3.0), TypeError, "must contain integers"),
        ((1, 2), ValueError, "must be at least 2"),
        ((2, 2), ValueError, "unique and strictly increasing"),
        ((3, 2), ValueError, "unique and strictly increasing"),
    ],
)
def test_mtp_config_rejects_invalid_horizons(
    horizons: tuple[object, ...],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        MTPConfig(horizons=horizons)


@pytest.mark.parametrize("invalid_value", [True, "0.2"])
def test_mtp_config_rejects_nonreal_loss_weight(invalid_value: object) -> None:
    with pytest.raises(TypeError, match="loss_weight must be a real number"):
        MTPConfig(loss_weight=invalid_value)


def test_mtp_config_accepts_zero_weight_when_disabled() -> None:
    assert MTPConfig(enabled=False, loss_weight=0).loss_weight == 0


def test_mtp_config_rejects_invalid_loss_weight() -> None:
    with pytest.raises(ValueError, match="loss_weight must be nonnegative"):
        MTPConfig(loss_weight=-0.1)
    with pytest.raises(ValueError, match="must be positive when MTP is enabled"):
        MTPConfig(enabled=True, loss_weight=0)


def test_experiment_config_defaults() -> None:
    config = ExperimentConfig()

    assert config.seed == 1337
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.stream, StreamConfig)
    assert isinstance(config.memory, MemoryConfig)
    assert config.model.max_local_tokens == config.stream.local_window
    assert config.memory.code_dim == config.model.d_model


def test_experiment_config_nested_defaults_are_not_shared() -> None:
    first = ExperimentConfig()
    second = ExperimentConfig()

    assert first.model is not second.model
    assert first.memory is not second.memory


@pytest.mark.parametrize("invalid_value", [True, 1.0, "1"])
def test_experiment_config_rejects_noninteger_seed(invalid_value: object) -> None:
    with pytest.raises(TypeError, match="seed must be an integer"):
        ExperimentConfig(seed=invalid_value)


def test_experiment_config_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="seed must be nonnegative"):
        ExperimentConfig(seed=-1)


def test_experiment_config_rejects_inconsistent_local_windows() -> None:
    with pytest.raises(ValueError, match="max_local_tokens must equal"):
        ExperimentConfig(stream=StreamConfig(segment_length=64, local_window=256))


def test_experiment_config_rejects_inconsistent_memory_width() -> None:
    with pytest.raises(ValueError, match="code_dim must equal"):
        ExperimentConfig(memory=MemoryConfig(code_dim=128))


def test_experiment_config_round_trip_preserves_all_fields() -> None:
    original = ExperimentConfig(
        seed=7,
        mtp=MTPConfig(enabled=True, horizons=(2, 5), loss_weight=0.3),
    )

    encoded = json.loads(json.dumps(original.to_dict()))
    restored = ExperimentConfig.from_dict(encoded)

    assert restored == original


def test_experiment_config_from_dict_uses_defaults_for_missing_sections() -> None:
    assert ExperimentConfig.from_dict({"seed": 9}) == ExperimentConfig(seed=9)


def test_experiment_config_from_dict_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown experiment config fields: extra"):
        ExperimentConfig.from_dict({"extra": 1})


def test_experiment_config_from_dict_rejects_nonmapping_section() -> None:
    with pytest.raises(TypeError, match="model must be a dictionary"):
        ExperimentConfig.from_dict({"model": []})
