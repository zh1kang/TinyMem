from dataclasses import FrozenInstanceError

import pytest

from tinymem.model.config import TrainingConfig


def test_training_config_defaults() -> None:
    config = TrainingConfig()

    assert config.optimizer == "adamw"
    assert config.learning_rate == 0.0003
    assert config.weight_decay == 0.1
    assert config.batch_size == 32
    assert config.gradient_clip_norm == 1.0
    assert config.warmup_steps == 200
    assert config.max_steps == 10_000


def test_training_config_is_frozen() -> None:
    config = TrainingConfig()

    with pytest.raises(FrozenInstanceError):
        config.batch_size = 64


def test_training_config_rejects_nonstring_optimizer() -> None:
    with pytest.raises(TypeError, match="optimizer must be a string"):
        TrainingConfig(optimizer=1)


def test_training_config_rejects_unsupported_optimizer() -> None:
    with pytest.raises(ValueError, match="optimizer must be 'adamw'"):
        TrainingConfig(optimizer="sgd")


@pytest.mark.parametrize("field_name", ["batch_size", "max_steps"])
@pytest.mark.parametrize("invalid_value", [True, 1.0, "1"])
def test_training_config_rejects_noninteger_positive_fields(
    field_name: str,
    invalid_value: object,
) -> None:
    with pytest.raises(TypeError, match=f"{field_name} must be an integer"):
        TrainingConfig(**{field_name: invalid_value})


@pytest.mark.parametrize("field_name", ["batch_size", "max_steps"])
@pytest.mark.parametrize("invalid_value", [0, -1])
def test_training_config_rejects_nonpositive_integer_fields(
    field_name: str,
    invalid_value: int,
) -> None:
    with pytest.raises(ValueError, match=f"{field_name} must be positive"):
        TrainingConfig(**{field_name: invalid_value})


@pytest.mark.parametrize("invalid_value", [True, 1.0, "1"])
def test_training_config_rejects_noninteger_warmup(invalid_value: object) -> None:
    with pytest.raises(TypeError, match="warmup_steps must be an integer"):
        TrainingConfig(warmup_steps=invalid_value)


def test_training_config_accepts_zero_warmup() -> None:
    assert TrainingConfig(warmup_steps=0).warmup_steps == 0


def test_training_config_rejects_negative_warmup() -> None:
    with pytest.raises(ValueError, match="warmup_steps must be nonnegative"):
        TrainingConfig(warmup_steps=-1)


@pytest.mark.parametrize("field_name", ["learning_rate", "gradient_clip_norm"])
@pytest.mark.parametrize("invalid_value", [True, "1.0"])
def test_training_config_rejects_nonreal_positive_fields(
    field_name: str,
    invalid_value: object,
) -> None:
    with pytest.raises(TypeError, match=f"{field_name} must be a real number"):
        TrainingConfig(**{field_name: invalid_value})


@pytest.mark.parametrize("field_name", ["learning_rate", "gradient_clip_norm"])
@pytest.mark.parametrize("invalid_value", [0.0, -0.1])
def test_training_config_rejects_nonpositive_real_fields(
    field_name: str,
    invalid_value: float,
) -> None:
    with pytest.raises(ValueError, match=f"{field_name} must be positive"):
        TrainingConfig(**{field_name: invalid_value})


@pytest.mark.parametrize("field_name", ["learning_rate", "gradient_clip_norm"])
def test_training_config_accepts_integer_real_fields(field_name: str) -> None:
    assert getattr(TrainingConfig(**{field_name: 1}), field_name) == 1


@pytest.mark.parametrize("invalid_value", [True, "0.0"])
def test_training_config_rejects_nonreal_weight_decay(
    invalid_value: object,
) -> None:
    with pytest.raises(TypeError, match="weight_decay must be a real number"):
        TrainingConfig(weight_decay=invalid_value)


def test_training_config_accepts_zero_weight_decay() -> None:
    assert TrainingConfig(weight_decay=0.0).weight_decay == 0.0


def test_training_config_rejects_negative_weight_decay() -> None:
    with pytest.raises(ValueError, match="weight_decay must be nonnegative"):
        TrainingConfig(weight_decay=-0.1)


def test_training_config_accepts_warmup_equal_to_max_steps() -> None:
    config = TrainingConfig(warmup_steps=10, max_steps=10)

    assert config.warmup_steps == config.max_steps


def test_training_config_rejects_warmup_larger_than_max_steps() -> None:
    with pytest.raises(ValueError, match="warmup_steps must not exceed max_steps"):
        TrainingConfig(warmup_steps=11, max_steps=10)
