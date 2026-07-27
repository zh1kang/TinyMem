from dataclasses import FrozenInstanceError

import pytest

from tinymem.model.config import ModelConfig


def test_model_config_defaults() -> None:
    config = ModelConfig()

    assert config.vocab_size == 512
    assert config.d_model == 256
    assert config.n_layers == 4
    assert config.n_heads == 4
    assert config.d_ff == 1024
    assert config.dropout == 0.0
    assert config.max_local_tokens == 128
    assert config.positional_encoding == "rope"
    assert config.tie_embeddings is True
    assert config.d_model // config.n_heads == 64


def test_model_config_is_frozen() -> None:
    config = ModelConfig()

    with pytest.raises(FrozenInstanceError):
        config.d_model = 128


@pytest.mark.parametrize(
    "field_name",
    [
        "vocab_size",
        "d_model",
        "n_layers",
        "n_heads",
        "d_ff",
        "max_local_tokens",
    ],
)
@pytest.mark.parametrize("invalid_value", [0, -1])
def test_model_config_rejects_nonpositive_dimensions(
    field_name: str, invalid_value: int
) -> None:
    with pytest.raises(ValueError, match=f"{field_name} must be positive"):
        ModelConfig(**{field_name: invalid_value})


@pytest.mark.parametrize("invalid_value", [True, 4.0, "4"])
def test_model_config_rejects_noninteger_dimensions(invalid_value: object) -> None:
    with pytest.raises(TypeError, match="n_layers must be an integer"):
        ModelConfig(n_layers=invalid_value)


def test_model_config_rejects_uneven_attention_heads() -> None:
    with pytest.raises(ValueError, match="d_model must be divisible by n_heads"):
        ModelConfig(d_model=250, n_heads=4)


@pytest.mark.parametrize("invalid_value", [-0.1, 1.0])
def test_model_config_rejects_dropout_outside_valid_range(
    invalid_value: float,
) -> None:
    with pytest.raises(ValueError, match=r"dropout must be in \[0.0, 1.0\)"):
        ModelConfig(dropout=invalid_value)


@pytest.mark.parametrize("valid_value", [0.0, 0.5, 0.999])
def test_model_config_accepts_dropout_inside_valid_range(
    valid_value: float,
) -> None:
    assert ModelConfig(dropout=valid_value).dropout == valid_value


@pytest.mark.parametrize("invalid_value", [True, "0.1"])
def test_model_config_rejects_nonreal_dropout(invalid_value: object) -> None:
    with pytest.raises(TypeError, match="dropout must be a real number"):
        ModelConfig(dropout=invalid_value)


def test_model_config_rejects_unsupported_positional_encoding() -> None:
    with pytest.raises(ValueError, match="positional_encoding must be 'rope'"):
        ModelConfig(positional_encoding="absolute")


def test_model_config_rejects_nonstring_positional_encoding() -> None:
    with pytest.raises(TypeError, match="positional_encoding must be a string"):
        ModelConfig(positional_encoding=1)


def test_model_config_rejects_nonboolean_embedding_tying() -> None:
    with pytest.raises(TypeError, match="tie_embeddings must be a boolean"):
        ModelConfig(tie_embeddings="yes")


def test_model_config_accepts_disabled_embedding_tying() -> None:
    assert ModelConfig(tie_embeddings=False).tie_embeddings is False
