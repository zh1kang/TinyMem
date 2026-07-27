from dataclasses import FrozenInstanceError

import pytest

from tinymem.model.config import StreamConfig


def test_stream_config_defaults() -> None:
    config = StreamConfig()

    assert config.segment_length == 64
    assert config.local_window == 128


def test_stream_config_is_frozen() -> None:
    config = StreamConfig()

    with pytest.raises(FrozenInstanceError):
        config.segment_length = 32


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("segment_length", True),
        ("segment_length", 64.0),
        ("segment_length", "64"),
        ("local_window", False),
        ("local_window", 128.0),
        ("local_window", "128"),
    ],
)
def test_stream_config_rejects_noninteger_fields(
    field_name: str,
    invalid_value: object,
) -> None:
    with pytest.raises(TypeError, match="must be integers"):
        StreamConfig(**{field_name: invalid_value})


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("segment_length", 0),
        ("segment_length", -1),
        ("local_window", 0),
        ("local_window", -1),
    ],
)
def test_stream_config_rejects_nonpositive_fields(
    field_name: str,
    invalid_value: int,
) -> None:
    with pytest.raises(ValueError, match="must be positive integers"):
        StreamConfig(**{field_name: invalid_value})


def test_stream_config_accepts_segment_equal_to_local_window() -> None:
    config = StreamConfig(segment_length=128, local_window=128)

    assert config.segment_length == config.local_window


def test_stream_config_rejects_segment_larger_than_local_window() -> None:
    with pytest.raises(ValueError, match="must not exceed local_window"):
        StreamConfig(segment_length=129, local_window=128)
