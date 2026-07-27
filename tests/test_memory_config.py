from dataclasses import FrozenInstanceError

import pytest

from tinymem.model.config import MemoryConfig


def test_memory_config_defaults() -> None:
    config = MemoryConfig()

    assert config.n_slots == 8
    assert config.codebook_size == 256
    assert config.code_dim == 256
    assert config.codes_per_write == 2
    assert config.update_interval_segments == 1
    assert config.controller_actions == ("keep", "write")


def test_memory_config_is_frozen() -> None:
    config = MemoryConfig()

    with pytest.raises(FrozenInstanceError):
        config.n_slots = 16


@pytest.mark.parametrize(
    "field_name",
    [
        "n_slots",
        "codebook_size",
        "code_dim",
        "codes_per_write",
        "update_interval_segments",
    ],
)
@pytest.mark.parametrize("invalid_value", [True, 1.0, "1"])
def test_memory_config_rejects_noninteger_fields(
    field_name: str,
    invalid_value: object,
) -> None:
    with pytest.raises(TypeError, match=f"{field_name} must be an integer"):
        MemoryConfig(**{field_name: invalid_value})


@pytest.mark.parametrize(
    "field_name",
    [
        "n_slots",
        "codebook_size",
        "code_dim",
        "codes_per_write",
        "update_interval_segments",
    ],
)
@pytest.mark.parametrize("invalid_value", [0, -1])
def test_memory_config_rejects_nonpositive_fields(
    field_name: str,
    invalid_value: int,
) -> None:
    with pytest.raises(ValueError, match=f"{field_name} must be positive"):
        MemoryConfig(**{field_name: invalid_value})


def test_memory_config_accepts_codes_per_write_equal_to_slots() -> None:
    config = MemoryConfig(n_slots=4, codes_per_write=4)

    assert config.codes_per_write == config.n_slots


def test_memory_config_rejects_more_codes_per_write_than_slots() -> None:
    with pytest.raises(ValueError, match="codes_per_write must not exceed n_slots"):
        MemoryConfig(n_slots=4, codes_per_write=5)


def test_memory_config_rejects_non_tuple_actions() -> None:
    with pytest.raises(TypeError, match="controller_actions must be a tuple"):
        MemoryConfig(controller_actions=["keep", "write"])


def test_memory_config_rejects_nonstring_action() -> None:
    with pytest.raises(TypeError, match="must contain strings only"):
        MemoryConfig(controller_actions=("keep", 1))


@pytest.mark.parametrize(
    ("actions", "message"),
    [
        ((), "must be nonempty"),
        (("keep", "keep"), "must contain unique values"),
        (("write", "keep"), "must be exactly"),
        (("keep", "delete"), "must be exactly"),
    ],
)
def test_memory_config_rejects_invalid_action_contract(
    actions: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        MemoryConfig(controller_actions=actions)
