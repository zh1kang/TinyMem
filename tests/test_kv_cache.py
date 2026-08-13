import pytest
import torch

from tinymem.model.kv_cache import KVCache


def make_states(length: int, *, offset: float = 0.0) -> torch.Tensor:
    return (
        torch.arange(length * 2, dtype=torch.float32)
        .reshape(1, 1, length, 2)
        .add(offset)
    )


def test_empty_cache_has_zero_size_and_position() -> None:
    cache = KVCache(max_length=4)

    assert cache.sequence_length == 0
    assert cache.end_position == 0
    assert cache.nbytes == 0
    with pytest.raises(ValueError, match="empty"):
        cache.get()


def test_append_stores_and_concatenates_states() -> None:
    cache = KVCache(max_length=4)
    first_keys = make_states(2)
    first_values = make_states(2, offset=100)
    second_keys = make_states(1, offset=10)
    second_values = make_states(1, offset=110)

    cache.append(first_keys, first_values)
    cache.append(second_keys, second_values)
    keys, values = cache.get()

    assert cache.sequence_length == 3
    assert cache.end_position == 3
    torch.testing.assert_close(keys, torch.cat((first_keys, second_keys), dim=-2))
    torch.testing.assert_close(values, torch.cat((first_values, second_values), dim=-2))


def test_append_trims_oldest_states_and_updates_start_position() -> None:
    cache = KVCache(max_length=3)
    keys = make_states(5)
    values = make_states(5, offset=100)

    cache.append(keys, values)
    cached_keys, cached_values = cache.get()

    assert cache.sequence_length == 3
    assert cache.start_position == 2
    assert cache.end_position == 5
    torch.testing.assert_close(cached_keys, keys[..., 2:, :])
    torch.testing.assert_close(cached_values, values[..., 2:, :])


@pytest.mark.parametrize(
    "keys, values",
    [
        (torch.zeros(1, 1, 2, 2), torch.zeros(1, 2, 2, 2)),
        (torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 1, 2)),
        (torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 3)),
        (torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2, dtype=torch.float64)),
    ],
)
def test_append_rejects_incompatible_states(
    keys: torch.Tensor,
    values: torch.Tensor,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        KVCache(max_length=4).append(keys, values)


def test_clear_resets_cache_state() -> None:
    cache = KVCache(max_length=4, start_position=7)
    cache.append(make_states(2), make_states(2, offset=100))

    cache.clear()

    assert cache.sequence_length == 0
    assert cache.start_position == 0
    assert cache.end_position == 0
    assert cache.nbytes == 0
