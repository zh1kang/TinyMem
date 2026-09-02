import pytest
import torch

from tinymem.model.latent_kv_cache import LatentKVCache


def make_latents(length: int, *, offset: float = 0.0) -> torch.Tensor:
    return (
        torch.arange(length * 3, dtype=torch.float32)
        .reshape(1, length, 3)
        .add(offset)
    )


def test_empty_latent_cache_has_zero_size_and_position() -> None:
    cache = LatentKVCache(max_length=4)

    assert cache.sequence_length == 0
    assert cache.latent_dim is None
    assert cache.end_position == 0
    assert cache.nbytes == 0
    with pytest.raises(ValueError, match="empty"):
        cache.get()


def test_append_stores_and_concatenates_latents() -> None:
    cache = LatentKVCache(max_length=4)
    first = make_latents(2)
    second = make_latents(1, offset=10)

    cache.append(first)
    cache.append(second)

    assert cache.sequence_length == 3
    assert cache.latent_dim == 3
    assert cache.end_position == 3
    torch.testing.assert_close(cache.get(), torch.cat((first, second), dim=1))


def test_append_trims_oldest_latents_and_updates_start_position() -> None:
    cache = LatentKVCache(max_length=3)
    latents = make_latents(5)

    cache.append(latents)

    assert cache.sequence_length == 3
    assert cache.start_position == 2
    assert cache.end_position == 5
    torch.testing.assert_close(cache.get(), latents[:, 2:, :])


def test_nbytes_uses_actual_tensor_storage() -> None:
    latents = torch.zeros(2, 4, 3, dtype=torch.float16)
    cache = LatentKVCache(max_length=4, latents=latents)

    assert cache.nbytes == latents.numel() * latents.element_size()


@pytest.mark.parametrize(
    "first, second",
    [
        (torch.zeros(1, 2, 3), torch.zeros(2, 1, 3)),
        (torch.zeros(1, 2, 3), torch.zeros(1, 1, 4)),
        (torch.zeros(1, 2, 3), torch.zeros(1, 1, 3, dtype=torch.float64)),
    ],
)
def test_append_rejects_incompatible_latents(
    first: torch.Tensor,
    second: torch.Tensor,
) -> None:
    cache = LatentKVCache(max_length=4, latents=first)

    with pytest.raises(ValueError):
        cache.append(second)


@pytest.mark.parametrize(
    "latents",
    [
        torch.zeros(1, 3),
        torch.zeros(1, 2, 3, 4),
        torch.zeros(1, 2, 3, dtype=torch.long),
    ],
)
def test_latent_cache_rejects_invalid_tensor(latents: torch.Tensor) -> None:
    with pytest.raises((TypeError, ValueError)):
        LatentKVCache(max_length=4, latents=latents)


def test_clear_resets_latent_cache() -> None:
    cache = LatentKVCache(max_length=4, latents=make_latents(2), start_position=7)

    cache.clear()

    assert cache.sequence_length == 0
    assert cache.latent_dim is None
    assert cache.start_position == 0
    assert cache.end_position == 0
    assert cache.nbytes == 0
