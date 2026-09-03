import pytest
import torch

from tinymem.model.kv_cache import KVCache
from tinymem.model.latent_kv_cache import LatentKVCache
from tinymem.model.memory_input import AttentionMemory
from tinymem.model.mla_lite import MLALiteAttention


def make_attention() -> MLALiteAttention:
    return MLALiteAttention(
        d_model=16,
        n_heads=4,
        kv_latent_dim=4,
        max_position_embeddings=16,
    )


def test_mla_lite_preserves_shape() -> None:
    inputs = torch.randn(2, 7, 16)

    outputs = make_attention()(inputs)

    assert outputs.shape == inputs.shape


def test_mla_lite_uses_one_shared_latent_for_keys_and_values() -> None:
    attention = make_attention().eval()
    inputs = torch.randn(2, 3, 16)
    cache = LatentKVCache(max_length=16)

    attention(inputs, cache=cache)

    torch.testing.assert_close(cache.get(), attention.kv_down_proj(inputs))
    assert cache.get().shape == (2, 3, 4)


def test_mla_lite_cache_uses_fewer_bytes_than_mha_cache() -> None:
    inputs = torch.randn(2, 5, 16)
    latent_cache = LatentKVCache(max_length=16)
    standard_cache = KVCache(max_length=16)
    attention = make_attention().eval()

    attention(inputs, cache=latent_cache)
    standard_cache.append(
        torch.zeros(2, 4, 5, 4),
        torch.zeros(2, 4, 5, 4),
    )

    assert latent_cache.nbytes == 2 * 5 * 4 * inputs.element_size()
    assert standard_cache.nbytes == 2 * 2 * 5 * 16 * inputs.element_size()
    assert latent_cache.nbytes < standard_cache.nbytes


def test_cached_mla_lite_matches_full_attention() -> None:
    torch.manual_seed(101)
    attention = make_attention().eval()
    inputs = torch.randn(1, 6, 16)
    expected = attention(inputs)
    cache = LatentKVCache(max_length=16)
    outputs = []

    for index in range(inputs.shape[1]):
        outputs.append(
            attention(
                inputs[:, index : index + 1],
                position_offset=cache.end_position,
                cache=cache,
            )
        )

    torch.testing.assert_close(torch.cat(outputs, dim=1), expected)


def test_mla_lite_cannot_leak_future_tokens() -> None:
    torch.manual_seed(103)
    attention = make_attention().eval()
    original = torch.randn(1, 4, 16)
    changed = original.clone()
    changed[:, -1] = torch.randn(16)

    original_output = attention(original)
    changed_output = attention(changed)

    torch.testing.assert_close(original_output[:, :-1], changed_output[:, :-1])


def test_mla_lite_parameters_receive_finite_gradients() -> None:
    attention = make_attention()
    loss = attention(torch.randn(2, 5, 16)).square().mean()

    loss.backward()

    for parameter in attention.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_mla_lite_supports_external_semantic_memory() -> None:
    torch.manual_seed(107)
    attention = make_attention().eval()
    inputs = torch.randn(1, 1, 16)
    memory = AttentionMemory(
        values=torch.randn(1, 2, 16),
        valid=torch.tensor([[True, True]]),
        positions=torch.tensor([[1, 3]]),
    )

    expected = attention(inputs, position_offset=5)
    actual = attention(inputs, position_offset=5, memory=memory)

    assert not torch.allclose(actual, expected)


def test_invalid_mla_lite_memory_slots_have_no_effect() -> None:
    torch.manual_seed(109)
    attention = make_attention().eval()
    inputs = torch.randn(1, 1, 16)
    memory = AttentionMemory(
        values=torch.randn(1, 2, 16),
        valid=torch.tensor([[False, False]]),
        positions=torch.tensor([[-1, -1]]),
    )

    expected = attention(inputs, position_offset=5)
    actual = attention(inputs, position_offset=5, memory=memory)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("d_model", "n_heads", "kv_latent_dim"),
    [
        (0, 4, 4),
        (16, 0, 4),
        (15, 4, 4),
        (16, 4, 0),
        (16, 4, 16),
        (12, 4, 4),
        (True, 4, 4),
    ],
)
def test_mla_lite_rejects_invalid_dimensions(
    d_model: object,
    n_heads: object,
    kv_latent_dim: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        MLALiteAttention(d_model, n_heads, kv_latent_dim, 16)


def test_mla_lite_rejects_wrong_cache_type() -> None:
    with pytest.raises(TypeError, match="LatentKVCache"):
        make_attention()(torch.randn(1, 1, 16), cache=KVCache(16))
