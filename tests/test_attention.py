import pytest
import torch

from tinymem.model.attention import CausalSelfAttention
from tinymem.model.kv_cache import KVCache


def test_attention_preserves_shape() -> None:
    attention = CausalSelfAttention(
        d_model=16,
        n_heads=4,
        max_position_embeddings=32,
    )
    inputs = torch.randn(2, 7, 16)

    outputs = attention(inputs)

    assert outputs.shape == inputs.shape


@pytest.mark.parametrize(
    ("d_model", "n_heads"),
    [(0, 2), (16, 0), (15, 4), (16, True), (True, 4)],
)
def test_attention_rejects_invalid_dimensions(d_model: object, n_heads: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_position_embeddings=32,
        )


@pytest.mark.parametrize("max_position_embeddings", [0, -1, True, 32.0])
def test_attention_rejects_invalid_max_positions(
    max_position_embeddings: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        CausalSelfAttention(
            d_model=16,
            n_heads=4,
            max_position_embeddings=max_position_embeddings,
        )


@pytest.mark.parametrize("dropout", [-0.1, 1.0, True, "0.1"])
def test_attention_rejects_invalid_dropout(dropout: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        CausalSelfAttention(
            d_model=16,
            n_heads=4,
            max_position_embeddings=32,
            dropout=dropout,
        )


def test_attention_rejects_wrong_input_shape() -> None:
    with pytest.raises((TypeError, ValueError)):
        CausalSelfAttention(16, 4, 32)(torch.randn(2, 7))


@pytest.mark.parametrize("inputs", [torch.ones(2, 7, 16, dtype=torch.long), torch.ones(2, 7, 15)])
def test_attention_rejects_invalid_input_tensor(inputs: torch.Tensor) -> None:
    with pytest.raises((TypeError, ValueError)):
        CausalSelfAttention(16, 4, 32)(inputs)


def test_attention_rejects_future_position_overflow() -> None:
    attention = CausalSelfAttention(16, 4, 4)

    with pytest.raises(ValueError, match="position"):
        attention(torch.randn(1, 5, 16))


def test_cached_attention_matches_full_prefill() -> None:
    torch.manual_seed(12)
    attention = CausalSelfAttention(16, 4, 32).eval()
    inputs = torch.randn(1, 6, 16)

    full_output = attention(inputs)
    cache = KVCache(max_length=32)
    cached_output = attention(inputs, cache=cache, position_offset=0)

    torch.testing.assert_close(cached_output, full_output)
    assert cache.sequence_length == inputs.shape[1]


def test_cached_attention_matches_token_by_token_decoding() -> None:
    torch.manual_seed(13)
    attention = CausalSelfAttention(16, 4, 32).eval()
    inputs = torch.randn(1, 6, 16)

    full_output = attention(inputs)
    cache = KVCache(max_length=32)
    token_outputs = []
    for index in range(inputs.shape[1]):
        token_outputs.append(
            attention(
                inputs[:, index : index + 1],
                cache=cache,
                position_offset=cache.end_position,
            )
        )
    cached_output = torch.cat(token_outputs, dim=1)

    torch.testing.assert_close(cached_output, full_output)


def test_cached_attention_requires_contiguous_position_offset() -> None:
    attention = CausalSelfAttention(16, 4, 32)
    cache = KVCache(max_length=32)

    with pytest.raises(ValueError, match="cache.end_position"):
        attention(torch.randn(1, 1, 16), cache=cache, position_offset=1)


def test_attention_rejects_invalid_cache_type() -> None:
    attention = CausalSelfAttention(16, 4, 32)

    with pytest.raises(TypeError, match="KVCache"):
        attention(torch.randn(1, 1, 16), cache=object())


def test_attention_observer_receives_pre_dropout_probabilities() -> None:
    attention = CausalSelfAttention(16, 4, 32, dropout=0.5).train()
    inputs = torch.randn(2, 3, 16)
    observations: list[tuple[torch.Tensor, torch.Tensor]] = []
    dropout_inputs: list[torch.Tensor] = []

    hook = attention.dropout.register_forward_pre_hook(
        lambda _module, args: dropout_inputs.append(args[0].detach().clone())
    )
    try:
        attention(
            inputs,
            position_offset=4,
            attention_observer=lambda probabilities, positions: observations.append(
                (probabilities, positions)
            ),
        )
    finally:
        hook.remove()

    assert len(observations) == 1
    probabilities, positions = observations[0]
    assert probabilities.shape == (2, 4, 3, 3)
    assert torch.equal(positions, torch.tensor([4, 5, 6]))
    torch.testing.assert_close(probabilities, dropout_inputs[0])
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(2, 4, 3),
    )


def test_attention_observer_cannot_mutate_attention_computation() -> None:
    torch.manual_seed(14)
    attention = CausalSelfAttention(16, 4, 32).eval()
    inputs = torch.randn(1, 3, 16)
    expected = attention(inputs)

    actual = attention(
        inputs,
        attention_observer=lambda probabilities, positions: (
            probabilities.zero_(),
            positions.fill_(99),
        ),
    )

    torch.testing.assert_close(actual, expected)


def test_cached_attention_observer_receives_absolute_window_positions() -> None:
    attention = CausalSelfAttention(16, 4, 4).eval()
    cache = KVCache(max_length=4)
    observed_positions: list[torch.Tensor] = []

    for position in range(6):
        attention(
            torch.randn(1, 1, 16),
            cache=cache,
            position_offset=cache.end_position,
            attention_observer=lambda _probabilities, positions: (
                observed_positions.append(positions)
            ),
        )

    assert torch.equal(observed_positions[-1], torch.tensor([2, 3, 4, 5]))


def test_attention_rejects_noncallable_observer() -> None:
    attention = CausalSelfAttention(16, 4, 32)

    with pytest.raises(TypeError, match="callable"):
        attention(torch.randn(1, 2, 16), attention_observer=object())
