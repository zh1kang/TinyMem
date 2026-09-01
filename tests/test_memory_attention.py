import pytest
import torch

from tinymem.model.attention import CausalSelfAttention
from tinymem.model.kv_cache import KVCache
from tinymem.model.memory_input import AttentionMemory


def make_memory(*, valid: bool = True) -> AttentionMemory:
    return AttentionMemory(
        values=torch.tensor(
            [[[1.0] * 8, [-2.0] * 8, [0.5] * 8]],
            dtype=torch.float32,
        ),
        valid=torch.tensor([[valid, valid, valid]]),
        positions=(
            torch.tensor([[1, 2, 3]])
            if valid
            else torch.tensor([[-1, -1, -1]])
        ),
    )


def test_valid_memory_changes_attention_output() -> None:
    torch.manual_seed(51)
    attention = CausalSelfAttention(8, 2, 8).eval()
    inputs = torch.randn(1, 1, 8)

    without_memory = attention(inputs, position_offset=5)
    with_memory = attention(inputs, position_offset=5, memory=make_memory())

    assert not torch.allclose(with_memory, without_memory)


def test_invalid_memory_slots_have_no_effect() -> None:
    torch.manual_seed(53)
    attention = CausalSelfAttention(8, 2, 8).eval()
    inputs = torch.randn(1, 1, 8)

    expected = attention(inputs, position_offset=5)
    actual = attention(
        inputs,
        position_offset=5,
        memory=make_memory(valid=False),
    )

    torch.testing.assert_close(actual, expected)


def test_memory_keys_do_not_enter_the_local_kv_cache() -> None:
    attention = CausalSelfAttention(8, 2, 8).eval()
    cache = KVCache(max_length=4)

    attention(
        torch.randn(1, 1, 8),
        position_offset=0,
        cache=cache,
    )
    attention(
        torch.randn(1, 1, 8),
        position_offset=1,
        cache=cache,
        memory=AttentionMemory(
            values=torch.randn(1, 2, 8),
            valid=torch.tensor([[True, False]]),
            positions=torch.tensor([[0, -1]]),
        ),
    )

    assert cache.sequence_length == 2


def test_observer_receives_full_probabilities_and_local_positions() -> None:
    attention = CausalSelfAttention(8, 2, 8).eval()
    observations: list[tuple[torch.Tensor, torch.Tensor]] = []

    attention(
        torch.randn(1, 1, 8),
        position_offset=5,
        memory=make_memory(),
        attention_observer=lambda probabilities, positions: observations.append(
            (probabilities, positions)
        ),
    )

    probabilities, positions = observations[0]
    assert probabilities.shape == (1, 2, 1, 4)
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(1, 2, 1),
    )
    assert torch.equal(positions, torch.tensor([5]))


def test_memory_attention_rejects_nonpast_positions() -> None:
    attention = CausalSelfAttention(8, 2, 8)
    memory = AttentionMemory(
        values=torch.randn(1, 1, 8),
        valid=torch.tensor([[True]]),
        positions=torch.tensor([[5]]),
    )

    with pytest.raises(ValueError, match="precede local queries"):
        attention(torch.randn(1, 1, 8), position_offset=5, memory=memory)


def test_memory_attention_rejects_incompatible_shape() -> None:
    attention = CausalSelfAttention(8, 2, 8)
    memory = AttentionMemory(
        values=torch.randn(2, 1, 8),
        valid=torch.ones(2, 1, dtype=torch.bool),
        positions=torch.zeros(2, 1, dtype=torch.long),
    )

    with pytest.raises(ValueError, match="same batch size"):
        attention(torch.randn(1, 1, 8), position_offset=1, memory=memory)
