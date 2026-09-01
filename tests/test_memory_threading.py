import torch

from tinymem.model.block import TransformerBlock
from tinymem.model.config import ModelConfig
from tinymem.model.kv_cache import KVCache
from tinymem.model.memory_input import AttentionMemory
from tinymem.model.transformer import DecoderOnlyTransformer


def make_memory(*, valid: bool = True) -> AttentionMemory:
    return AttentionMemory(
        values=torch.tensor([[[1.0] * 8, [-2.0] * 8]]),
        valid=torch.tensor([[valid, valid]]),
        positions=(
            torch.tensor([[1, 3]])
            if valid
            else torch.tensor([[-1, -1]])
        ),
    )


def make_model(*, max_local_tokens: int = 8) -> DecoderOnlyTransformer:
    return DecoderOnlyTransformer(
        ModelConfig(
            vocab_size=16,
            d_model=8,
            n_layers=2,
            n_heads=2,
            d_ff=16,
            max_local_tokens=max_local_tokens,
        )
    ).eval()


def test_block_normalizes_memory_before_attention() -> None:
    block = TransformerBlock(8, 2, 16, 8).eval()
    memory = make_memory()
    observed: list[AttentionMemory] = []

    hook = block.attention.register_forward_pre_hook(
        lambda _module, _args, kwargs: observed.append(kwargs["memory"]),
        with_kwargs=True,
    )
    try:
        block(torch.randn(1, 1, 8), position_offset=5, memory=memory)
    finally:
        hook.remove()

    assert len(observed) == 1
    torch.testing.assert_close(
        observed[0].values,
        block.norm_attention(memory.values),
    )
    assert torch.equal(observed[0].valid, memory.valid)
    assert torch.equal(observed[0].positions, memory.positions)


def test_transformer_memory_changes_local_logits() -> None:
    torch.manual_seed(61)
    model = make_model()
    input_ids = torch.tensor([[4]])

    without_memory = model(input_ids, position_offset=5)
    with_memory = model(input_ids, position_offset=5, memory=make_memory())

    assert not torch.allclose(with_memory, without_memory)


def test_invalid_transformer_memory_has_no_effect() -> None:
    torch.manual_seed(67)
    model = make_model()
    input_ids = torch.tensor([[4]])

    expected = model(input_ids, position_offset=5)
    actual = model(
        input_ids,
        position_offset=5,
        memory=make_memory(valid=False),
    )

    torch.testing.assert_close(actual, expected)


def test_transformer_memory_stays_outside_every_local_cache() -> None:
    model = make_model(max_local_tokens=4)
    caches = [KVCache(max_length=4) for _ in range(model.config.n_layers)]
    for position in range(4):
        model(
            torch.tensor([[position + 1]]),
            position_offset=position,
            caches=caches,
        )

    memory = AttentionMemory(
        values=torch.randn(1, 1, 8),
        valid=torch.tensor([[True]]),
        positions=torch.tensor([[0]]),
    )
    model(
        torch.tensor([[5]]),
        position_offset=4,
        caches=caches,
        memory=memory,
    )

    assert all(cache.sequence_length == 4 for cache in caches)
    assert all(cache.start_position == 1 for cache in caches)
