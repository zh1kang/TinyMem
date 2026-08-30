import pytest
import torch

from tinymem.model.config import ModelConfig
from tinymem.model.kv_cache import KVCache
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.training.losses import next_token_cross_entropy


def make_config(**overrides: object) -> ModelConfig:
    values: dict[str, object] = {
        "vocab_size": 32,
        "d_model": 16,
        "n_layers": 2,
        "n_heads": 4,
        "d_ff": 32,
        "max_local_tokens": 16,
        "tie_embeddings": True,
    }
    values.update(overrides)
    return ModelConfig(**values)


def test_transformer_rejects_invalid_config() -> None:
    with pytest.raises(TypeError):
        DecoderOnlyTransformer(config=16)


def test_transformer_preserves_logit_shape() -> None:
    model = DecoderOnlyTransformer(make_config())
    input_ids = torch.randint(0, 32, (2, 7))

    logits = model(input_ids)

    assert logits.shape == (2, 7, 32)


def test_transformer_ties_embedding_and_output_weights() -> None:
    model = DecoderOnlyTransformer(make_config(tie_embeddings=True))

    assert model.lm_head.weight is model.token_embedding.weight


def test_transformer_can_leave_weights_untied() -> None:
    model = DecoderOnlyTransformer(make_config(tie_embeddings=False))

    assert model.lm_head.weight is not model.token_embedding.weight


@pytest.mark.parametrize("input_ids", [torch.randn(2, 7), torch.ones(2, 7, 1, dtype=torch.long)])
def test_transformer_rejects_invalid_input_shape_or_dtype(
    input_ids: torch.Tensor,
) -> None:
    model = DecoderOnlyTransformer(make_config())

    with pytest.raises((TypeError, ValueError)):
        model(input_ids)


def test_transformer_rejects_out_of_range_token_ids() -> None:
    model = DecoderOnlyTransformer(make_config())

    with pytest.raises(ValueError, match="token ID"):
        model(torch.tensor([[0, 31, 32]]))


@pytest.mark.parametrize(
    ("input_ids", "position_offset", "message"),
    [
        (torch.empty(1, 0, dtype=torch.long), 0, "at least one token"),
        (torch.ones(1, 16, dtype=torch.long), 1, "max_local_tokens"),
    ],
)
def test_transformer_rejects_invalid_sequence_limits(
    input_ids: torch.Tensor,
    position_offset: int,
    message: str,
) -> None:
    model = DecoderOnlyTransformer(make_config())

    with pytest.raises(ValueError, match=message):
        model(input_ids, position_offset=position_offset)


def test_transformer_cannot_leak_future_tokens() -> None:
    torch.manual_seed(7)
    model = DecoderOnlyTransformer(make_config()).eval()
    original = torch.tensor([[1, 2, 3, 4]])
    changed_future = torch.tensor([[1, 2, 3, 9]])

    original_logits = model(original)
    changed_logits = model(changed_future)

    torch.testing.assert_close(original_logits[:, :3], changed_logits[:, :3])


def test_transformer_parameters_receive_finite_gradients() -> None:
    model = DecoderOnlyTransformer(make_config())
    input_ids = torch.randint(0, 32, (2, 7))

    loss = next_token_cross_entropy(model(input_ids), input_ids)
    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_transformer_overfits_a_tiny_batch() -> None:
    torch.manual_seed(11)
    config = make_config(
        vocab_size=8,
        d_model=16,
        n_layers=1,
        n_heads=2,
        d_ff=32,
        max_local_tokens=8,
    )
    model = DecoderOnlyTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    input_ids = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])

    for _ in range(100):
        optimizer.zero_grad()
        loss = next_token_cross_entropy(model(input_ids), input_ids)
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.01


def test_cached_transformer_matches_full_prefill() -> None:
    torch.manual_seed(22)
    model = DecoderOnlyTransformer(make_config()).eval()
    input_ids = torch.randint(0, 32, (1, 6))
    caches = [KVCache(max_length=16) for _ in range(model.config.n_layers)]

    full_logits = model(input_ids)
    cached_logits = model(input_ids, caches=caches)

    torch.testing.assert_close(cached_logits, full_logits)
    assert all(cache.sequence_length == input_ids.shape[1] for cache in caches)


def test_cached_transformer_matches_token_by_token_decoding() -> None:
    torch.manual_seed(23)
    model = DecoderOnlyTransformer(make_config()).eval()
    input_ids = torch.randint(0, 32, (1, 6))
    caches = [KVCache(max_length=16) for _ in range(model.config.n_layers)]

    full_logits = model(input_ids)
    cached_outputs = []
    for index in range(input_ids.shape[1]):
        position_offset = caches[0].end_position
        cached_outputs.append(
            model(
                input_ids[:, index : index + 1],
                position_offset=position_offset,
                caches=caches,
            )
        )

    cached_logits = torch.cat(cached_outputs, dim=1)
    torch.testing.assert_close(cached_logits, full_logits)


def test_transformer_forwards_observer_to_every_attention_layer() -> None:
    model = DecoderOnlyTransformer(make_config()).eval()
    observations: list[tuple[torch.Size, torch.Tensor]] = []

    model(
        torch.tensor([[1, 2, 3]]),
        position_offset=5,
        attention_observer=lambda probabilities, positions: observations.append(
            (probabilities.shape, positions)
        ),
    )

    assert len(observations) == model.config.n_layers
    for shape, positions in observations:
        assert shape == (1, model.config.n_heads, 3, 3)
        assert torch.equal(positions, torch.tensor([5, 6, 7]))


@pytest.mark.parametrize(
    "caches",
    [
        [],
        [KVCache(max_length=16)],
        [object(), object()],
        (KVCache(max_length=16), KVCache(max_length=16)),
    ],
)
def test_transformer_rejects_invalid_caches(caches: object) -> None:
    model = DecoderOnlyTransformer(make_config())
    input_ids = torch.ones(1, 2, dtype=torch.long)

    with pytest.raises((TypeError, ValueError)):
        model(input_ids, caches=caches)
