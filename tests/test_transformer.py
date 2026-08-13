import pytest
import torch

from tinymem.model.config import ModelConfig
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
