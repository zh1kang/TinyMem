import pytest
import torch

from tinymem.model.config import ModelConfig
from tinymem.model.transformer import DecoderOnlyTransformer


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
