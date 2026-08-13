import pytest
import torch

from tinymem.model.config import ModelConfig
from tinymem.model.generation import generate
from tinymem.model.transformer import DecoderOnlyTransformer


def make_model() -> DecoderOnlyTransformer:
    return DecoderOnlyTransformer(
        ModelConfig(
            vocab_size=16,
            d_model=8,
            n_layers=1,
            n_heads=2,
            d_ff=16,
            max_local_tokens=8,
        )
    ).eval()


def test_greedy_generation_is_deterministic() -> None:
    torch.manual_seed(3)
    model = make_model()
    prompt = torch.tensor([[1, 2, 3]])

    first = generate(model, prompt, max_new_tokens=3)
    second = generate(model, prompt, max_new_tokens=3)

    assert torch.equal(first, second)
    assert first.shape == (1, 6)
    assert torch.equal(first[:, :3], prompt)


def test_zero_token_generation_returns_prompt_copy() -> None:
    model = make_model()
    prompt = torch.tensor([[1, 2]])

    result = generate(model, prompt, max_new_tokens=0)

    assert torch.equal(result, prompt)
    assert result is not prompt


def test_temperature_sampling_accepts_generator() -> None:
    torch.manual_seed(4)
    model = make_model()
    prompt = torch.tensor([[1, 2]])
    generator = torch.Generator().manual_seed(10)

    result = generate(
        model,
        prompt,
        max_new_tokens=2,
        temperature=0.8,
        generator=generator,
    )

    assert result.shape == (1, 4)
    assert torch.equal(result[:, :2], prompt)


@pytest.mark.parametrize(
    ("argument", "value"),
    [("max_new_tokens", True), ("max_new_tokens", -1), ("temperature", -0.1)],
)
def test_generation_rejects_invalid_sampling_arguments(
    argument: str,
    value: object,
) -> None:
    model = make_model()
    prompt = torch.tensor([[1, 2]])

    with pytest.raises((TypeError, ValueError)):
        generate(model, prompt, **{argument: value})


def test_generation_rejects_prompt_beyond_local_context() -> None:
    model = make_model()
    prompt = torch.tensor([[1, 2, 3, 4, 5, 6]])

    with pytest.raises(ValueError, match="max_local_tokens"):
        generate(model, prompt, max_new_tokens=3)
