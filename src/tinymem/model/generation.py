"""Autoregressive decoding utilities for the TinyMem Transformer."""

import math
from numbers import Real

import torch

from tinymem.model.transformer import DecoderOnlyTransformer


@torch.no_grad()
def generate(
    model: DecoderOnlyTransformer,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Append tokens sampled from the model's next-token distribution."""

    if not isinstance(model, DecoderOnlyTransformer):
        raise TypeError(
            f"model must be a DecoderOnlyTransformer, got {type(model)}"
        )
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError(
            f"input_ids must be a torch.Tensor, got {type(input_ids)}"
        )
    if input_ids.ndim != 2:
        raise ValueError(
            f"input_ids must be a rank-two tensor, got shape {input_ids.shape}"
        )
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"input_ids must be an integer tensor, got dtype {input_ids.dtype}"
        )
    if input_ids.shape[1] == 0:
        raise ValueError("input_ids must contain at least one token")
    if (input_ids < 0).any() or (input_ids >= model.config.vocab_size).any():
        raise ValueError("input_ids contain a token ID outside the vocabulary")

    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise TypeError(
            f"max_new_tokens must be an int, got {type(max_new_tokens)}"
        )
    if max_new_tokens < 0:
        raise ValueError(
            f"max_new_tokens must be nonnegative, got {max_new_tokens}"
        )
    if isinstance(temperature, bool) or not isinstance(temperature, Real):
        raise TypeError(f"temperature must be a real number, got {type(temperature)}")
    if not math.isfinite(float(temperature)) or temperature < 0:
        raise ValueError(
            f"temperature must be finite and nonnegative, got {temperature}"
        )

    total_length = input_ids.shape[1] + max_new_tokens
    if total_length > model.config.max_local_tokens:
        raise ValueError(
            "prompt plus generated tokens must not exceed "
            f"max_local_tokens ({model.config.max_local_tokens})"
        )

    generated = input_ids.clone()
    if max_new_tokens == 0:
        return generated

    caches = model.create_caches()
    logits = model(generated, caches=caches)
    for step in range(max_new_tokens):
        next_logits = logits[:, -1, :]
        if temperature == 0:
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            probabilities = torch.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
                generator=generator,
            )
        generated = torch.cat((generated, next_token), dim=1)
        if step + 1 < max_new_tokens:
            logits = model(
                next_token,
                position_offset=caches[0].end_position,
                caches=caches,
            )

    return generated
