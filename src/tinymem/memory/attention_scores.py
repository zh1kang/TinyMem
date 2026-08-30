"""Attention-based importance scores for bounded memory policies."""

import torch


def attention_received(attention_prob: torch.Tensor) -> torch.Tensor:
    """Aggregate pre-dropout attention into one score per key.

    Input shape:  [batch, heads, queries, keys]
    Output shape: [batch, keys]
    """
    if not isinstance(attention_prob, torch.Tensor):
        raise TypeError("attention_prob must be a torch.Tensor")
    if attention_prob.ndim != 4:
        raise ValueError(
            "attention_prob must have shape [batch, heads, queries, keys]"
        )
    if not attention_prob.is_floating_point():
        raise TypeError("attention_prob must be a floating-point tensor")
    if any(size == 0 for size in attention_prob.shape):
        raise ValueError("attention_prob dimensions must be nonempty")
    if not torch.isfinite(attention_prob).all():
        raise ValueError("attention_prob must contain only finite values")
    if (attention_prob < 0).any():
        raise ValueError("attention_prob must be nonnegative")

    probability_mass = attention_prob.sum(dim=-1)
    if not torch.allclose(
        probability_mass,
        torch.ones_like(probability_mass),
        rtol=1e-4,
        atol=1e-6,
    ):
        raise ValueError("attention_prob must sum to one across keys")

    probabilities = attention_prob.to(torch.float32)
    head_average = probabilities.mean(dim=1)
    return head_average.sum(dim=1)
