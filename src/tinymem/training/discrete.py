"""Training objectives and schedules for discrete memory."""

from dataclasses import dataclass
import math
from numbers import Integral, Real

import torch


@dataclass(frozen=True)
class GumbelTemperatureSchedule:
    """Linearly anneal a positive Gumbel-Softmax temperature."""

    start: float
    end: float
    anneal_steps: int

    def __post_init__(self) -> None:
        for name, value in (("start", self.start), ("end", self.end)):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if isinstance(self.anneal_steps, bool) or not isinstance(
            self.anneal_steps,
            Integral,
        ):
            raise TypeError("anneal_steps must be an integer")
        if self.anneal_steps <= 0:
            raise ValueError("anneal_steps must be positive")

    def value(self, step: int) -> float:
        """Return the schedule value at a zero-based optimization step."""
        if isinstance(step, bool) or not isinstance(step, Integral):
            raise TypeError("step must be an integer")
        if step < 0:
            raise ValueError("step must be nonnegative")
        fraction = min(int(step) / self.anneal_steps, 1.0)
        return float(self.start + fraction * (self.end - self.start))


def codebook_usage_loss(
    probabilities: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Return negative aggregate entropy to discourage codebook collapse."""
    if not isinstance(probabilities, torch.Tensor):
        raise TypeError("probabilities must be a torch.Tensor")
    if not isinstance(valid, torch.Tensor):
        raise TypeError("valid must be a torch.Tensor")
    if probabilities.ndim != 3:
        raise ValueError(
            "probabilities must have shape [batch, assignments, codebook_size]"
        )
    if probabilities.shape[0] == 0 or probabilities.shape[1] == 0:
        raise ValueError("probabilities must contain rows and assignments")
    if probabilities.shape[2] <= 1:
        raise ValueError("probabilities must contain more than one code")
    if not probabilities.is_floating_point():
        raise TypeError("probabilities must be floating point")
    if valid.shape != probabilities.shape[:2]:
        raise ValueError(f"valid must have shape {probabilities.shape[:2]}")
    if valid.dtype != torch.bool:
        raise TypeError("valid must be a boolean tensor")
    if valid.device != probabilities.device:
        raise ValueError("valid and probabilities must share a device")
    if not valid.any():
        raise ValueError("usage loss requires a valid assignment")

    expanded_valid = valid.unsqueeze(-1).to(dtype=probabilities.dtype)
    mean_probability = (
        probabilities * expanded_valid
    ).sum(dim=(0, 1)) / expanded_valid.sum()
    tiny = torch.finfo(mean_probability.dtype).tiny
    return (
        mean_probability * mean_probability.clamp_min(tiny).log()
    ).sum()
