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
    assignments: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Return negative aggregate entropy to discourage codebook collapse."""
    if not isinstance(assignments, torch.Tensor):
        raise TypeError("assignments must be a torch.Tensor")
    if not isinstance(valid, torch.Tensor):
        raise TypeError("valid must be a torch.Tensor")
    if assignments.ndim != 3:
        raise ValueError(
            "assignments must have shape [batch, assignments, codebook_size]"
        )
    if assignments.shape[0] == 0 or assignments.shape[1] == 0:
        raise ValueError("assignments must contain rows and assignments")
    if assignments.shape[2] <= 1:
        raise ValueError("assignments must contain more than one code")
    if not assignments.is_floating_point():
        raise TypeError("assignments must be floating point")
    if valid.shape != assignments.shape[:2]:
        raise ValueError(f"valid must have shape {assignments.shape[:2]}")
    if valid.dtype != torch.bool:
        raise TypeError("valid must be a boolean tensor")
    if valid.device != assignments.device:
        raise ValueError("valid and assignments must share a device")
    if not valid.any():
        raise ValueError("usage loss requires a valid assignment")

    expanded_valid = valid.unsqueeze(-1).to(dtype=assignments.dtype)
    mean_probability = (
        assignments * expanded_valid
    ).sum(dim=(0, 1)) / expanded_valid.sum()
    tiny = torch.finfo(mean_probability.dtype).tiny
    return (
        mean_probability * mean_probability.clamp_min(tiny).log()
    ).sum()
