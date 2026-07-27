"""Deterministic random-number initialization."""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed every random-number generator used by TinyMem."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("Seed must be an integer and cannot be a boolean.")
    if seed < 0:
        raise ValueError("Seed must be a nonnegative integer.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
