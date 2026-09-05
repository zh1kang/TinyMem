"""Paired history-cluster uncertainty, conditional on the trained checkpoints."""

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import fmean


@dataclass(frozen=True)
class PairedAccuracyInterval:
    difference: float
    lower: float
    upper: float
    confidence: float
    histories: int
    optimization_seeds: tuple[int, ...]
    resamples: int
    bootstrap_seed: int


def paired_history_interval(
    left: Mapping[str, Mapping[int, float]], right: Mapping[str, Mapping[int, float]],
    *, confidence: float = 0.95, resamples: int = 10000, bootstrap_seed: int = 20260904,
) -> PairedAccuracyInterval:
    """Resample paired histories, not correlated questions or seed replicates.

    Each value is one checkpoint's mean accuracy for all queries in a history.
    Seed IDs must align across methods; deterministic baselines can repeat scores.
    The interval covers history sampling only, not optimization-seed uncertainty.
    """
    if len(left) < 2 or left.keys() != right.keys():
        raise ValueError("at least two matching history groups are required")
    if any(not isinstance(key, str) or not key for key in left):
        raise ValueError("history IDs must be nonempty strings")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 2:
        raise ValueError("resamples must be an integer of at least two")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int) or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    histories = sorted(left)
    seed_ids = set(left[histories[0]])
    if not seed_ids or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seed_ids):
        raise ValueError("optimization seeds must be nonnegative integers")
    seeds = tuple(sorted(seed_ids))
    differences = []
    for history in histories:
        for scores in (left[history], right[history]):
            if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in scores):
                raise ValueError("optimization seeds must be nonnegative integers")
        if set(left[history]) != seed_ids or set(right[history]) != seed_ids:
            raise ValueError("optimization seeds must match for every history and method")
        for scores in (left[history], right[history]):
            if any(not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1 for value in scores.values()):
                raise ValueError("accuracy fractions must be finite and between zero and one")
        differences.append(fmean(left[history][seed] - right[history][seed] for seed in seeds))
    rng = random.Random(bootstrap_seed)
    sampled = sorted(fmean(rng.choices(differences, k=len(histories))) for _ in range(resamples))

    def percentile(probability: float) -> float:
        index = probability * (resamples - 1)
        lower = math.floor(index)
        upper = math.ceil(index)
        return sampled[lower] + (sampled[upper] - sampled[lower]) * (index - lower)

    tail = (1 - confidence) / 2
    return PairedAccuracyInterval(
        fmean(differences), percentile(tail), percentile(1 - tail), float(confidence),
        len(histories), seeds, resamples, bootstrap_seed,
    )
