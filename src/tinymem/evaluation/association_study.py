"""Apply the declared five-baseline rules to an eight-known, one-absent study."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from tinymem.evaluation.paired_bootstrap import PairedAccuracyInterval


ASSOCIATION_COMPARATORS = frozenset((
    "mean_pool", "recent_native", "recent_vocabulary", "latest_vocabulary", "latest_template",
))


@dataclass(frozen=True)
class AssociationStudyDesign:
    worlds: int
    seeds: tuple[int, int, int]
    bootstrap_resamples: int
    bootstrap_seed: int

    def __post_init__(self) -> None:
        for name, value, minimum in (("worlds", self.worlds, 2), ("bootstrap_resamples", self.bootstrap_resamples, 2),
                                     ("bootstrap_seed", self.bootstrap_seed, 0)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer at least {minimum}")
        if (not isinstance(self.seeds, tuple) or len(self.seeds) != 3
                or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in self.seeds)
                or len(set(self.seeds)) != 3):
            raise ValueError("seeds must be a tuple of three distinct nonnegative integers")


@dataclass(frozen=True)
class AssociationStudyAssessment:
    reader_qualified: bool
    known_result: Literal["superior", "inferior", "inconclusive"]
    minimum_known_point_gain: float
    absent_noninferiority: bool | None
    absent_observed_threshold: bool
    overall_result: Literal["positive", "reader_limited", "inferior", "inconclusive", "known_gain_abstention_unresolved"]
    practical_positive: bool


def assess_association_study(
    known: Mapping[str, PairedAccuracyInterval], absent: Mapping[str, PairedAccuracyInterval],
    *, design: AssociationStudyDesign, reader_known_correct: int, reader_absent_correct: int,
    writer_absent_correct: Mapping[int, int],
) -> AssociationStudyAssessment:
    """Keep superiority, observed effect size, and abstention safety distinct.

    Intervals are the declared 99% paired-world estimates, conditional on three
    checkpoints. Failure to establish superiority or noninferiority is not equality.
    """
    if known.keys() != ASSOCIATION_COMPARATORS or absent.keys() != ASSOCIATION_COMPARATORS:
        raise ValueError("all five declared comparators are required; references must stay separate")
    worlds, seeds = design.worlds, design.seeds
    for interval in (*known.values(), *absent.values()):
        if (isinstance(interval.histories, bool) or not isinstance(interval.histories, int)
                or not isinstance(interval.optimization_seeds, tuple)
                or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in interval.optimization_seeds)
                or isinstance(interval.resamples, bool) or not isinstance(interval.resamples, int)
                or isinstance(interval.bootstrap_seed, bool) or not isinstance(interval.bootstrap_seed, int)
                or interval.confidence != 0.99 or interval.histories != worlds or interval.optimization_seeds != seeds
                or interval.resamples != design.bootstrap_resamples or interval.bootstrap_seed != design.bootstrap_seed):
            raise ValueError("all intervals must use 99% confidence and the declared world, seed, and bootstrap design")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not -1 <= value <= 1
               for value in (interval.difference, interval.lower, interval.upper)):
            raise ValueError("accuracy differences and interval bounds must be finite fractions")
        if interval.lower > interval.upper:
            raise ValueError("interval bounds must be ordered")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in writer_absent_correct) or set(writer_absent_correct) != set(seeds):
        raise ValueError("absent counts must cover exactly the same optimization seeds")
    counts = [(reader_known_correct, 8 * worlds), (reader_absent_correct, worlds),
              *((count, worlds) for count in writer_absent_correct.values())]
    if any(isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= maximum for count, maximum in counts):
        raise ValueError("correct counts must be integers within their query totals")

    reader_qualified = reader_known_correct * 100 >= 95 * 8 * worlds and reader_absent_correct * 100 >= 95 * worlds
    known_result: Literal["superior", "inferior", "inconclusive"]
    if all(interval.lower > 0 for interval in known.values()):
        known_result = "superior"
    elif any(interval.upper < 0 for interval in known.values()):
        known_result = "inferior"
    else:
        known_result = "inconclusive"
    minimum_gain = min(interval.difference for interval in known.values())
    observed_absent = all(count * 100 >= 95 * worlds for count in writer_absent_correct.values())
    noninferiority = all(interval.lower > -0.05 for interval in absent.values()) if known_result == "superior" else None
    overall: Literal["positive", "reader_limited", "inferior", "inconclusive", "known_gain_abstention_unresolved"]
    if not reader_qualified:
        overall = "reader_limited"
    elif known_result != "superior":
        overall = known_result
    elif noninferiority and observed_absent:
        overall = "positive"
    else:
        overall = "known_gain_abstention_unresolved"
    return AssociationStudyAssessment(
        reader_qualified, known_result, minimum_gain, noninferiority, observed_absent,
        overall, overall == "positive" and minimum_gain >= 0.05,
    )
