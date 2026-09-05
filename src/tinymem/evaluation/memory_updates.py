"""Paired update measurements for one history and one fixed checkpoint."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from tinymem.data.memory_updates import UpdateEpisode, validate_update_episode
from tinymem.data.opaque_qa1 import ROOMS
from tinymem.data.reader_gate import ReaderCase
from tinymem.evaluation.longmemeval import normalized_answer


@dataclass(frozen=True)
class UpdatePrediction:
    """One raw output; the validated dataset, not this record, supplies gold."""

    case_id: str
    prediction: str

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id or self.case_id != self.case_id.strip():
            raise ValueError("prediction case_id must be a nonempty stripped string")
        if not isinstance(self.prediction, str):
            raise TypeError("prediction must be a string; empty output is allowed and scored as an error")


@dataclass(frozen=True)
class CountRate:
    """Retain the original counts, including an explicitly undefined 0/0."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if type(self.numerator) is not int or type(self.denominator) is not int:
            raise TypeError("rate counts must be integers, not booleans or floats")
        if not 0 <= self.numerator <= self.denominator:
            raise ValueError("rate requires 0 <= numerator <= denominator")

    @property
    def value(self) -> float | None:
        return self.numerator / self.denominator if self.denominator else None

    def to_dict(self) -> dict[str, int | float | None]:
        return {"numerator": self.numerator, "denominator": self.denominator, "value": self.value}


@dataclass(frozen=True)
class UpdateMetrics:
    episode_id: str
    source_group_ids: tuple[str, str]
    rates: Mapping[str, CountRate]
    transitions: Mapping[str, Mapping[str, int]]

    def __post_init__(self) -> None:
        # Aggregation must not overwrite a history's evidence via an alias.
        object.__setattr__(self, "rates", MappingProxyType(dict(self.rates)))
        object.__setattr__(self, "transitions", MappingProxyType({
            name: MappingProxyType(dict(counts)) for name, counts in self.transitions.items()
        }))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "memory_update_metrics_v1",
            "episode_id": self.episode_id,
            "source_group_ids": list(self.source_group_ids),
            "rates": {name: rate.to_dict() for name, rate in self.rates.items()},
            "transitions": {name: dict(counts) for name, counts in self.transitions.items()},
        }


def _rate(values: Iterable[bool]) -> CountRate:
    flags = tuple(values)
    return CountRate(sum(flags), len(flags))


def _state_rates(cases: Sequence[ReaderCase], predictions: dict[str, str]) -> dict[str, CountRate]:
    values = [predictions[case.case_id] for case in cases]
    correct = [value == case.answer for value, case in zip(values, cases, strict=True)]
    known = [i for i, case in enumerate(cases) if case.answer != "unknown"]
    absent = [i for i, case in enumerate(cases) if case.answer == "unknown"]
    return {
        "accuracy": _rate(correct),
        "known_accuracy": _rate(correct[i] for i in known),
        "absent_accuracy": _rate(correct[i] for i in absent),
        "known_false_abstention": _rate(values[i] == "unknown" for i in known),
        "absent_error": _rate(values[i] != "unknown" for i in absent),
        "absent_false_answer": _rate(values[i] in ROOMS for i in absent),
        "absent_invalid_output": _rate(values[i] not in (*ROOMS, "unknown") for i in absent),
        "invalid_output": _rate(value not in (*ROOMS, "unknown") for value in values),
    }


def _transition_rates(before: Sequence[bool], after: Sequence[bool]) -> tuple[dict[str, CountRate], dict[str, int]]:
    pairs = tuple(zip(before, after, strict=True))
    cc = sum(old and new for old, new in pairs)
    cw = sum(old and not new for old, new in pairs)
    wc = sum(not old and new for old, new in pairs)
    ww = sum(not old and not new for old, new in pairs)
    n = len(pairs)
    return {
        "before_accuracy": CountRate(cc + cw, n),
        "after_accuracy": CountRate(cc + wc, n),
        "joint_preservation": CountRate(cc, n),
        "conditional_preservation": CountRate(cc, cc + cw),
        "conditional_forgetting": CountRate(cw, cc + cw),
        "unconditional_forgetting": CountRate(cw, n),
    }, {"CC": cc, "CW": cw, "WC": wc, "WW": ww}


def score_update_episode(episode: UpdateEpisode, predictions: Sequence[UpdatePrediction]) -> UpdateMetrics:
    """Score all four states, aligned by case ID, with one shared before sample.

    Exactly 40 records are required; duplicates, partial runs, and foreign history
    or stage IDs fail. This is one checkpoint's evidence, not a multi-seed report.
    Conditional denominators differ across histories and methods: within each
    optimization seed, pool history counts before division, not history rates.
    Branch/cohort metrics reuse observations: keep their keys separate, never
    pool them as independent samples (including the repeated before measures).
    """
    validate_update_episode(episode)
    if not isinstance(predictions, Sequence) or isinstance(predictions, (str, bytes)):
        raise TypeError("predictions must be a sequence of UpdatePrediction records")
    if any(not isinstance(row, UpdatePrediction) for row in predictions):
        raise TypeError("predictions must contain UpdatePrediction records")
    stages = {"before": episode.before, **{branch.kind: branch.queries for branch in episode.branches}}
    expected = {case.case_id for cases in stages.values() for case in cases}
    ids = [row.case_id for row in predictions]
    if len(ids) != len(expected) or len(set(ids)) != len(ids) or set(ids) != expected:
        raise ValueError("predictions must cover each of the 40 history/stage case IDs exactly once")
    values = {row.case_id: normalized_answer(row.prediction) for row in predictions}
    rates = {f"{stage}.{name}": rate for stage, cases in stages.items()
             for name, rate in _state_rates(cases, values).items()}
    transitions = {}
    old_answers = [case.answer for case in episode.before]
    old_correct = [values[case.case_id] == case.answer for case in episode.before]
    for branch in episode.branches:
        answers = [case.answer for case in branch.queries]
        correct = [values[case.case_id] == case.answer for case in branch.queries]
        target = episode.entities.index(branch.target)
        target_value = values[branch.queries[target].case_id]
        changed = old_answers[target] != answers[target]
        has_stale_value = changed and old_answers[target] != "unknown"
        target_rates = {
            "before_accuracy": CountRate(int(old_correct[target]), 1),
            "after_accuracy": CountRate(int(correct[target]), 1),
            "update_accuracy": CountRate(int(changed and correct[target]), int(changed)),
            "stale_answer": CountRate(int(has_stale_value and target_value == old_answers[target]), int(has_stale_value)),
        }
        rates.update({f"{branch.kind}.target.{name}": rate for name, rate in target_rates.items()})
        unchanged = [i for i, old in enumerate(old_answers) if old != "unknown" and old == answers[i]]
        # Repetition directly refreshes its target. Keep it out of the collateral
        # (untouched) cohort as well as reporting all unchanged known bindings.
        cohorts = {"unchanged_known": unchanged, "untouched_known": [i for i in unchanged if i != target]}
        for cohort, indices in cohorts.items():
            prefix = f"{branch.kind}.{cohort}"
            paired, counts = _transition_rates([old_correct[i] for i in indices], [correct[i] for i in indices])
            rates.update({f"{prefix}.{name}": rate for name, rate in paired.items()})
            transitions[prefix] = counts
    return UpdateMetrics(episode.episode_id, episode.source_group_ids, rates, transitions)
