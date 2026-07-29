"""Validation helpers for controlled long-context examples."""

from dataclasses import dataclass

from tinymem.data.schema import ReasoningExample
from tinymem.data.symbolic_world import OracleResult


@dataclass(frozen=True)
class EvidenceDelay:
    """Distance from the last answer-bearing fact to the end of context."""

    characters: int
    trailing_evidence_facts: int


def measure_evidence_delay(
    example: ReasoningExample,
    result: OracleResult,
) -> EvidenceDelay:
    """Measure query delay from exact BABILong evidence spans."""
    if example.evidence_facts is None:
        raise ValueError("evidence delay requires exact evidence_facts")
    by_id = {fact.fact_id: fact for fact in example.evidence_facts}
    try:
        support = tuple(by_id[fact_id] for fact_id in result.supporting_fact_ids)
    except KeyError as error:
        raise ValueError(f"oracle support contains unknown fact ID {error.args[0]}") from error
    if not support:
        raise ValueError("evidence delay requires at least one supporting fact")

    latest_end = max(fact.end_char for fact in support)
    latest_index = max(
        index
        for index, fact in enumerate(example.evidence_facts)
        if fact.fact_id in result.supporting_fact_ids
    )
    return EvidenceDelay(
        characters=len(example.context) - latest_end,
        trailing_evidence_facts=len(example.evidence_facts) - latest_index - 1,
    )


def require_evidence_outside_local_window(
    example: ReasoningExample,
    result: OracleResult,
    *,
    local_window_characters: int,
) -> EvidenceDelay:
    """Reject an example whose newest support remains inside a local window."""
    if isinstance(local_window_characters, bool) or not isinstance(
        local_window_characters, int
    ):
        raise TypeError("local_window_characters must be an integer")
    if local_window_characters <= 0:
        raise ValueError("local_window_characters must be positive")
    delay = measure_evidence_delay(example, result)
    if delay.characters < local_window_characters:
        raise ValueError("answer-bearing evidence is inside the local window")
    return delay
