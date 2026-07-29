"""Standard data structures shared by controlled reasoning datasets."""

from dataclasses import dataclass


SUPPORTED_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class EvidenceFact:
    """One controlled fact and its exact location in source context."""

    fact_id: int
    text: str
    start_char: int
    end_char: int

    def __post_init__(self) -> None:
        if isinstance(self.fact_id, bool) or not isinstance(self.fact_id, int):
            raise TypeError("fact_id must be an integer")
        if self.fact_id <= 0:
            raise ValueError("fact_id must be positive")
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("text must be a nonempty string")
        for field_name in ("start_char", "end_char"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
        if self.start_char < 0 or self.end_char <= self.start_char:
            raise ValueError("evidence character span must be nonempty and nonnegative")


@dataclass(frozen=True)
class ReasoningExample:
    """One controlled question-answer example with source provenance."""

    dataset: str
    task_id: str
    split: str
    context: str
    question: str
    answer: str
    supporting_fact_ids: tuple[int, ...] | None
    source_length: int
    source_example_id: str
    context_fact_ids: tuple[int, ...] | None = None
    evidence_facts: tuple[EvidenceFact, ...] | None = None

    def __post_init__(self) -> None:
        required_strings = (
            "dataset",
            "task_id",
            "split",
            "context",
            "question",
            "answer",
            "source_example_id",
        )
        for field_name in required_strings:
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string")
            if not value.strip():
                raise ValueError(f"{field_name} must be nonempty")

        if self.split not in SUPPORTED_SPLITS:
            supported = ", ".join(SUPPORTED_SPLITS)
            raise ValueError(f"unsupported split {self.split!r}; choose from: {supported}")

        if isinstance(self.source_length, bool) or not isinstance(
            self.source_length, int
        ):
            raise TypeError("source_length must be an integer")
        if self.source_length != len(self.context):
            raise ValueError("source_length must equal the context character length")

        if self.supporting_fact_ids is not None:
            if not isinstance(self.supporting_fact_ids, tuple):
                raise TypeError("supporting_fact_ids must be a tuple or None")
            for fact_id in self.supporting_fact_ids:
                if isinstance(fact_id, bool) or not isinstance(fact_id, int):
                    raise TypeError("supporting_fact_ids must contain integers")
                if fact_id <= 0:
                    raise ValueError("supporting_fact_ids must be positive")
            if len(set(self.supporting_fact_ids)) != len(self.supporting_fact_ids):
                raise ValueError("supporting_fact_ids must be unique")

        if self.context_fact_ids is not None:
            if not isinstance(self.context_fact_ids, tuple):
                raise TypeError("context_fact_ids must be a tuple or None")
            for fact_id in self.context_fact_ids:
                if isinstance(fact_id, bool) or not isinstance(fact_id, int):
                    raise TypeError("context_fact_ids must contain integers")
                if fact_id <= 0:
                    raise ValueError("context_fact_ids must be positive")
            if len(set(self.context_fact_ids)) != len(self.context_fact_ids):
                raise ValueError("context_fact_ids must be unique")
            if len(self.context_fact_ids) != len(self.context.splitlines()):
                raise ValueError("context_fact_ids must align with context lines")

        if self.supporting_fact_ids is not None and self.context_fact_ids is not None:
            unavailable = set(self.supporting_fact_ids) - set(self.context_fact_ids)
            if unavailable:
                raise ValueError("supporting_fact_ids must refer to context facts")

        if self.evidence_facts is not None:
            if not isinstance(self.evidence_facts, tuple):
                raise TypeError("evidence_facts must be a tuple or None")
            previous_end = -1
            evidence_ids: set[int] = set()
            for evidence in self.evidence_facts:
                if not isinstance(evidence, EvidenceFact):
                    raise TypeError("evidence_facts must contain EvidenceFact values")
                if evidence.fact_id in evidence_ids:
                    raise ValueError("evidence fact IDs must be unique")
                if evidence.start_char < previous_end:
                    raise ValueError("evidence facts must be ordered and nonoverlapping")
                if evidence.end_char > len(self.context):
                    raise ValueError("evidence fact span exceeds context length")
                if self.context[evidence.start_char : evidence.end_char] != evidence.text:
                    raise ValueError("evidence fact text must match its context span")
                evidence_ids.add(evidence.fact_id)
                previous_end = evidence.end_char
            if self.supporting_fact_ids is not None:
                unavailable = set(self.supporting_fact_ids) - evidence_ids
                if unavailable:
                    raise ValueError("supporting_fact_ids must refer to evidence facts")
