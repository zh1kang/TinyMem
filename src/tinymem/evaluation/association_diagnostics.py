"""Timing, update, and answer-pattern diagnostics for controlled associations."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tinymem.data.reader_gate import ReaderCase
from tinymem.data.symbolic_world import parse_qa1_movement, parse_qa1_question
from tinymem.evaluation.longmemeval import normalized_answer


@dataclass(frozen=True)
class AssociationHistoryLabel:
    first_mention_chunk: int | None
    last_mention_chunk: int | None
    movement_count: int
    location_changes: int


@dataclass(frozen=True)
class AssociationAnswerPattern:
    queries: int
    correct: int
    distinct_predictions: int
    distinct_answers: int
    constant_answer_ceiling_correct: int


def summarize_association_answers(
    cases: Sequence[ReaderCase], predictions: Mapping[str, str],
) -> AssociationAnswerPattern:
    """Describe known-query answers for one world, not its causal mechanism.

    The constant-answer ceiling uses reference labels in hindsight.
    A score below this ceiling does not establish query blindness.
    """
    if not cases or any(not case.case_id for case in cases):
        raise ValueError("cases must have nonempty identifiers")
    identifiers = {case.case_id for case in cases}
    if len(identifiers) != len(cases) or set(predictions) != identifiers:
        raise ValueError("predictions must match unique case identifiers exactly")
    if len({(case.history_id, case.context) for case in cases}) != 1:
        raise ValueError("queries must belong to one shared history")
    if len({case.question for case in cases}) != len(cases):
        raise ValueError("questions must be distinct")
    if any(case.category != "opaque_qa1_known" for case in cases):
        raise ValueError("answer patterns require known-entity queries only")
    answers = [normalized_answer(case.answer) for case in cases]
    if any(answer in ("", "unknown") for answer in answers):
        raise ValueError("known-entity references must be nonempty known answers")
    values = [normalized_answer(predictions[case.case_id]) for case in cases]
    counts = Counter(answers)
    return AssociationAnswerPattern(
        len(cases), sum(value == answer for value, answer in zip(values, answers, strict=True)),
        len(set(values)), len(counts), max(counts.values()),
    )


def label_association_history(
    chunks: Sequence[str], cases: Sequence[ReaderCase],
) -> dict[str, AssociationHistoryLabel]:
    """Use one-based chunk indices; repeated destinations are not location changes.

    Last mention is a timing label, not proof that earlier evidence is insufficient.
    No model predictions enter these labels.
    """
    if not chunks or any(not isinstance(chunk, str) or not chunk.strip() for chunk in chunks):
        raise ValueError("chunks must be nonempty strings")
    if not cases or any(not case.case_id for case in cases):
        raise ValueError("cases must have nonempty identifiers")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("case identifiers must be unique")
    context = "\n".join(chunks)
    if any(case.context != context for case in cases):
        raise ValueError("chunks must reconstruct every query's exact context")

    entities: dict[str, tuple[str, AssociationHistoryLabel]] = {}
    fact_id = 0
    for chunk_index, chunk in enumerate(chunks, 1):
        for sentence in chunk.splitlines():
            fact_id += 1
            person, location = parse_qa1_movement(sentence, fact_id)
            previous = entities.get(person)
            if previous is None:
                label = AssociationHistoryLabel(chunk_index, chunk_index, 1, 0)
            else:
                old_location, old_label = previous
                label = AssociationHistoryLabel(
                    old_label.first_mention_chunk, chunk_index, old_label.movement_count + 1,
                    old_label.location_changes + int(old_location != location.value),
                )
            entities[person] = location.value, label

    labels = {}
    for case in cases:
        person = parse_qa1_question(case.question)
        answer, label = entities.get(person, ("unknown", AssociationHistoryLabel(None, None, 0, 0)))
        if case.answer != answer:
            raise ValueError("query answer disagrees with symbolic history replay")
        labels[case.case_id] = label
    return labels
