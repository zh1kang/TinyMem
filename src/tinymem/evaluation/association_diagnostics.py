"""History-only timing and update labels for controlled association questions."""

from collections.abc import Sequence
from dataclasses import dataclass

from tinymem.data.reader_gate import ReaderCase
from tinymem.data.symbolic_world import parse_qa1_movement, parse_qa1_question


@dataclass(frozen=True)
class AssociationHistoryLabel:
    first_mention_chunk: int | None
    last_mention_chunk: int | None
    movement_count: int
    location_changes: int


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
