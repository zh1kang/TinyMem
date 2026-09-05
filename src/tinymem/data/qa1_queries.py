"""Multiple known and missing-entity questions about one existing qa1 history."""

from collections.abc import Sequence

from tinymem.data.reader_gate import ReaderCase
from tinymem.data.symbolic_world import parse_qa1_movement, parse_qa1_question


def qa1_world_queries(case: ReaderCase, people: Sequence[str]) -> tuple[ReaderCase, ...]:
    """Replay movements without changing the source context or split identity."""
    if case.category != "babi_qa1" or not case.context:
        raise ValueError("a nonempty babi_qa1 history is required")
    if not people or isinstance(people, str):
        raise ValueError("people must be a nonempty sequence of distinct names")
    for person in people:
        if not isinstance(person, str) or not person or person != person.strip() or any(char in person for char in "?\n\r"):
            raise ValueError("each person must be a nonempty name without query delimiters")
    if len(set(people)) != len(people):
        raise ValueError("people must be distinct")
    world = {}
    for fact_id, sentence in enumerate(case.context.splitlines(), start=1):
        person, location = parse_qa1_movement(sentence, fact_id)
        world[person] = location.value
    if world.get(parse_qa1_question(case.question), "unknown") != case.answer:
        raise ValueError("source answer disagrees with symbolic history replay")
    return tuple(ReaderCase(
        f"{case.case_id}:world-query:{person}",
        "babi_qa1" if person in world else "missing_entity",
        case.history_id, case.context, f"Where is {person}?", world.get(person, "unknown"),
    ) for person in people)
