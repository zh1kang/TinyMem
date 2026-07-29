"""Symbolic replay for bAbI task 4 direct spatial relations."""

from dataclasses import dataclass

from tinymem.data.schema import ReasoningExample
from tinymem.data.symbolic_world import (
    OracleResult,
    SupportedValue,
    iter_controlled_facts,
    validate_fact_inputs,
)


INVERSE_RELATIONS = {
    "north": "south",
    "south": "north",
    "east": "west",
    "west": "east",
}


@dataclass(frozen=True)
class SpatialFact:
    subject: str
    relation: str
    reference: str
    fact_id: int


def parse_qa4_fact(sentence: str, fact_id: int) -> SpatialFact:
    """Parse 'The A is <relation> of the B.'"""
    validate_fact_inputs(sentence, fact_id)
    prefix = "The "
    if not sentence.startswith(prefix):
        raise ValueError("qa4 fact must start with 'The '")
    body = sentence[len(prefix) : -1]
    matches = tuple(
        (relation, f" is {relation} of the ")
        for relation in INVERSE_RELATIONS
        if f" is {relation} of the " in body
    )
    if len(matches) != 1:
        raise ValueError("qa4 fact must contain one supported spatial relation")
    relation, separator = matches[0]
    subject, reference = body.split(separator, maxsplit=1)
    if not subject or not reference:
        raise ValueError("qa4 fact must contain two places")
    return SpatialFact(subject, relation, reference, fact_id)


def parse_qa4_question(question: str) -> tuple[str, str]:
    """Return the relation and reference used by the spatial lookup."""
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    if not question or question != question.strip() or not question.endswith("?"):
        raise ValueError("question must be a complete qa4 question")

    body = question[:-1]
    for relation in INVERSE_RELATIONS:
        direct_prefix = f"What is {relation} of the "
        if body.startswith(direct_prefix):
            reference = body[len(direct_prefix) :]
            if reference:
                return relation, reference

        indirect_prefix = "What is the "
        indirect_suffix = f" {relation} of"
        if body.startswith(indirect_prefix) and body.endswith(indirect_suffix):
            subject = body[len(indirect_prefix) : -len(indirect_suffix)]
            if subject:
                return INVERSE_RELATIONS[relation], subject
    raise ValueError("unsupported qa4 question structure")


def interpret_qa4(example: ReasoningExample) -> OracleResult:
    """Replay a qa4 example and answer one direct relation query."""
    if example.task_id != "qa4":
        raise ValueError(f"interpret_qa4 does not support task {example.task_id!r}")
    relations: dict[tuple[str, str], SupportedValue] = {}
    for sentence, fact_id in iter_controlled_facts(example):
        fact = parse_qa4_fact(sentence, fact_id)
        relations[(fact.relation, fact.reference)] = SupportedValue(
            fact.subject,
            (fact.fact_id,),
        )
        relations[(INVERSE_RELATIONS[fact.relation], fact.subject)] = SupportedValue(
            fact.reference,
            (fact.fact_id,),
        )

    key = parse_qa4_question(example.question)
    answer = relations.get(key)
    if answer is None:
        raise ValueError(f"qa4 relation is unknown for query {example.question!r}")
    return OracleResult(answer.value, answer.supporting_fact_ids)
