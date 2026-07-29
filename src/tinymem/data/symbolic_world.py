"""Symbolic world-state replay for controlled reasoning tasks."""

from dataclasses import dataclass, field

from tinymem.data.schema import ReasoningExample


@dataclass(frozen=True)
class SupportedValue:
    """A symbolic value together with the facts that establish it."""

    value: str
    supporting_fact_ids: tuple[int, ...]


@dataclass
class WorldState:
    """Mutable state produced by replaying facts in temporal order."""

    person_locations: dict[str, SupportedValue] = field(default_factory=dict)


@dataclass(frozen=True)
class OracleResult:
    """The answer reconstructed by symbolic replay."""

    answer: str
    supporting_fact_ids: tuple[int, ...]


def parse_qa1_movement(sentence: str, fact_id: int) -> tuple[str, SupportedValue]:
    """Parse one qa1 movement sentence into a person and new location."""
    if not isinstance(sentence, str):
        raise TypeError("sentence must be a string")
    if not sentence:
        raise ValueError("sentence must be nonempty")
    if sentence != sentence.strip():
        raise ValueError("sentence must not contain surrounding whitespace")
    if not sentence.endswith("."):
        raise ValueError("sentence must end with a period")

    if isinstance(fact_id, bool) or not isinstance(fact_id, int):
        raise TypeError("fact_id must be an integer")
    if fact_id <= 0:
        raise ValueError("fact_id must be positive")

    body = sentence[:-1]
    movement_separators = (
        " moved to the ",
        " went to the ",
        " went back to the ",
        " journeyed to the ",
        " travelled to the ",
    )
    matches = tuple(
        separator for separator in movement_separators if separator in body
    )
    if len(matches) != 1:
        raise ValueError("sentence must contain exactly one supported movement")

    person, destination = body.split(matches[0], maxsplit=1)
    person = person.strip()
    destination = destination.strip()
    if not person:
        raise ValueError("movement sentence must contain a person")
    if not destination:
        raise ValueError("movement sentence must contain a destination")

    return person, SupportedValue(
        value=destination,
        supporting_fact_ids=(fact_id,),
    )


def parse_qa1_question(question: str) -> str:
    """Extract the queried person from 'Where is <person>?'"""
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    if not question:
        raise ValueError("question must be nonempty")
    if question != question.strip():
        raise ValueError("question must not contain surrounding whitespace")

    prefix = "Where is "
    if not question.startswith(prefix) or not question.endswith("?"):
        raise ValueError("question must have the form 'Where is <person>?'")

    person = question[len(prefix) : -1]
    if not person or person != person.strip():
        raise ValueError("question must contain a person")
    if "?" in person:
        raise ValueError("question must contain exactly one final question mark")
    return person


def interpret_qa1(example: ReasoningExample) -> OracleResult:
    """Replay a qa1 example and reconstruct its answer and provenance."""
    if not isinstance(example, ReasoningExample):
        raise TypeError("example must be a ReasoningExample")
    if example.task_id != "qa1":
        raise ValueError(f"interpret_qa1 does not support task {example.task_id!r}")
    if example.context_fact_ids is None:
        raise ValueError("qa1 interpretation requires context_fact_ids")

    state = WorldState()
    for sentence, fact_id in zip(
        example.context.splitlines(),
        example.context_fact_ids,
        strict=True,
    ):
        person, location = parse_qa1_movement(sentence, fact_id)
        state.person_locations[person] = location

    queried_person = parse_qa1_question(example.question)
    if queried_person not in state.person_locations:
        raise ValueError(f"location is unknown for person {queried_person!r}")

    location = state.person_locations[queried_person]
    return OracleResult(
        answer=location.value,
        supporting_fact_ids=location.supporting_fact_ids,
    )


def validate_oracle_result(
    example: ReasoningExample,
    result: OracleResult,
) -> None:
    """Require symbolic answer and provenance to match published labels."""
    if not isinstance(example, ReasoningExample):
        raise TypeError("example must be a ReasoningExample")
    if not isinstance(result, OracleResult):
        raise TypeError("result must be an OracleResult")

    if result.answer != example.answer:
        raise ValueError(
            f"oracle answer {result.answer!r} does not match published answer "
            f"{example.answer!r} for {example.source_example_id}"
        )
    if (
        example.supporting_fact_ids is not None
        and result.supporting_fact_ids != example.supporting_fact_ids
    ):
        raise ValueError(
            f"oracle support {result.supporting_fact_ids!r} does not match "
            f"published support {example.supporting_fact_ids!r} for "
            f"{example.source_example_id}"
        )


def interpret(example: ReasoningExample) -> OracleResult:
    """Dispatch a supported task to its symbolic interpreter."""
    if not isinstance(example, ReasoningExample):
        raise TypeError("example must be a ReasoningExample")
    if example.task_id != "qa1":
        raise ValueError(f"unsupported symbolic task: {example.task_id!r}")

    result = interpret_qa1(example)
    validate_oracle_result(example, result)
    return result
