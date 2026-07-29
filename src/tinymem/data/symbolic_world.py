"""Symbolic world-state replay for controlled reasoning tasks."""

from dataclasses import dataclass, field

from tinymem.data.schema import ReasoningExample


MOVEMENT_SEPARATORS = (
    " moved to the ",
    " went to the ",
    " went back to the ",
    " journeyed to the ",
    " travelled to the ",
)

ACQUISITION_SEPARATORS = (
    " picked up the ",
    " grabbed the ",
    " got the ",
    " took the ",
)

DROP_SEPARATORS = (
    " left the ",
    " discarded the ",
    " put down the ",
    " dropped the ",
)


@dataclass(frozen=True)
class SupportedValue:
    """A symbolic value together with the facts that establish it."""

    value: str
    supporting_fact_ids: tuple[int, ...]


@dataclass
class WorldState:
    """Mutable state produced by replaying facts in temporal order."""

    person_locations: dict[str, SupportedValue] = field(default_factory=dict)
    object_holders: dict[str, SupportedValue] = field(default_factory=dict)
    object_locations: dict[str, SupportedValue] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectAction:
    """One qa2 acquisition or drop event."""

    action: str
    person: str
    object_name: str
    fact_id: int


@dataclass(frozen=True)
class OracleResult:
    """The answer reconstructed by symbolic replay."""

    answer: str
    supporting_fact_ids: tuple[int, ...]


def iter_controlled_facts(example: ReasoningExample):
    """Yield controlled facts from bAbI lines or BABILong evidence spans."""
    if example.evidence_facts is not None:
        for evidence in example.evidence_facts:
            yield evidence.text, evidence.fact_id
        return
    if example.context_fact_ids is None:
        raise ValueError("symbolic interpretation requires context_fact_ids or evidence_facts")
    yield from zip(
        example.context.splitlines(),
        example.context_fact_ids,
        strict=True,
    )


def validate_fact_inputs(sentence: str, fact_id: int) -> None:
    """Validate the shared input contract for one controlled fact."""
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


def parse_qa1_movement(sentence: str, fact_id: int) -> tuple[str, SupportedValue]:
    """Parse one qa1 movement sentence into a person and new location."""
    validate_fact_inputs(sentence, fact_id)

    body = sentence[:-1]
    matches = tuple(
        separator for separator in MOVEMENT_SEPARATORS if separator in body
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


def parse_object_action(
    sentence: str,
    fact_id: int,
    *,
    acquisition_requires_there: bool,
) -> ObjectAction:
    """Parse one controlled object acquisition or drop event."""
    validate_fact_inputs(sentence, fact_id)
    if not isinstance(acquisition_requires_there, bool):
        raise TypeError("acquisition_requires_there must be a boolean")
    body = sentence[:-1]
    matches = tuple(
        (action, separator)
        for action, separators in (
            ("acquire", ACQUISITION_SEPARATORS),
            ("drop", DROP_SEPARATORS),
        )
        for separator in separators
        if separator in body
    )
    if len(matches) != 1:
        raise ValueError("sentence must contain exactly one supported object action")

    action, separator = matches[0]
    person, object_text = body.split(separator, maxsplit=1)
    person = person.strip()
    object_text = object_text.strip()
    has_there_suffix = object_text.endswith(" there")
    if has_there_suffix:
        object_text = object_text[: -len(" there")].strip()

    if not person:
        raise ValueError("object action must contain a person")
    if not object_text:
        raise ValueError("object action must contain an object")
    if action == "acquire" and acquisition_requires_there and not has_there_suffix:
        raise ValueError("qa2 acquisition must end with 'there.'")

    return ObjectAction(
        action=action,
        person=person,
        object_name=object_text,
        fact_id=fact_id,
    )


def parse_qa2_object_action(sentence: str, fact_id: int) -> ObjectAction:
    """Parse one qa2 object acquisition or drop event."""
    return parse_object_action(
        sentence,
        fact_id,
        acquisition_requires_there=True,
    )


def parse_qa2_question(question: str) -> str:
    """Extract the queried object from 'Where is the <object>?'"""
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    if not question:
        raise ValueError("question must be nonempty")
    if question != question.strip():
        raise ValueError("question must not contain surrounding whitespace")

    prefix = "Where is the "
    if not question.startswith(prefix) or not question.endswith("?"):
        raise ValueError("question must have the form 'Where is the <object>?'")
    object_name = question[len(prefix) : -1]
    if not object_name or object_name != object_name.strip():
        raise ValueError("question must contain an object")
    if "?" in object_name:
        raise ValueError("question must contain exactly one final question mark")
    return object_name


def interpret_qa1(example: ReasoningExample) -> OracleResult:
    """Replay a qa1 example and reconstruct its answer and provenance."""
    if not isinstance(example, ReasoningExample):
        raise TypeError("example must be a ReasoningExample")
    if example.task_id != "qa1":
        raise ValueError(f"interpret_qa1 does not support task {example.task_id!r}")
    state = WorldState()
    for sentence, fact_id in iter_controlled_facts(example):
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


def interpret_qa2(example: ReasoningExample) -> OracleResult:
    """Replay a qa2 example with possession and dropped-object locations."""
    if not isinstance(example, ReasoningExample):
        raise TypeError("example must be a ReasoningExample")
    if example.task_id != "qa2":
        raise ValueError(f"interpret_qa2 does not support task {example.task_id!r}")
    state = WorldState()
    for sentence, fact_id in iter_controlled_facts(example):
        if any(separator in sentence for separator in MOVEMENT_SEPARATORS):
            person, location = parse_qa1_movement(sentence, fact_id)
            state.person_locations[person] = location
            continue

        event = parse_qa2_object_action(sentence, fact_id)
        if event.action == "acquire":
            if event.object_name in state.object_holders:
                raise ValueError(f"object {event.object_name!r} is already held")
            ground_location = state.object_locations.get(event.object_name)
            person_location = state.person_locations.get(event.person)
            if (
                ground_location is not None
                and person_location is not None
                and ground_location.value != person_location.value
            ):
                raise ValueError(
                    f"object {event.object_name!r} is not at "
                    f"{event.person!r}'s location"
                )
            state.object_locations.pop(event.object_name, None)
            state.object_holders[event.object_name] = SupportedValue(
                value=event.person,
                supporting_fact_ids=(event.fact_id,),
            )
            continue

        holder = state.object_holders.get(event.object_name)
        if holder is None or holder.value != event.person:
            raise ValueError(
                f"person {event.person!r} cannot drop object {event.object_name!r}"
            )
        person_location = state.person_locations.get(event.person)
        if person_location is None:
            state.object_locations.pop(event.object_name, None)
        else:
            state.object_locations[event.object_name] = SupportedValue(
                value=person_location.value,
                supporting_fact_ids=(
                    event.fact_id,
                    *person_location.supporting_fact_ids,
                ),
            )
        del state.object_holders[event.object_name]

    object_name = parse_qa2_question(example.question)
    holder = state.object_holders.get(object_name)
    if holder is not None:
        holder_location = state.person_locations[holder.value]
        return OracleResult(
            answer=holder_location.value,
            supporting_fact_ids=(
                *holder.supporting_fact_ids,
                *holder_location.supporting_fact_ids,
            ),
        )

    object_location = state.object_locations.get(object_name)
    if object_location is None:
        raise ValueError(f"location is unknown for object {object_name!r}")
    return OracleResult(
        answer=object_location.value,
        supporting_fact_ids=object_location.supporting_fact_ids,
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
    if example.task_id == "qa1":
        result = interpret_qa1(example)
    elif example.task_id == "qa2":
        result = interpret_qa2(example)
    elif example.task_id == "qa3":
        from tinymem.data.symbolic_qa3 import interpret_qa3

        result = interpret_qa3(example)
    elif example.task_id == "qa4":
        from tinymem.data.symbolic_qa4 import interpret_qa4

        result = interpret_qa4(example)
    elif example.task_id == "qa5":
        from tinymem.data.symbolic_qa5 import interpret_qa5

        result = interpret_qa5(example)
    else:
        raise ValueError(f"unsupported symbolic task: {example.task_id!r}")

    validate_oracle_result(example, result)
    return result
