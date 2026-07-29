"""Symbolic replay for bAbI task 3 object-location histories."""

from dataclasses import dataclass, field

from tinymem.data.schema import ReasoningExample
from tinymem.data.symbolic_world import (
    MOVEMENT_SEPARATORS,
    OracleResult,
    SupportedValue,
    iter_controlled_facts,
    parse_object_action,
    parse_qa1_movement,
)


@dataclass
class HistoryState:
    person_locations: dict[str, SupportedValue] = field(default_factory=dict)
    object_holders: dict[str, SupportedValue] = field(default_factory=dict)
    object_histories: dict[str, list[SupportedValue]] = field(default_factory=dict)
    object_actions: dict[str, int] = field(default_factory=dict)


def parse_qa3_question(question: str) -> tuple[str, str]:
    """Extract object and target location from a qa3 history question."""
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    if not question or question != question.strip():
        raise ValueError("question must be nonempty without surrounding whitespace")

    prefix = "Where was the "
    separator = " before the "
    if not question.startswith(prefix) or not question.endswith("?"):
        raise ValueError(
            "question must have the form "
            "'Where was the <object> before the <location>?'"
        )
    body = question[len(prefix) : -1]
    if body.count(separator) != 1:
        raise ValueError("qa3 question must contain one 'before the' relation")
    object_name, target_location = body.split(separator, maxsplit=1)
    if not object_name or not target_location:
        raise ValueError("qa3 question must contain an object and target location")
    return object_name, target_location


def append_location(
    state: HistoryState,
    object_name: str,
    location: SupportedValue,
) -> None:
    """Append only actual location changes to an object's history."""
    history = state.object_histories.setdefault(object_name, [])
    if not history or history[-1].value != location.value:
        history.append(location)


def interpret_qa3(example: ReasoningExample) -> OracleResult:
    """Replay a qa3 example and answer one historical-location query."""
    if not isinstance(example, ReasoningExample):
        raise TypeError("example must be a ReasoningExample")
    if example.task_id != "qa3":
        raise ValueError(f"interpret_qa3 does not support task {example.task_id!r}")
    state = HistoryState()
    for sentence, fact_id in iter_controlled_facts(example):
        if any(separator in sentence for separator in MOVEMENT_SEPARATORS):
            person, location = parse_qa1_movement(sentence, fact_id)
            state.person_locations[person] = location
            for object_name, holder in state.object_holders.items():
                if holder.value == person:
                    append_location(state, object_name, location)
            continue

        event = parse_object_action(
            sentence,
            fact_id,
            acquisition_requires_there=False,
        )
        state.object_actions[event.object_name] = event.fact_id
        if event.action == "acquire":
            if event.object_name in state.object_holders:
                raise ValueError(f"object {event.object_name!r} is already held")
            state.object_holders[event.object_name] = SupportedValue(
                event.person,
                (event.fact_id,),
            )
            person_location = state.person_locations.get(event.person)
            if person_location is not None:
                append_location(state, event.object_name, person_location)
            continue

        holder = state.object_holders.get(event.object_name)
        if holder is None or holder.value != event.person:
            raise ValueError(
                f"person {event.person!r} cannot drop object {event.object_name!r}"
            )
        person_location = state.person_locations.get(event.person)
        if person_location is not None:
            append_location(state, event.object_name, person_location)
        del state.object_holders[event.object_name]

    object_name, target_location = parse_qa3_question(example.question)
    history = state.object_histories.get(object_name, [])
    target_index = next(
        (index for index in range(len(history) - 1, -1, -1) if history[index].value == target_location),
        None,
    )
    if target_index is None or target_index == 0:
        raise ValueError(
            f"no prior location is known for object {object_name!r} before "
            f"{target_location!r}"
        )
    action_fact_id = state.object_actions.get(object_name)
    if action_fact_id is None:
        raise ValueError(f"no object action is known for {object_name!r}")

    target = history[target_index]
    previous = history[target_index - 1]
    return OracleResult(
        answer=previous.value,
        supporting_fact_ids=(
            action_fact_id,
            *target.supporting_fact_ids,
            *previous.supporting_fact_ids,
        ),
    )
    iter_controlled_facts,
