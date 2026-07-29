import pytest

from tinymem.data.schema import ReasoningExample
from tinymem.data.symbolic_world import (
    ObjectAction,
    OracleResult,
    SupportedValue,
    interpret,
    interpret_qa1,
    interpret_qa2,
    parse_qa1_movement,
    parse_qa1_question,
    parse_qa2_object_action,
    parse_qa2_question,
    validate_oracle_result,
)


def make_qa1_example(**overrides: object) -> ReasoningExample:
    values: dict[str, object] = {
        "dataset": "babi",
        "task_id": "qa1",
        "split": "test",
        "context": (
            "Mary moved to the bathroom.\n"
            "John went to the hallway.\n"
            "Mary travelled to the office."
        ),
        "question": "Where is Mary?",
        "answer": "office",
        "supporting_fact_ids": (4,),
        "source_length": 0,
        "source_example_id": "qa1-test:episode-1:question-5",
        "context_fact_ids": (1, 2, 4),
    }
    values.update(overrides)
    if "source_length" not in overrides:
        values["source_length"] = len(values["context"])
    return ReasoningExample(**values)


def make_qa2_example(**overrides: object) -> ReasoningExample:
    values: dict[str, object] = {
        "dataset": "babi",
        "task_id": "qa2",
        "split": "test",
        "context": (
            "Mary moved to the bathroom.\n"
            "Mary got the football there.\n"
            "Mary went to the garden."
        ),
        "question": "Where is the football?",
        "answer": "garden",
        "supporting_fact_ids": (2, 3),
        "source_length": 0,
        "source_example_id": "qa2-test:episode-1:question-4",
        "context_fact_ids": (1, 2, 3),
    }
    values.update(overrides)
    if "source_length" not in overrides:
        values["source_length"] = len(values["context"])
    return ReasoningExample(**values)


@pytest.mark.parametrize(
    ("sentence", "person", "destination"),
    [
        ("Mary moved to the bathroom.", "Mary", "bathroom"),
        ("John went to the hallway.", "John", "hallway"),
        ("Daniel went back to the kitchen.", "Daniel", "kitchen"),
        ("Sandra journeyed to the garden.", "Sandra", "garden"),
        ("Mary travelled to the office.", "Mary", "office"),
    ],
)
def test_parse_qa1_movement_supports_official_forms(
    sentence: str,
    person: str,
    destination: str,
) -> None:
    parsed_person, location = parse_qa1_movement(sentence, 7)

    assert parsed_person == person
    assert location == SupportedValue(destination, (7,))


@pytest.mark.parametrize("sentence", [None, 1, True])
def test_parse_qa1_movement_rejects_nonstring_sentence(sentence: object) -> None:
    with pytest.raises(TypeError, match="sentence must be a string"):
        parse_qa1_movement(sentence, 1)


@pytest.mark.parametrize(
    ("sentence", "message"),
    [
        ("", "sentence must be nonempty"),
        (" Mary moved to the bathroom.", "surrounding whitespace"),
        ("Mary moved to the bathroom. ", "surrounding whitespace"),
        ("Mary moved to the bathroom", "end with a period"),
        ("Mary teleported into the bathroom.", "one supported movement"),
        (" moved to the bathroom.", "surrounding whitespace"),
        ("Mary moved to the .", "contain a destination"),
        (
            "Mary moved to the office then went to the garden.",
            "one supported movement",
        ),
    ],
)
def test_parse_qa1_movement_rejects_invalid_sentence(
    sentence: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_qa1_movement(sentence, 1)


@pytest.mark.parametrize("fact_id", [True, 1.0, "1", None])
def test_parse_qa1_movement_rejects_noninteger_fact_id(fact_id: object) -> None:
    with pytest.raises(TypeError, match="fact_id must be an integer"):
        parse_qa1_movement("Mary moved to the bathroom.", fact_id)


@pytest.mark.parametrize("fact_id", [0, -1])
def test_parse_qa1_movement_rejects_nonpositive_fact_id(fact_id: int) -> None:
    with pytest.raises(ValueError, match="fact_id must be positive"):
        parse_qa1_movement("Mary moved to the bathroom.", fact_id)


def test_parse_qa1_question_extracts_person() -> None:
    assert parse_qa1_question("Where is Mary?") == "Mary"


@pytest.mark.parametrize("question", [None, 1, True])
def test_parse_qa1_question_rejects_nonstring_question(question: object) -> None:
    with pytest.raises(TypeError, match="question must be a string"):
        parse_qa1_question(question)


@pytest.mark.parametrize(
    "question",
    [
        "",
        " Where is Mary?",
        "Where is Mary? ",
        "Where was Mary?",
        "Where is Mary",
        "Where is ?",
        "Where is Mary??",
    ],
)
def test_parse_qa1_question_rejects_invalid_structure(question: str) -> None:
    with pytest.raises(ValueError):
        parse_qa1_question(question)


def test_interpret_qa1_uses_latest_location_and_provenance() -> None:
    result = interpret_qa1(make_qa1_example())

    assert result == OracleResult(answer="office", supporting_fact_ids=(4,))


def test_interpret_qa1_rejects_missing_context_provenance() -> None:
    with pytest.raises(ValueError, match="requires context_fact_ids"):
        interpret_qa1(make_qa1_example(context_fact_ids=None))


def test_interpret_qa1_rejects_unknown_person() -> None:
    with pytest.raises(ValueError, match="unknown for person 'Sandra'"):
        interpret_qa1(make_qa1_example(question="Where is Sandra?"))


def test_interpret_qa1_rejects_other_task() -> None:
    with pytest.raises(ValueError, match="does not support task 'qa2'"):
        interpret_qa1(make_qa1_example(task_id="qa2"))


def test_validate_oracle_result_accepts_matching_labels() -> None:
    validate_oracle_result(
        make_qa1_example(),
        OracleResult(answer="office", supporting_fact_ids=(4,)),
    )


def test_validate_oracle_result_rejects_answer_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match published answer"):
        validate_oracle_result(
            make_qa1_example(),
            OracleResult(answer="bathroom", supporting_fact_ids=(4,)),
        )


def test_validate_oracle_result_rejects_support_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match published support"):
        validate_oracle_result(
            make_qa1_example(),
            OracleResult(answer="office", supporting_fact_ids=(1,)),
        )


def test_validate_oracle_result_skips_unavailable_support_labels() -> None:
    validate_oracle_result(
        make_qa1_example(supporting_fact_ids=None),
        OracleResult(answer="office", supporting_fact_ids=(4,)),
    )


def test_interpret_dispatches_and_validates_qa1() -> None:
    assert interpret(make_qa1_example()) == OracleResult("office", (4,))


def test_interpret_rejects_unsupported_task() -> None:
    with pytest.raises(ValueError, match="unsupported symbolic task: 'qa6'"):
        interpret(make_qa1_example(task_id="qa6"))


@pytest.mark.parametrize(
    ("sentence", "action"),
    [
        ("Mary picked up the football there.", "acquire"),
        ("Mary grabbed the football there.", "acquire"),
        ("Mary got the football there.", "acquire"),
        ("Mary took the football there.", "acquire"),
        ("Mary left the football.", "drop"),
        ("Mary discarded the football.", "drop"),
        ("Mary put down the football.", "drop"),
        ("Mary dropped the football.", "drop"),
        ("Mary left the football there.", "drop"),
    ],
)
def test_parse_qa2_object_action_supports_official_forms(
    sentence: str,
    action: str,
) -> None:
    assert parse_qa2_object_action(sentence, 8) == ObjectAction(
        action=action,
        person="Mary",
        object_name="football",
        fact_id=8,
    )


def test_parse_qa2_object_action_requires_there_for_acquisition() -> None:
    with pytest.raises(ValueError, match="must end with 'there.'"):
        parse_qa2_object_action("Mary got the football.", 1)


def test_parse_qa2_question_extracts_object() -> None:
    assert parse_qa2_question("Where is the football?") == "football"


@pytest.mark.parametrize(
    "question",
    ["Where is football?", "Where was the football?", "Where is the ?"],
)
def test_parse_qa2_question_rejects_invalid_structure(question: str) -> None:
    with pytest.raises(ValueError):
        parse_qa2_question(question)


def test_interpret_qa2_moves_held_object_with_person() -> None:
    assert interpret_qa2(make_qa2_example()) == OracleResult("garden", (2, 3))


def test_interpret_qa2_leaves_dropped_object_in_place() -> None:
    context = (
        "Mary moved to the garden.\n"
        "Mary got the football there.\n"
        "Mary dropped the football.\n"
        "Mary moved to the office."
    )
    example = make_qa2_example(
        context=context,
        context_fact_ids=(1, 2, 3, 4),
        supporting_fact_ids=(3, 1),
    )

    assert interpret_qa2(example) == OracleResult("garden", (3, 1))


def test_interpret_qa2_rejects_drop_by_nonholder() -> None:
    context = (
        "Mary moved to the garden.\n"
        "John moved to the garden.\n"
        "Mary got the football there.\n"
        "John dropped the football."
    )
    example = make_qa2_example(
        context=context,
        context_fact_ids=(1, 2, 3, 4),
    )

    with pytest.raises(ValueError, match="cannot drop object"):
        interpret_qa2(example)


def test_interpret_qa2_allows_irrelevant_action_with_unknown_person_location() -> None:
    context = (
        "Daniel moved to the bedroom.\n"
        "Daniel picked up the apple there.\n"
        "Mary grabbed the milk there.\n"
        "Mary left the milk.\n"
        "Daniel put down the apple there."
    )
    example = make_qa2_example(
        context=context,
        context_fact_ids=(1, 2, 3, 4, 5),
        question="Where is the apple?",
        answer="bedroom",
        supporting_fact_ids=(5, 1),
    )

    assert interpret_qa2(example) == OracleResult("bedroom", (5, 1))


def test_interpret_dispatches_and_validates_qa2() -> None:
    assert interpret(make_qa2_example()) == OracleResult("garden", (2, 3))
