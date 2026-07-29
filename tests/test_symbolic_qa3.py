import pytest

from tinymem.data.schema import ReasoningExample
from tinymem.data.symbolic_qa3 import interpret_qa3, parse_qa3_question
from tinymem.data.symbolic_world import OracleResult


def make_example() -> ReasoningExample:
    context = (
        "Mary moved to the bathroom.\n"
        "Mary got the football there.\n"
        "Mary moved to the office.\n"
        "Mary moved to the bathroom.\n"
        "Mary dropped the football."
    )
    return ReasoningExample(
        dataset="babi",
        task_id="qa3",
        split="test",
        context=context,
        question="Where was the football before the bathroom?",
        answer="office",
        supporting_fact_ids=(5, 4, 3),
        source_length=len(context),
        source_example_id="qa3-test-1",
        context_fact_ids=(1, 2, 3, 4, 5),
    )


def test_parse_qa3_question() -> None:
    assert parse_qa3_question("Where was the football before the bathroom?") == (
        "football",
        "bathroom",
    )


def test_interpret_qa3_returns_previous_location_and_chain() -> None:
    assert interpret_qa3(make_example()) == OracleResult("office", (5, 4, 3))


def test_parse_qa3_question_rejects_wrong_structure() -> None:
    with pytest.raises(ValueError):
        parse_qa3_question("Where is the football?")
