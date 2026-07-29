import pytest

from tinymem.data.schema import ReasoningExample
from tinymem.data.symbolic_qa4 import interpret_qa4, parse_qa4_question
from tinymem.data.symbolic_world import OracleResult


def make_example(question: str, answer: str) -> ReasoningExample:
    context = "The office is north of the kitchen.\nThe garden is south of the kitchen."
    return ReasoningExample(
        dataset="babi",
        task_id="qa4",
        split="test",
        context=context,
        question=question,
        answer=answer,
        supporting_fact_ids=(1,),
        source_length=len(context),
        source_example_id="qa4-test-1",
        context_fact_ids=(1, 2),
    )


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is north of the kitchen?", ("north", "kitchen")),
        ("What is the office north of?", ("south", "office")),
    ],
)
def test_parse_qa4_question(question: str, expected: tuple[str, str]) -> None:
    assert parse_qa4_question(question) == expected


def test_interpret_qa4_answers_direct_relation() -> None:
    assert interpret_qa4(make_example("What is north of the kitchen?", "office")) == OracleResult(
        "office",
        (1,),
    )


def test_interpret_qa4_answers_inverse_relation() -> None:
    assert interpret_qa4(make_example("What is the office north of?", "kitchen")) == OracleResult(
        "kitchen",
        (1,),
    )
