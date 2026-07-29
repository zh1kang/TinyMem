import pytest

from tinymem.data.schema import ReasoningExample
from tinymem.data.symbolic_qa5 import interpret_qa5, parse_qa5_transfer
from tinymem.data.symbolic_world import OracleResult


def make_example(question: str, answer: str) -> ReasoningExample:
    context = "Bill gave the football to Fred."
    return ReasoningExample(
        dataset="babi",
        task_id="qa5",
        split="test",
        context=context,
        question=question,
        answer=answer,
        supporting_fact_ids=(1,),
        source_length=len(context),
        source_example_id="qa5-test-1",
        context_fact_ids=(1,),
    )


def test_parse_qa5_transfer() -> None:
    transfer = parse_qa5_transfer("Bill handed the football to Fred.", 3)

    assert (transfer.giver, transfer.object_name, transfer.recipient, transfer.fact_id) == (
        "Bill",
        "football",
        "Fred",
        3,
    )


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        ("What did Bill give to Fred?", "football"),
        ("Who gave the football?", "Bill"),
        ("Who gave the football to Fred?", "Bill"),
        ("Who received the football?", "Fred"),
        ("Who did Bill give the football to?", "Fred"),
    ],
)
def test_interpret_qa5_supports_official_questions(question: str, answer: str) -> None:
    assert interpret_qa5(make_example(question, answer)) == OracleResult(answer, (1,))


def test_interpret_qa5_rejects_ambiguous_answers() -> None:
    context = "Bill gave the apple to Fred.\nBill passed the football to Fred."
    example = ReasoningExample(
        dataset="babi",
        task_id="qa5",
        split="test",
        context=context,
        question="What did Bill give to Fred?",
        answer="apple",
        supporting_fact_ids=(1,),
        source_length=len(context),
        source_example_id="qa5-ambiguous",
        context_fact_ids=(1, 2),
    )

    with pytest.raises(ValueError, match="ambiguous qa5 answers"):
        interpret_qa5(example)
