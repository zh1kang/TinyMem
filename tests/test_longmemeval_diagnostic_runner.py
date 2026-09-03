import pytest

from scripts.diagnose_longmemeval import select_examples
from tinymem.data.longmemeval import LongMemEvalExample


def make_example(question_id: str, question_type: str) -> LongMemEvalExample:
    return LongMemEvalExample(
        question_id=question_id,
        question_type=question_type,
        question="Question?",
        answer=question_id,
        question_date="2024-01-01",
        sessions=(),
        answer_session_ids=(),
    )


def test_select_examples_takes_source_order_per_type() -> None:
    examples = [
        make_example("a1", "a"),
        make_example("a2", "a"),
        make_example("b1", "b"),
        make_example("b2", "b"),
    ]

    selected = select_examples(
        examples,
        max_examples=None,
        examples_per_type=1,
        full=False,
    )

    assert [example.question_id for example in selected] == ["a1", "b1"]


def test_select_examples_supports_explicit_full_and_prefix_modes() -> None:
    examples = [
        make_example("a1", "a"),
        make_example("a2", "a"),
        make_example("b1", "b"),
    ]

    assert select_examples(
        examples,
        max_examples=2,
        examples_per_type=None,
        full=False,
    ) == examples[:2]
    assert select_examples(
        examples,
        max_examples=None,
        examples_per_type=None,
        full=True,
    ) == examples


def test_select_examples_rejects_too_small_diagnostics() -> None:
    examples = [make_example("a1", "a"), make_example("b1", "b")]

    with pytest.raises(ValueError, match="at least 2"):
        select_examples(
            examples,
            max_examples=1,
            examples_per_type=None,
            full=False,
        )
