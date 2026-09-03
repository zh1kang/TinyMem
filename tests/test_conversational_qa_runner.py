import pytest

from tinymem.data.sampling import select_reasoning_examples
from tinymem.data.schema import ReasoningExample


def make_example(index: int) -> ReasoningExample:
    context = f"Fact {index}."
    return ReasoningExample(
        dataset="babi",
        task_id="qa1",
        split="train",
        context=context,
        question="Question?",
        answer=str(index),
        supporting_fact_ids=(1,),
        source_length=len(context),
        source_example_id=f"example-{index}",
        context_fact_ids=(1,),
    )


def test_select_examples_is_seeded_without_replacement() -> None:
    examples = [make_example(index) for index in range(10)]

    first = select_reasoning_examples(examples, count=4, seed=7)
    second = select_reasoning_examples(examples, count=4, seed=7)

    assert first == second
    assert len({example.source_example_id for example in first}) == 4


def test_select_examples_rejects_unavailable_count() -> None:
    with pytest.raises(ValueError, match="available"):
        select_reasoning_examples([make_example(1)], count=2, seed=7)


def test_select_all_examples_preserves_source_order() -> None:
    examples = [make_example(index) for index in range(4)]

    assert select_reasoning_examples(examples, count=None, seed=7) == examples
