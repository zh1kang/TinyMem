import pytest

from tinymem.data.schema import ReasoningExample
from tinymem.evaluation.conversational_evidence import locate_evidence
from tinymem.training.conversational_qa import format_conversational_qa_prompt


def make_example(supporting: tuple[int, ...]) -> ReasoningExample:
    context = "Mary moved to the bathroom.\nJohn went to the hallway.\nMary went to the garden."
    return ReasoningExample(
        dataset="babi",
        task_id="qa1",
        split="test",
        context=context,
        question="Where is Mary?",
        answer="garden",
        supporting_fact_ids=supporting,
        source_length=len(context),
        source_example_id="qa1_test.txt:episode-000001:question-4",
        context_fact_ids=(1, 2, 3),
    )


def test_locate_evidence_finds_fact_line_offsets() -> None:
    example = make_example((3,))
    prompt = format_conversational_qa_prompt(example)

    placement = locate_evidence(example)

    encoded = prompt.encode("utf-8")
    assert placement.prompt_bytes == len(encoded)
    start = placement.evidence_starts[0]
    assert encoded[start:].startswith(b"User: Mary went to the garden.\n")
    assert placement.earliest_distance == len(encoded) - start


def test_in_final_segment_uses_fixed_blocks_from_byte_zero() -> None:
    placement = locate_evidence(make_example((1, 3)))
    header_and_first = placement.evidence_starts[0]

    assert placement.in_final_segment(4096)
    assert not placement.in_final_segment(header_and_first + 1)
    with pytest.raises(ValueError, match="positive"):
        placement.in_final_segment(0)


def test_locate_evidence_uses_original_fact_ids() -> None:
    context = "Mary moved to the bathroom.\nMary went to the garden."
    example = ReasoningExample(
        dataset="babi",
        task_id="qa1",
        split="test",
        context=context,
        question="Where is Mary?",
        answer="garden",
        supporting_fact_ids=(9,),
        source_length=len(context),
        source_example_id="qa1_test.txt:episode-000002:question-10",
        context_fact_ids=(7, 9),
    )

    placement = locate_evidence(example)

    assert placement.prompt_bytes - placement.earliest_distance == (
        placement.evidence_starts[0]
    )
    prompt = format_conversational_qa_prompt(example).encode("utf-8")
    assert prompt[placement.evidence_starts[0] :].startswith(
        b"User: Mary went to the garden.\n"
    )


def test_locate_evidence_requires_supporting_fact_ids() -> None:
    example = make_example(())

    with pytest.raises(ValueError, match="requires supporting_fact_ids"):
        locate_evidence(example)
