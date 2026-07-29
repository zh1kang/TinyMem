from dataclasses import FrozenInstanceError

import pytest

from tinymem.data.schema import EvidenceFact, ReasoningExample


def make_example(**overrides: object) -> ReasoningExample:
    values: dict[str, object] = {
        "dataset": "babi",
        "task_id": "qa2",
        "split": "train",
        "context": "6 Mary went to the garden.\n12 Mary dropped the football.",
        "question": "Where is the football?",
        "answer": "garden",
        "supporting_fact_ids": (12, 6),
        "source_example_id": "qa2-train-000001-q14",
        "context_fact_ids": (6, 12),
    }
    values.update(overrides)
    if "source_length" not in values:
        context = values["context"]
        values["source_length"] = len(context) if isinstance(context, str) else 0
    return ReasoningExample(**values)


def test_reasoning_example_preserves_all_fields() -> None:
    example = make_example()

    assert example.dataset == "babi"
    assert example.task_id == "qa2"
    assert example.split == "train"
    assert example.question == "Where is the football?"
    assert example.answer == "garden"
    assert example.supporting_fact_ids == (12, 6)
    assert example.source_length == len(example.context)
    assert example.source_example_id == "qa2-train-000001-q14"
    assert example.context_fact_ids == (6, 12)


def test_reasoning_example_is_frozen() -> None:
    example = make_example()

    with pytest.raises(FrozenInstanceError):
        example.answer = "hallway"


@pytest.mark.parametrize(
    "field_name",
    [
        "dataset",
        "task_id",
        "split",
        "context",
        "question",
        "answer",
        "source_example_id",
    ],
)
def test_reasoning_example_rejects_nonstring_fields(field_name: str) -> None:
    with pytest.raises(TypeError, match=f"{field_name} must be a string"):
        make_example(**{field_name: 1})


@pytest.mark.parametrize(
    "field_name",
    [
        "dataset",
        "task_id",
        "context",
        "question",
        "answer",
        "source_example_id",
    ],
)
def test_reasoning_example_rejects_empty_required_strings(field_name: str) -> None:
    with pytest.raises(ValueError, match=f"{field_name} must be nonempty"):
        make_example(**{field_name: "  "})


@pytest.mark.parametrize("split", ["training", "dev", "holdout"])
def test_reasoning_example_rejects_unsupported_split(split: str) -> None:
    with pytest.raises(ValueError, match="unsupported split"):
        make_example(split=split)


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_reasoning_example_accepts_standard_splits(split: str) -> None:
    assert make_example(split=split).split == split


@pytest.mark.parametrize("invalid_length", [True, 1.0, "10"])
def test_reasoning_example_rejects_noninteger_source_length(
    invalid_length: object,
) -> None:
    with pytest.raises(TypeError, match="source_length must be an integer"):
        make_example(source_length=invalid_length)


def test_reasoning_example_rejects_incorrect_source_length() -> None:
    with pytest.raises(ValueError, match="must equal the context character length"):
        make_example(source_length=1)


def test_reasoning_example_distinguishes_unknown_from_empty_support() -> None:
    unknown = make_example(supporting_fact_ids=None)
    explicitly_empty = make_example(supporting_fact_ids=())

    assert unknown.supporting_fact_ids is None
    assert explicitly_empty.supporting_fact_ids == ()


def test_reasoning_example_rejects_mutable_supporting_fact_ids() -> None:
    with pytest.raises(TypeError, match="must be a tuple or None"):
        make_example(supporting_fact_ids=[12, 6])


@pytest.mark.parametrize("fact_ids", [(12, True), (12, 6.0), (12, "6")])
def test_reasoning_example_rejects_noninteger_supporting_fact_ids(
    fact_ids: tuple[object, ...],
) -> None:
    with pytest.raises(TypeError, match="must contain integers"):
        make_example(supporting_fact_ids=fact_ids)


@pytest.mark.parametrize("fact_ids", [(0,), (-1,), (12, 0)])
def test_reasoning_example_rejects_nonpositive_supporting_fact_ids(
    fact_ids: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        make_example(supporting_fact_ids=fact_ids)


def test_reasoning_example_rejects_duplicate_supporting_fact_ids() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        make_example(supporting_fact_ids=(6, 6))


def test_reasoning_example_preserves_supporting_fact_order() -> None:
    example = make_example(supporting_fact_ids=(12, 6))

    assert example.supporting_fact_ids == (12, 6)


def test_reasoning_example_distinguishes_unknown_context_fact_ids() -> None:
    assert make_example(context_fact_ids=None).context_fact_ids is None


def test_reasoning_example_rejects_misaligned_context_fact_ids() -> None:
    with pytest.raises(ValueError, match="must align with context lines"):
        make_example(context_fact_ids=(6,))


def test_reasoning_example_rejects_support_not_in_context() -> None:
    with pytest.raises(ValueError, match="must refer to context facts"):
        make_example(supporting_fact_ids=(13, 6))


def test_reasoning_example_validates_exact_evidence_spans() -> None:
    context = "Book. Mary went to the garden. More."
    start = context.index("Mary")
    evidence = EvidenceFact(1, "Mary went to the garden.", start, start + 24)

    example = make_example(
        task_id="qa1",
        context=context,
        context_fact_ids=None,
        supporting_fact_ids=(1,),
        evidence_facts=(evidence,),
    )

    assert example.context[evidence.start_char:evidence.end_char] == evidence.text


def test_reasoning_example_rejects_inexact_evidence_span() -> None:
    context = "Mary went to the garden."
    with pytest.raises(ValueError, match="must match its context span"):
        make_example(
            context=context,
            context_fact_ids=None,
            supporting_fact_ids=None,
            evidence_facts=(EvidenceFact(1, "Mary went to the office.", 0, len(context)),),
        )
