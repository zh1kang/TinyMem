from dataclasses import FrozenInstanceError, replace

import pytest

from tinymem.data.opaque_qa1 import make_opaque_qa1_world
from tinymem.data.reader_gate import ReaderCase
from tinymem.evaluation.association_diagnostics import (
    AssociationAnswerPattern, AssociationHistoryLabel, label_association_history, summarize_association_answers,
)


CHUNKS = (
    "Mary moved to the kitchen.\nJohn went to the garden.",
    "Mary travelled to the office.",
    "John journeyed to the garden.",
    "Mary went back to the kitchen.",
)


def cases(chunks=CHUNKS):
    return tuple(ReaderCase(name, "opaque_qa1_known", "history", "\n".join(chunks), f"Where is {name}?", answer)
                 for name, answer in (("Mary", "kitchen"), ("John", "garden"), ("Alice", "unknown")))


def test_timing_changes_repeated_destinations_and_absent_entities():
    labels = label_association_history(CHUNKS, cases())
    assert labels == {
        "Mary": AssociationHistoryLabel(1, 4, 3, 2),
        "John": AssociationHistoryLabel(1, 3, 2, 0),
        "Alice": AssociationHistoryLabel(None, None, 0, 0),
    }
    with pytest.raises(FrozenInstanceError):
        labels["Mary"].movement_count = 99


def test_many_changes_inside_one_chunk_are_counted():
    chunks = ("\n".join(CHUNKS),)
    labels = label_association_history(chunks, cases(chunks))
    assert labels["Mary"] == AssociationHistoryLabel(1, 1, 3, 2)
    assert labels["John"] == AssociationHistoryLabel(1, 1, 2, 0)


def test_query_order_does_not_change_labels():
    assert label_association_history(CHUNKS, cases()) == label_association_history(CHUNKS, cases()[::-1])


def test_real_world_controls_preserve_timing_and_change_counts():
    left = "Mary moved to the kitchen.\nJohn went to the garden.\nDaniel moved to the office.\nSandra went to the bathroom.\nMary went to the bedroom."
    right = "Mary went to the hallway.\nJohn moved to the bedroom.\nDaniel went to the office.\nSandra went to the kitchen."
    sources = (ReaderCase("a", "babi_qa1", "a", left, "Where is Mary?", "bedroom"),
               ReaderCase("b", "babi_qa1", "b", right, "Where is Sandra?", "kitchen"))
    labels = []
    for options in ({}, {"opaque": False}, {"counterfactual": True}):
        world = make_opaque_qa1_world(sources, ("a", "b"), world_id="fixture", seed=19, **options)
        mapped = label_association_history(world.chunks, world.queries)
        labels.append([mapped[case.case_id] for case in world.queries])
    assert labels[0] == labels[1] == labels[2]
    assert [label.last_mention_chunk for label in labels[0]] == [3, 1, 3, 3, 2, 2, 4, 4, None]
    assert labels[0][0].location_changes == 1


@pytest.mark.parametrize("chunks", [(), ("",), (" ",), (None,), ("Mary moved to the kitchen.\n\nJohn went to the garden.",)])
def test_invalid_chunks_fail(chunks):
    examples = cases(chunks) if all(isinstance(chunk, str) for chunk in chunks) else cases()
    with pytest.raises(ValueError):
        label_association_history(chunks, examples)


@pytest.mark.parametrize("examples", [(), (cases()[0], cases()[0]), (replace(cases()[0], case_id=""),),
    (replace(cases()[0], context="other"),), (replace(cases()[0], answer="office"),),
    (replace(cases()[2], answer="kitchen"),), (replace(cases()[0], question="Where was Mary?"),)])
def test_invalid_queries_fail(examples):
    with pytest.raises(ValueError):
        label_association_history(CHUNKS, examples)


def answer_cases():
    context = "\n".join(CHUNKS) + "\nDaniel went to the kitchen."
    return tuple(ReaderCase(name, "opaque_qa1_known", "history", context, f"Where is {name}?", answer)
                 for name, answer in (("Mary", "kitchen"), ("John", "garden"), ("Daniel", "kitchen")))


def test_one_answer_can_be_often_correct_without_distinguishing_entities():
    result = summarize_association_answers(answer_cases(), {"Mary": " Kitchen!", "John": "kitchen", "Daniel": "KITCHEN"})
    assert result == AssociationAnswerPattern(3, 2, 1, 2, 2)
    with pytest.raises(FrozenInstanceError):
        result.correct = 3


def test_binding_answers_can_exceed_the_constant_answer_ceiling():
    examples = answer_cases()
    predictions = {case.case_id: case.answer for case in examples}
    assert summarize_association_answers(examples, predictions) == AssociationAnswerPattern(3, 3, 2, 2, 2)
    assert summarize_association_answers(examples[::-1], predictions) == summarize_association_answers(examples, predictions)


def test_low_accuracy_does_not_imply_identical_predictions():
    result = summarize_association_answers(answer_cases(), {"Mary": "office", "John": "hallway", "Daniel": "bathroom"})
    assert result == AssociationAnswerPattern(3, 0, 3, 2, 2)


def test_unknown_and_empty_predictions_are_kept_as_observed_answers():
    assert summarize_association_answers(answer_cases(), dict.fromkeys(("Mary", "John", "Daniel"), "unknown")).distinct_predictions == 1
    assert summarize_association_answers(answer_cases(), dict.fromkeys(("Mary", "John", "Daniel"), "")).correct == 0


def test_identical_references_make_identical_correct_predictions_valid():
    examples = tuple(replace(case, answer="kitchen", context=case.context.replace("garden", "kitchen")) for case in answer_cases())
    assert summarize_association_answers(examples, dict.fromkeys(("Mary", "John", "Daniel"), "kitchen")) == AssociationAnswerPattern(3, 3, 1, 1, 3)


@pytest.mark.parametrize("change", ["empty", "duplicate_id", "empty_id", "missing_prediction", "extra_prediction",
                                    "mixed_context", "mixed_history_id", "duplicate_question", "absent", "unknown", "empty_reference"])
def test_invalid_answer_pattern_inputs_fail(change):
    examples = list(answer_cases())
    predictions = {case.case_id: case.answer for case in examples}
    if change == "empty":
        examples = []
    elif change == "duplicate_id":
        examples[1] = replace(examples[1], case_id=examples[0].case_id)
    elif change == "empty_id":
        examples[1] = replace(examples[1], case_id="")
    elif change == "missing_prediction":
        predictions.pop("John")
    elif change == "extra_prediction":
        predictions["absent"] = "unknown"
    else:
        changes = {"mixed_context": {"context": "different"}, "mixed_history_id": {"history_id": "different"},
                   "duplicate_question": {"question": examples[0].question}, "absent": {"category": "opaque_qa1_missing"},
                   "unknown": {"answer": "Unknown!"}, "empty_reference": {"answer": "..."}}
        examples[1] = replace(examples[1], **changes[change])
    with pytest.raises(ValueError):
        summarize_association_answers(examples, predictions)
