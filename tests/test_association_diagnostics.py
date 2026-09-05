from dataclasses import FrozenInstanceError, replace

import pytest

from tinymem.data.opaque_qa1 import make_opaque_qa1_world
from tinymem.data.reader_gate import ReaderCase
from tinymem.evaluation.association_diagnostics import AssociationHistoryLabel, label_association_history


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
