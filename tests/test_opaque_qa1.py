import re
from dataclasses import replace

import pytest

from tinymem.data.opaque_qa1 import PEOPLE, make_opaque_qa1_world
from tinymem.data.reader_gate import ReaderCase


def sources():
    left = "Mary moved to the kitchen.\nJohn went to the garden.\nDaniel journeyed to the bathroom.\nSandra travelled to the office.\nMary went back to the bedroom."
    right = "Mary went to the hallway.\nJohn moved to the bedroom.\nDaniel went back to the office.\nSandra journeyed to the kitchen."
    return (ReaderCase("a", "babi_qa1", "a", left, "Where is Mary?", "bedroom"),
            ReaderCase("b", "babi_qa1", "b", right, "Where is Sandra?", "kitchen"))


def world(**kwargs):
    return make_opaque_qa1_world(sources(), ("group-a", "group-b"), world_id="train:0", seed=17, **kwargs)


def test_reproducible_distinct_80_bit_names_and_shared_history():
    result = world()
    assert result == world()
    assert len(set(result.entities)) == 9
    assert all(re.fullmatch(r"person[0-9a-f]{20}", name) for name in result.entities)
    assert len(result.chunks) == 4
    assert len({case.context for case in result.queries}) == len({case.history_id for case in result.queries}) == 1
    assert result.queries[0].context == "\n".join(result.chunks)
    assert len(result.queries) == 9
    assert all(case.category == "opaque_qa1_known" for case in result.queries[:8])
    assert result.queries[-1].answer == "unknown"
    assert result.entities[-1] not in result.queries[0].context
    assert result.source_group_ids == ("group-a", "group-b")
    assert result.source_case_ids == ("a", "b")


def test_independent_replay_and_source_chronology():
    result = world()
    state = {}
    order = []
    for sentence in result.queries[0].context.splitlines():
        name, room = sentence.split()[0], sentence.split()[-1].removesuffix(".")
        state[name] = room
        order.append(result.entities.index(name))
    assert order == [0, 1, 4, 5, 2, 3, 0, 6, 7]
    assert [case.answer for case in result.queries[:8]] == [state[name] for name in result.entities[:8]]
    assert "went back to the" in result.queries[0].context


def test_short_control_preserves_rooms_and_counterfactual_changes_every_binding():
    original, short, changed = world(), world(opaque=False), world(counterfactual=True)
    assert original.entities == changed.entities
    assert short.entities[:4] == PEOPLE
    assert [case.answer for case in original.queries] == [case.answer for case in short.queries]
    assert all(a.answer != b.answer for a, b in zip(original.queries[:8], changed.queries[:8], strict=True))
    assert original.queries[-1].answer == changed.queries[-1].answer == "unknown"
    assert original.queries[0].history_id != changed.queries[0].history_id
    assert {case.case_id for case in original.queries}.isdisjoint(case.case_id for case in changed.queries)


def test_world_and_split_identity_change_names():
    changed = make_opaque_qa1_world(sources(), ("group-a", "group-b"), world_id="development:0", seed=17)
    assert set(world().entities).isdisjoint(changed.entities)


@pytest.mark.parametrize("field,value", [("answer", "office"), ("category", "babi_qa2"),
    ("context", "Mary moved to the kitchen."),
    ("question", "Where is Alice?"),
    ("context", sources()[0].context.replace("bedroom.", "bedroom .")),
    ("context", sources()[0].context.replace("Mary moved", "Mary  moved")),
    ("context", sources()[0].context.replace("Mary", "Alice")),
    ("context", sources()[0].context.replace("kitchen", "attic"))])
def test_invalid_sources(field, value):
    left, right = sources()
    with pytest.raises(ValueError):
        make_opaque_qa1_world((replace(left, **{field: value}), right), ("a", "b"), world_id="x", seed=1)


@pytest.mark.parametrize("groups", [("a", "a"), ("", "b"), ("a",)])
def test_invalid_group_identity(groups):
    with pytest.raises(ValueError):
        make_opaque_qa1_world(sources(), groups, world_id="x", seed=1)


@pytest.mark.parametrize("seed", [-1, True, 1.2])
def test_invalid_seed(seed):
    with pytest.raises(ValueError):
        make_opaque_qa1_world(sources(), ("a", "b"), world_id="x", seed=seed)


def test_duplicate_source_context_rejected_even_with_distinct_group_labels():
    with pytest.raises(ValueError):
        make_opaque_qa1_world((sources()[0], sources()[0]), ("a", "b"), world_id="x", seed=1)
