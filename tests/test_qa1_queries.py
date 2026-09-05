from dataclasses import replace

import pytest

from tinymem.data.qa1_queries import qa1_world_queries
from tinymem.data.reader_gate import ReaderCase


@pytest.fixture
def case():
    return ReaderCase(
        "source", "babi_qa1", "connected-history",
        "Mary went to the kitchen.\nJohn travelled to the hallway.\nMary moved to the office.",
        "Where is Mary?", "office",
    )


def test_world_queries_keep_history_and_use_latest_fact_or_unknown(case):
    queries = qa1_world_queries(case, ("John", "Daniel", "Mary"))
    assert [row.question for row in queries] == ["Where is John?", "Where is Daniel?", "Where is Mary?"]
    assert [row.answer for row in queries] == ["hallway", "unknown", "office"]
    assert [row.category for row in queries] == ["babi_qa1", "missing_entity", "babi_qa1"]
    assert len({row.case_id for row in queries}) == 3
    assert all(row.history_id == case.history_id and row.context == case.context for row in queries)
    assert queries == qa1_world_queries(case, ("John", "Daniel", "Mary"))


@pytest.mark.parametrize("people", [(), "Mary", ("Mary", "Mary"), ("",), (" Mary",), ("Mary?",), ("Mary\nJohn",), (1,)])
def test_world_queries_reject_invalid_people(case, people):
    with pytest.raises(ValueError):
        qa1_world_queries(case, people)


def test_world_queries_reject_wrong_semantics_and_source_answer(case):
    for changed in (replace(case, context=""), replace(case, category="babi_qa2")):
        with pytest.raises(ValueError, match="babi_qa1"):
            qa1_world_queries(changed, ("Mary",))
    with pytest.raises(ValueError, match="symbolic"):
        qa1_world_queries(replace(case, answer="kitchen"), ("Mary",))
    with pytest.raises(ValueError, match="movement"):
        qa1_world_queries(replace(case, context="Mary picked up the apple."), ("Mary",))
