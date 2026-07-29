import json
from pathlib import Path

import pytest

from tinymem.data.longmemeval import load_longmemeval_file


def make_record() -> dict[str, object]:
    return {
        "question_id": "q1",
        "question_type": "temporal-reasoning",
        "question": "What changed?",
        "answer": "the location",
        "question_date": "2024/01/02",
        "haystack_dates": ["2024/01/01"],
        "haystack_session_ids": ["s1"],
        "haystack_sessions": [[
            {"role": "user", "content": "I moved.", "has_answer": True},
            {"role": "assistant", "content": "Noted.", "has_answer": False},
        ]],
        "answer_session_ids": ["s1"],
    }


def test_load_longmemeval_preserves_sessions_as_test_only(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text(json.dumps([make_record()]), encoding="utf-8")
    example = load_longmemeval_file(path)[0]
    assert example.split == "test"
    assert example.training_allowed is False
    assert example.sessions[0].messages[0].content == "I moved."
    assert example.answer_session_ids == ("s1",)


def test_load_longmemeval_rejects_unaligned_sessions(tmp_path: Path) -> None:
    record = make_record()
    record["haystack_dates"] = []
    path = tmp_path / "data.json"
    path.write_text(json.dumps([record]), encoding="utf-8")
    with pytest.raises(ValueError, match="session arrays must align"):
        load_longmemeval_file(path)


def test_load_longmemeval_preserves_integer_count_answers(tmp_path: Path) -> None:
    record = make_record()
    record["answer"] = 3
    path = tmp_path / "data.json"
    path.write_text(json.dumps([record]), encoding="utf-8")
    assert load_longmemeval_file(path)[0].answer == 3
