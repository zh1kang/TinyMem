"""Strict held-out reader for the cleaned LongMemEval release."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LongMemMessage:
    role: str
    content: str
    has_answer: bool | None


@dataclass(frozen=True)
class LongMemSession:
    session_id: str
    date: str
    messages: tuple[LongMemMessage, ...]


@dataclass(frozen=True)
class LongMemEvalExample:
    question_id: str
    question_type: str
    question: str
    answer: str | int
    question_date: str
    sessions: tuple[LongMemSession, ...]
    answer_session_ids: tuple[str, ...]
    split: str = "test"
    training_allowed: bool = False

    @property
    def gold_unanswerable(self) -> bool:
        return self.question_id.endswith("_abs")


def _required_string(record: dict[str, object], key: str, index: int) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"LongMemEval record {index} field {key!r} must be a nonempty string")
    return value


def _required_answer(record: dict[str, object], index: int) -> str | int:
    value = record.get("answer")
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(
            f"LongMemEval record {index} field 'answer' must be a string or integer"
        )
    if isinstance(value, str) and not value.strip():
        raise ValueError(
            f"LongMemEval record {index} field 'answer' must be nonempty"
        )
    return value


def load_longmemeval_file(path: str | Path) -> list[LongMemEvalExample]:
    """Load LongMemEval as test-only ordered sessions."""
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"LongMemEval source file does not exist: {source_path}")
    try:
        records = json.loads(source_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid LongMemEval JSON in {source_path}") from error
    if not isinstance(records, list) or not records:
        raise ValueError("LongMemEval source must be a nonempty JSON array")

    examples: list[LongMemEvalExample] = []
    seen_ids: set[str] = set()
    for index, raw_record in enumerate(records, start=1):
        if not isinstance(raw_record, dict):
            raise TypeError(f"LongMemEval record {index} must be a JSON object")
        record = raw_record
        question_id = _required_string(record, "question_id", index)
        if question_id in seen_ids:
            raise ValueError(f"duplicate LongMemEval question_id {question_id!r}")
        seen_ids.add(question_id)
        session_ids = record.get("haystack_session_ids")
        dates = record.get("haystack_dates")
        raw_sessions = record.get("haystack_sessions")
        answer_ids = record.get("answer_session_ids")
        if not all(isinstance(value, list) for value in (session_ids, dates, raw_sessions, answer_ids)):
            raise ValueError(f"LongMemEval record {index} session fields must be arrays")
        if not (len(session_ids) == len(dates) == len(raw_sessions)):
            raise ValueError(f"LongMemEval record {index} session arrays must align")
        if not all(isinstance(value, str) for value in session_ids + dates + answer_ids):
            raise ValueError(f"LongMemEval record {index} session identifiers and dates must be strings")
        if not set(answer_ids).issubset(set(session_ids)):
            raise ValueError(f"LongMemEval record {index} answer sessions are not in the haystack")

        sessions: list[LongMemSession] = []
        for session_id, date, raw_messages in zip(session_ids, dates, raw_sessions, strict=True):
            if not isinstance(raw_messages, list) or not raw_messages:
                raise ValueError(f"LongMemEval record {index} sessions must contain messages")
            messages: list[LongMemMessage] = []
            for raw_message in raw_messages:
                if not isinstance(raw_message, dict):
                    raise ValueError(f"LongMemEval record {index} messages must be objects")
                role = raw_message.get("role")
                content = raw_message.get("content")
                has_answer = raw_message.get("has_answer")
                if role not in {"user", "assistant"} or not isinstance(content, str):
                    raise ValueError(f"LongMemEval record {index} has an invalid message")
                if has_answer is not None and not isinstance(has_answer, bool):
                    raise ValueError(f"LongMemEval record {index} has invalid has_answer metadata")
                messages.append(LongMemMessage(role, content, has_answer))
            sessions.append(LongMemSession(session_id, date, tuple(messages)))

        examples.append(LongMemEvalExample(
            question_id=question_id,
            question_type=_required_string(record, "question_type", index),
            question=_required_string(record, "question", index),
            answer=_required_answer(record, index),
            question_date=_required_string(record, "question_date", index),
            sessions=tuple(sessions),
            answer_session_ids=tuple(answer_ids),
        ))
    return examples
