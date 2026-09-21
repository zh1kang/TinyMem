"""One question with the visible history that determines its answer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReaderCase:
    case_id: str
    category: str
    history_id: str
    context: str
    question: str
    answer: str
