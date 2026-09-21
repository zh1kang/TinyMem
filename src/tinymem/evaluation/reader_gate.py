"""Fixed reader prompts and exact-match answer scoring."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Sequence

from tinymem.data.reader_gate import ReaderCase

_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)

READER_SYSTEM_PROMPT = (
    "Answer the question using only the supplied facts. "
    "Facts are in chronological order; later facts update earlier facts about the same entity. "
    "Reply with only the answer, without an explanation. "
    "If the facts do not determine the answer, reply with unknown."
)
READER_GATE_CATEGORIES = ("babi_qa1", "correction_changed", "correction_unchanged")


def reader_messages(case: ReaderCase, *, condition: str) -> list[dict[str, str]]:
    if condition not in ("full_context", "question_only"):
        raise ValueError("unknown reader gate condition")
    context = case.context if condition == "full_context" else "(no facts supplied)"
    return [
        {"role": "system", "content": READER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Facts:\n{context}\n\nQuestion: {case.question}"},
    ]


def normalized_answer(text: str) -> str:
    """Return a stable lowercase alphanumeric answer form."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return " ".join(_TOKEN_PATTERN.findall(text.casefold()))


def reader_exact_match(prediction: str, answer: str, category: str) -> bool:
    if category in ("exact_copy", "randomized_bindings"):
        return prediction.strip() == answer
    return normalized_answer(prediction) == normalized_answer(answer)


def summarize_reader_predictions(predictions: Sequence[dict[str, object]]) -> dict[str, object]:
    if not predictions:
        raise ValueError("predictions must be nonempty")
    groups = defaultdict(list)
    references = defaultdict(Counter)
    seen = set()
    for row in predictions:
        identity = (row["condition"], row["case_id"])
        if identity in seen:
            raise ValueError("duplicate reader prediction")
        seen.add(identity)
        key = (row["condition"], row["category"])
        correct = reader_exact_match(str(row["prediction"]), str(row["answer"]), str(row["category"]))
        groups[key].append(correct)
        references[key][str(row["answer"])] += 1
    results = {}
    for (condition, category), values in sorted(groups.items()):
        results.setdefault(condition, {})[category] = {
            "count": len(values), "correct": sum(values), "exact_accuracy": sum(values) / len(values),
            "majority_answer_prior": max(references[(condition, category)].values()) / len(values),
        }
    full = results.get("full_context", {})
    complete = all(category in full and full[category]["count"] >= 100 for category in READER_GATE_CATEGORIES)
    return {
        "by_condition": results, "gate_threshold": 0.95,
        "gate_categories": list(READER_GATE_CATEGORIES),
        "minimum_examples_per_gate_category": 100,
        "gate_passed": complete and all(full[category]["exact_accuracy"] >= 0.95 for category in READER_GATE_CATEGORIES),
        "claim": "visible_evidence_development_competence_not_compression",
    }
