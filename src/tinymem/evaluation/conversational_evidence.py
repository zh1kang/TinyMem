"""Locate supporting evidence inside byte-level conversational prompts."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

from tinymem.data.schema import ReasoningExample
from tinymem.training.conversational_qa import format_conversational_qa_prompt


@dataclass(frozen=True)
class EvidencePlacement:
    """Describe where the supporting facts sit relative to the answer position.

    ``prompt_bytes`` is the encoded prompt length, ``evidence_starts`` holds the
    byte offset of each supporting fact line, and ``earliest_distance`` is the
    number of bytes from the earliest supporting fact to the answer position.
    """

    prompt_bytes: int
    evidence_starts: tuple[int, ...]
    earliest_distance: int

    def in_final_segment(self, segment_length: int) -> bool:
        """Return whether every supporting fact shares the answer's segment.

        The segmented decoder cuts the prompt into fixed blocks from byte zero,
        so the answer position only attends to bytes at or after the start of
        the final block.
        """
        if isinstance(segment_length, bool) or not isinstance(
            segment_length,
            Integral,
        ):
            raise TypeError("segment_length must be an integer")
        if segment_length <= 0:
            raise ValueError("segment_length must be positive")
        final_start = ((self.prompt_bytes - 1) // int(segment_length)) * int(
            segment_length
        )
        return min(self.evidence_starts) >= final_start


def locate_evidence(example: ReasoningExample) -> EvidencePlacement:
    """Compute supporting-fact byte offsets in the conversational prompt."""
    if not isinstance(example, ReasoningExample):
        raise TypeError("example must be a ReasoningExample")
    if example.context_fact_ids is None:
        raise ValueError("evidence placement requires context_fact_ids")
    if not example.supporting_fact_ids:
        raise ValueError("evidence placement requires supporting_fact_ids")

    prompt = format_conversational_qa_prompt(example)
    header = f"[session {example.dataset}:{example.task_id} | undated]\n"
    offset = len(header.encode("utf-8"))
    line_starts = []
    for line in example.context.splitlines():
        line_starts.append(offset)
        offset += len(f"User: {line}\n".encode("utf-8"))
    evidence_starts = tuple(
        line_starts[example.context_fact_ids.index(fact_id)]
        for fact_id in example.supporting_fact_ids
    )
    prompt_bytes = len(prompt.encode("utf-8"))
    return EvidencePlacement(
        prompt_bytes=prompt_bytes,
        evidence_starts=evidence_starts,
        earliest_distance=prompt_bytes - min(evidence_starts),
    )
