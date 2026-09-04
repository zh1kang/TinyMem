"""Two-segment byte QA examples whose answers require external memory."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

from tinymem.data.schema import ReasoningExample
from tinymem.tokenization.byte_tokenizer import ByteTokenizer


@dataclass(frozen=True)
class MemoryRequiredQAExample:
    """Hold one support segment and one answer-supervised query segment."""

    support_ids: tuple[int, ...]
    query_ids: tuple[int, ...]
    answer_ids: tuple[int, ...]
    segment_length: int
    source_example_id: str
    split: str

    def __post_init__(self) -> None:
        for name in ("support_ids", "query_ids", "answer_ids"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not values:
                raise ValueError(f"{name} must be a nonempty tuple")
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < ByteTokenizer.vocab_size
                for value in values
            ):
                raise ValueError(f"{name} must contain valid byte-token IDs")
        if isinstance(self.segment_length, bool) or not isinstance(
            self.segment_length,
            Integral,
        ):
            raise TypeError("segment_length must be an integer")
        if self.segment_length <= 0:
            raise ValueError("segment_length must be positive")
        if len(self.support_ids) > self.segment_length:
            raise ValueError("support_ids must fit inside one segment")
        if len(self.query_ids) + len(self.answer_ids) + 1 > self.segment_length:
            raise ValueError("query and answer must fit inside one segment")
        for name in ("source_example_id", "split"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")


def supporting_fact_text(example: ReasoningExample) -> str:
    """Return the only supporting fact from a qa1 example."""
    if not isinstance(example, ReasoningExample):
        raise TypeError("example must be a ReasoningExample")
    if example.task_id != "qa1":
        raise ValueError("memory-required byte QA currently supports qa1 only")
    if example.supporting_fact_ids is None or len(example.supporting_fact_ids) != 1:
        raise ValueError("qa1 examples must contain exactly one supporting fact")
    if example.context_fact_ids is None:
        raise ValueError("qa1 examples must contain context fact IDs")
    support_id = example.supporting_fact_ids[0]
    try:
        support_index = example.context_fact_ids.index(support_id)
    except ValueError as error:
        raise ValueError("supporting fact is absent from the context") from error
    return example.context.splitlines()[support_index]


def build_memory_required_qa_examples(
    examples: Sequence[ReasoningExample],
    tokenizer: ByteTokenizer,
    *,
    segment_length: int,
) -> list[MemoryRequiredQAExample]:
    """Keep only qa1 evidence and place each query in the next segment."""
    if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)):
        raise TypeError("examples must be a sequence")
    if not examples or not all(
        isinstance(example, ReasoningExample) for example in examples
    ):
        raise ValueError("examples must contain ReasoningExample values")
    if not isinstance(tokenizer, ByteTokenizer):
        raise TypeError("tokenizer must be a ByteTokenizer")
    if isinstance(segment_length, bool) or not isinstance(segment_length, Integral):
        raise TypeError("segment_length must be an integer")
    if segment_length <= 0:
        raise ValueError("segment_length must be positive")
    encoded = []
    for example in examples:
        support_text = f"Fact: {supporting_fact_text(example)}\n"
        support_ids = tuple(tokenizer.encode(support_text))
        if len(support_ids) > segment_length:
            raise ValueError("supporting fact does not fit inside one segment")
        query_ids = tuple(
            tokenizer.encode(f"Question: {example.question}\nAnswer:")
        )
        answer_ids = tuple(tokenizer.encode(example.answer))
        if len(query_ids) + len(answer_ids) + 1 > segment_length:
            raise ValueError("question and answer do not fit inside one segment")
        encoded.append(
            MemoryRequiredQAExample(
                support_ids=support_ids,
                query_ids=query_ids,
                answer_ids=answer_ids,
                segment_length=int(segment_length),
                source_example_id=example.source_example_id,
                split=example.split,
            )
        )
    return encoded
