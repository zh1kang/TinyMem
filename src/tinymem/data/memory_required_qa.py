"""Segmented byte QA examples whose answers require external memory."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

from tinymem.data.schema import ReasoningExample
from tinymem.tokenization.byte_tokenizer import ByteTokenizer


DISTRACTOR_TEXTS = (
    "A quiet melody played while rain touched the glass.\n",
    "The old clock ticked steadily through the afternoon.\n",
    "Someone folded a blue blanket beside a wooden chair.\n",
    "Warm sunlight faded as clouds crossed the sky.\n",
)


@dataclass(frozen=True)
class MemoryRequiredQAExample:
    """Hold support, intervening distractors, and an answer-supervised query."""

    support_ids: tuple[int, ...]
    query_ids: tuple[int, ...]
    answer_ids: tuple[int, ...]
    segment_length: int
    source_example_id: str
    split: str
    distractor_ids: tuple[tuple[int, ...], ...] = ()

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
        if not isinstance(self.distractor_ids, tuple):
            raise TypeError("distractor_ids must be a tuple")
        for distractor in self.distractor_ids:
            if not isinstance(distractor, tuple) or not distractor:
                raise ValueError("each distractor must be a nonempty tuple")
            if len(distractor) > self.segment_length:
                raise ValueError("each distractor must fit inside one segment")
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < ByteTokenizer.vocab_size
                for value in distractor
            ):
                raise ValueError("distractors must contain valid byte-token IDs")
        if len(self.query_ids) + len(self.answer_ids) + 1 > self.segment_length:
            raise ValueError("query and answer must fit inside one segment")
        for name in ("source_example_id", "split"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")

    @property
    def query_position_offset(self) -> int:
        """Return the absolute start position after all preceding segments."""
        return (1 + len(self.distractor_ids)) * self.segment_length


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
    distractor_segments: int = 0,
) -> list[MemoryRequiredQAExample]:
    """Place qa1 evidence, neutral distractors, and the query in separate segments."""
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
    if isinstance(distractor_segments, bool) or not isinstance(
        distractor_segments,
        Integral,
    ):
        raise TypeError("distractor_segments must be an integer")
    if distractor_segments < 0:
        raise ValueError("distractor_segments must be nonnegative")
    encoded = []
    for example_index, example in enumerate(examples):
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
        distractor_ids = tuple(
            tuple(
                tokenizer.encode(
                    DISTRACTOR_TEXTS[
                        (example_index + distractor_index) % len(DISTRACTOR_TEXTS)
                    ]
                )
            )
            for distractor_index in range(int(distractor_segments))
        )
        if any(len(distractor) > segment_length for distractor in distractor_ids):
            raise ValueError("distractor text does not fit inside one segment")
        encoded.append(
            MemoryRequiredQAExample(
                support_ids=support_ids,
                query_ids=query_ids,
                answer_ids=answer_ids,
                segment_length=int(segment_length),
                source_example_id=example.source_example_id,
                split=example.split,
                distractor_ids=distractor_ids,
            )
        )
    return encoded
