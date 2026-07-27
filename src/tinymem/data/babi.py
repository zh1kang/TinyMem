"""Parser for the official numbered bAbI text format."""

from collections.abc import Iterable
from pathlib import Path

from tinymem.data.schema import ReasoningExample, SUPPORTED_SPLITS


def split_numbered_line(raw_line: str) -> tuple[int, str]:
    """Separate a bAbI source line into its numeric ID and remaining payload."""
    if not isinstance(raw_line, str):
        raise TypeError("raw_line must be a string")

    line = raw_line.rstrip("\r\n")
    line_id_text, separator, payload = line.partition(" ")
    if not separator:
        raise ValueError("bAbI line must contain an ID followed by a payload")

    try:
        line_id = int(line_id_text)
    except ValueError as error:
        raise ValueError("bAbI line ID must be an integer") from error
    if line_id <= 0:
        raise ValueError("bAbI line ID must be positive")
    if not payload.strip():
        raise ValueError("bAbI line payload must be nonempty")
    return line_id, payload


def parse_question_payload(payload: str) -> tuple[str, str, tuple[int, ...]]:
    """Parse question text, answer, and supporting IDs from one payload."""
    if not isinstance(payload, str):
        raise TypeError("payload must be a string")

    fields = tuple(field.strip() for field in payload.split("\t"))
    if len(fields) != 3:
        raise ValueError("bAbI question must contain exactly three tab-separated fields")
    question, answer, supporting_text = fields
    if not question:
        raise ValueError("bAbI question text must be nonempty")
    if not answer:
        raise ValueError("bAbI answer must be nonempty")
    if not supporting_text:
        raise ValueError("bAbI supporting fact IDs must be nonempty")

    try:
        supporting_fact_ids = tuple(int(value) for value in supporting_text.split())
    except ValueError as error:
        raise ValueError("bAbI supporting fact IDs must be integers") from error
    if any(fact_id <= 0 for fact_id in supporting_fact_ids):
        raise ValueError("bAbI supporting fact IDs must be positive")
    return question, answer, supporting_fact_ids


def parse_babi_lines(
    lines: Iterable[str],
    *,
    task_id: str,
    split: str,
    source_name: str,
) -> list[ReasoningExample]:
    """Convert bAbI source lines into standard reasoning examples."""
    for field_name, value in (
        ("task_id", task_id),
        ("split", split),
        ("source_name", source_name),
    ):
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string")
        if not value.strip():
            raise ValueError(f"{field_name} must be nonempty")
    if split not in SUPPORTED_SPLITS:
        supported = ", ".join(SUPPORTED_SPLITS)
        raise ValueError(f"unsupported split {split!r}; choose from: {supported}")
    if isinstance(lines, str):
        raise TypeError("lines must be an iterable of source lines, not one string")

    examples: list[ReasoningExample] = []
    episode_number = 0
    previous_line_id = 0
    context_facts: list[str] = []
    fact_ids: set[int] = set()
    saw_source_line = False

    for raw_line in lines:
        if not isinstance(raw_line, str):
            raise TypeError("every source line must be a string")
        if not raw_line.strip():
            raise ValueError("bAbI source must not contain blank lines")

        saw_source_line = True
        line_id, payload = split_numbered_line(raw_line)
        if line_id == 1:
            episode_number += 1
            previous_line_id = 0
            context_facts = []
            fact_ids = set()
        elif episode_number == 0:
            raise ValueError("the first bAbI episode must begin with line ID 1")

        expected_line_id = previous_line_id + 1
        if line_id != expected_line_id:
            raise ValueError(
                f"expected bAbI line ID {expected_line_id}, received {line_id}"
            )
        previous_line_id = line_id

        if "\t" not in payload:
            context_facts.append(payload)
            fact_ids.add(line_id)
            continue

        question, answer, supporting_fact_ids = parse_question_payload(payload)
        unavailable_ids = tuple(
            fact_id for fact_id in supporting_fact_ids if fact_id not in fact_ids
        )
        if unavailable_ids:
            rendered = ", ".join(str(fact_id) for fact_id in unavailable_ids)
            raise ValueError(
                f"supporting fact IDs are not prior facts in this episode: {rendered}"
            )

        context = "\n".join(context_facts)
        source_example_id = (
            f"{source_name}:episode-{episode_number:06d}:question-{line_id}"
        )
        examples.append(
            ReasoningExample(
                dataset="babi",
                task_id=task_id,
                split=split,
                context=context,
                question=question,
                answer=answer,
                supporting_fact_ids=supporting_fact_ids,
                source_length=len(context),
                source_example_id=source_example_id,
            )
        )

    if not saw_source_line:
        raise ValueError("bAbI source must contain at least one line")
    if not examples:
        raise ValueError("bAbI source must contain at least one question")
    return examples


def load_babi_file(
    path: str | Path,
    *,
    task_id: str,
    split: str,
) -> list[ReasoningExample]:
    """Read one official bAbI file and parse all of its questions."""
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a string or Path")
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"bAbI source file does not exist: {source_path}")

    with source_path.open(encoding="utf-8") as handle:
        return parse_babi_lines(
            handle,
            task_id=task_id,
            split=split,
            source_name=source_path.name,
        )
