"""Adapter for BABILong JSON benchmark files."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from tinymem.data.schema import ReasoningExample, SUPPORTED_SPLITS


REQUIRED_RECORD_FIELDS = ("input", "question", "target")


def parse_babilong_records(
    records: Iterable[Mapping[str, object]],
    *,
    task_id: str,
    split: str,
    source_name: str,
) -> list[ReasoningExample]:
    """Convert BABILong JSON records into standard reasoning examples."""
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
    if isinstance(records, (str, bytes, Mapping)):
        raise TypeError("records must be an iterable of JSON objects")

    examples: list[ReasoningExample] = []
    for record_index, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise TypeError(f"BABILong record {record_index} must be a JSON object")

        missing_fields = [field for field in REQUIRED_RECORD_FIELDS if field not in record]
        if missing_fields:
            rendered = ", ".join(missing_fields)
            raise ValueError(
                f"BABILong record {record_index} is missing fields: {rendered}"
            )

        values: dict[str, str] = {}
        for field_name in REQUIRED_RECORD_FIELDS:
            value = record[field_name]
            if not isinstance(value, str):
                raise TypeError(
                    f"BABILong record {record_index} field {field_name!r} "
                    "must be a string"
                )
            if not value.strip():
                raise ValueError(
                    f"BABILong record {record_index} field {field_name!r} "
                    "must be nonempty"
                )
            values[field_name] = value

        context = values["input"]
        source_example_id = f"{source_name}:record-{record_index:06d}"
        examples.append(
            ReasoningExample(
                dataset="babilong",
                task_id=task_id,
                split=split,
                context=context,
                question=values["question"].strip(),
                answer=values["target"].strip(),
                supporting_fact_ids=None,
                source_length=len(context),
                source_example_id=source_example_id,
                context_fact_ids=None,
            )
        )

    if not examples:
        raise ValueError("BABILong source must contain at least one record")
    return examples


def load_babilong_file(
    path: str | Path,
    *,
    task_id: str,
    split: str,
) -> list[ReasoningExample]:
    """Read one BABILong JSON file and parse all records."""
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a string or Path")
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"BABILong source file does not exist: {source_path}")

    try:
        with source_path.open(encoding="utf-8") as handle:
            records = json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid BABILong JSON in {source_path}") from error

    return parse_babilong_records(
        records,
        task_id=task_id,
        split=split,
        source_name=source_path.name,
    )
