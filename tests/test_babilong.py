import json
from pathlib import Path

import pytest

from tinymem.data.babilong import load_babilong_file, parse_babilong_records


def make_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "input": "Book text. Mary journeyed to the bathroom. More book text.\n",
        "question": "Where is Mary? ",
        "target": "bathroom",
    }
    record.update(overrides)
    return record


def test_parse_babilong_records_preserves_long_context_exactly() -> None:
    record = make_record()

    example = parse_babilong_records(
        [record],
        task_id="qa1",
        split="test",
        source_name="1k.json",
    )[0]

    assert example.dataset == "babilong"
    assert example.task_id == "qa1"
    assert example.context == record["input"]
    assert example.source_length == len(example.context)
    assert example.question == "Where is Mary?"
    assert example.answer == "bathroom"
    assert example.supporting_fact_ids is None
    assert example.context_fact_ids is None
    assert example.source_example_id == "1k.json:record-000001"


def test_parse_babilong_records_assigns_stable_unique_ids() -> None:
    examples = parse_babilong_records(
        [make_record(), make_record(target="hallway")],
        task_id="qa1",
        split="test",
        source_name="1k.json",
    )

    assert examples[0].source_example_id == "1k.json:record-000001"
    assert examples[1].source_example_id == "1k.json:record-000002"


def test_parse_babilong_records_allows_additional_source_fields() -> None:
    example = parse_babilong_records(
        [make_record(extra_metadata=1)],
        task_id="qa1",
        split="test",
        source_name="1k.json",
    )[0]

    assert example.answer == "bathroom"


def test_parse_babilong_records_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one record"):
        parse_babilong_records(
            [],
            task_id="qa1",
            split="test",
            source_name="1k.json",
        )


@pytest.mark.parametrize("record", [None, [], "record"])
def test_parse_babilong_records_rejects_nonobject_record(record: object) -> None:
    with pytest.raises(TypeError, match="must be a JSON object"):
        parse_babilong_records(
            [record],
            task_id="qa1",
            split="test",
            source_name="1k.json",
        )


@pytest.mark.parametrize("missing_field", ["input", "question", "target"])
def test_parse_babilong_records_rejects_missing_field(missing_field: str) -> None:
    record = make_record()
    del record[missing_field]

    with pytest.raises(ValueError, match=f"missing fields: {missing_field}"):
        parse_babilong_records(
            [record],
            task_id="qa1",
            split="test",
            source_name="1k.json",
        )


@pytest.mark.parametrize("field_name", ["input", "question", "target"])
def test_parse_babilong_records_rejects_nonstring_field(field_name: str) -> None:
    with pytest.raises(TypeError, match=f"field '{field_name}'.*must be a string"):
        parse_babilong_records(
            [make_record(**{field_name: 1})],
            task_id="qa1",
            split="test",
            source_name="1k.json",
        )


@pytest.mark.parametrize("field_name", ["input", "question", "target"])
def test_parse_babilong_records_rejects_empty_field(field_name: str) -> None:
    with pytest.raises(ValueError, match=f"field '{field_name}'.*must be nonempty"):
        parse_babilong_records(
            [make_record(**{field_name: "  "})],
            task_id="qa1",
            split="test",
            source_name="1k.json",
        )


def test_load_babilong_file_reads_json_records(tmp_path: Path) -> None:
    path = tmp_path / "1k.json"
    path.write_text(json.dumps([make_record()]), encoding="utf-8")

    examples = load_babilong_file(path, task_id="qa1", split="test")

    assert len(examples) == 1
    assert examples[0].source_example_id == "1k.json:record-000001"


def test_load_babilong_file_rejects_nonarray_json(tmp_path: Path) -> None:
    path = tmp_path / "1k.json"
    path.write_text(json.dumps(make_record()), encoding="utf-8")

    with pytest.raises(TypeError, match="iterable of JSON objects"):
        load_babilong_file(path, task_id="qa1", split="test")


def test_load_babilong_file_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "1k.json"
    path.write_text("not JSON", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid BABILong JSON"):
        load_babilong_file(path, task_id="qa1", split="test")


def test_load_babilong_file_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_babilong_file(tmp_path / "missing.json", task_id="qa1", split="test")
