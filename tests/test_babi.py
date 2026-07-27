from pathlib import Path

import pytest

from tinymem.data.babi import (
    load_babi_file,
    parse_babi_lines,
    parse_question_payload,
    split_numbered_line,
)


def test_split_numbered_line_preserves_payload() -> None:
    assert split_numbered_line("12 Mary dropped the football.\n") == (
        12,
        "Mary dropped the football.",
    )


@pytest.mark.parametrize("line", ["missing-space", "0 fact", "x fact", "1  "])
def test_split_numbered_line_rejects_malformed_line(line: str) -> None:
    with pytest.raises(ValueError):
        split_numbered_line(line)


def test_parse_question_payload_preserves_supporting_order() -> None:
    assert parse_question_payload("Where is the football? \tgarden\t12 6") == (
        "Where is the football?",
        "garden",
        (12, 6),
    )


@pytest.mark.parametrize(
    "payload",
    [
        "Where is Mary?",
        "Where is Mary?\tbathroom",
        "Where is Mary?\tbathroom\t1\textra",
        "\tbathroom\t1",
        "Where is Mary?\t\t1",
        "Where is Mary?\tbathroom\t",
        "Where is Mary?\tbathroom\tx",
        "Where is Mary?\tbathroom\t0",
    ],
)
def test_parse_question_payload_rejects_malformed_question(payload: str) -> None:
    with pytest.raises(ValueError):
        parse_question_payload(payload)


def test_parse_babi_lines_creates_one_example_per_question() -> None:
    lines = [
        "1 Mary moved to the bathroom.\n",
        "2 John went to the hallway.\n",
        "3 Where is Mary? \tbathroom\t1\n",
        "4 Daniel went to the office.\n",
        "5 Where is Daniel? \toffice\t4\n",
    ]

    examples = parse_babi_lines(
        lines,
        task_id="qa1",
        split="train",
        source_name="qa1_train.txt",
    )

    assert len(examples) == 2
    assert examples[0].context == (
        "Mary moved to the bathroom.\nJohn went to the hallway."
    )
    assert examples[0].question == "Where is Mary?"
    assert examples[0].answer == "bathroom"
    assert examples[0].supporting_fact_ids == (1,)
    assert "Where is Mary?" not in examples[1].context
    assert examples[1].context.endswith("Daniel went to the office.")
    assert examples[1].source_length == len(examples[1].context)


def test_parse_babi_lines_resets_context_at_episode_boundary() -> None:
    lines = [
        "1 Mary moved to the bathroom.\n",
        "2 Where is Mary? \tbathroom\t1\n",
        "1 Sandra travelled to the office.\n",
        "2 Where is Sandra? \toffice\t1\n",
    ]

    examples = parse_babi_lines(
        lines,
        task_id="qa1",
        split="test",
        source_name="qa1_test.txt",
    )

    assert examples[1].context == "Sandra travelled to the office."
    assert "Mary" not in examples[1].context
    assert examples[0].source_example_id != examples[1].source_example_id


def test_parse_babi_lines_preserves_qa2_dependency_order() -> None:
    lines = [
        "1 Mary got the football.\n",
        "2 Mary went to the garden.\n",
        "3 Mary dropped the football.\n",
        "4 Where is the football? \tgarden\t3 2\n",
    ]

    example = parse_babi_lines(
        lines,
        task_id="qa2",
        split="validation",
        source_name="qa2_valid.txt",
    )[0]

    assert example.supporting_fact_ids == (3, 2)


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        ([], "at least one line"),
        (["1 Mary moved to the bathroom.\n"], "at least one question"),
        (["\n"], "must not contain blank lines"),
        (["2 Mary moved to the bathroom.\n"], "must begin with line ID 1"),
        (
            ["1 Mary moved to the bathroom.\n", "3 Where is Mary? \tbathroom\t1\n"],
            "expected bAbI line ID 2",
        ),
        (
            ["1 Mary moved to the bathroom.\n", "2 Where is Mary? \tbathroom\t2\n"],
            "not prior facts",
        ),
    ],
)
def test_parse_babi_lines_rejects_invalid_source(
    lines: list[str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_babi_lines(
            lines,
            task_id="qa1",
            split="train",
            source_name="qa1_train.txt",
        )


def test_load_babi_file_reads_utf8_source(tmp_path: Path) -> None:
    path = tmp_path / "qa1_train.txt"
    path.write_text(
        "1 Mary moved to the bathroom.\n"
        "2 Where is Mary? \tbathroom\t1\n",
        encoding="utf-8",
    )

    examples = load_babi_file(path, task_id="qa1", split="train")

    assert len(examples) == 1
    assert examples[0].source_example_id.startswith("qa1_train.txt:")


def test_load_babi_file_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_babi_file(tmp_path / "missing.txt", task_id="qa1", split="train")
