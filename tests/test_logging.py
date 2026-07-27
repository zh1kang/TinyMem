import json
from pathlib import Path

import pytest

from tinymem.utils.logging import JSONLMetricLogger


def read_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_metric_logger_writes_one_record_per_line(tmp_path: Path) -> None:
    path = tmp_path / "run" / "metrics.jsonl"

    with JSONLMetricLogger(path) as logger:
        logger.log({"loss": 2.5}, step=0)
        logger.log({"loss": 1.5, "accuracy": 0.75}, step=1)

    records = read_records(path)
    assert len(records) == 2
    assert records[0]["loss"] == 2.5
    assert records[0]["step"] == 0
    assert records[1]["accuracy"] == 0.75
    assert all("timestamp" in record for record in records)


def test_metric_logger_appends_to_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"

    with JSONLMetricLogger(path) as logger:
        logger.log({"loss": 2.0})
    with JSONLMetricLogger(path) as logger:
        logger.log({"loss": 1.0})

    assert [record["loss"] for record in read_records(path)] == [2.0, 1.0]


@pytest.mark.parametrize("invalid_step", [True, 1.5, "1"])
def test_metric_logger_rejects_noninteger_step(
    tmp_path: Path,
    invalid_step: object,
) -> None:
    with JSONLMetricLogger(tmp_path / "metrics.jsonl") as logger:
        with pytest.raises(TypeError, match="step must be an integer or None"):
            logger.log({"loss": 1.0}, step=invalid_step)


def test_metric_logger_rejects_negative_step(tmp_path: Path) -> None:
    with JSONLMetricLogger(tmp_path / "metrics.jsonl") as logger:
        with pytest.raises(ValueError, match="step must be nonnegative"):
            logger.log({"loss": 1.0}, step=-1)


@pytest.mark.parametrize("reserved", ["timestamp", "step"])
def test_metric_logger_rejects_reserved_metric_names(
    tmp_path: Path,
    reserved: str,
) -> None:
    with JSONLMetricLogger(tmp_path / "metrics.jsonl") as logger:
        with pytest.raises(ValueError, match="reserved fields"):
            logger.log({reserved: 1})


def test_metric_logger_rejects_non_json_numbers(tmp_path: Path) -> None:
    with JSONLMetricLogger(tmp_path / "metrics.jsonl") as logger:
        with pytest.raises(ValueError):
            logger.log({"loss": float("nan")})


def test_metric_logger_rejects_writes_after_close(tmp_path: Path) -> None:
    logger = JSONLMetricLogger(tmp_path / "metrics.jsonl")
    logger.close()

    with pytest.raises(RuntimeError, match="logger is closed"):
        logger.log({"loss": 1.0})
