import json
from pathlib import Path

import pytest

from tinymem.evaluation.mtp_comparison import (
    aggregate_mtp_delay_runs,
    load_mtp_delay_run,
)


def write_result(path: Path, *, seed: int, correct: int) -> None:
    path.write_text(
        json.dumps(
            {
                "seed": seed,
                "mtp_horizons": [2, 3, 4],
                "evaluation": {
                    "curve": [
                        {"label": "0-32", "correct": correct, "count": 10},
                        {"label": "33-64", "correct": 2, "count": 5},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )


def test_aggregate_mtp_delay_runs_pools_counts_and_seed_statistics(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_result(first, seed=1, correct=4)
    write_result(second, seed=2, correct=6)

    summary = aggregate_mtp_delay_runs(
        [
            load_mtp_delay_run(first, condition="adaptive_mtp4"),
            load_mtp_delay_run(second, condition="adaptive_mtp4"),
        ]
    )[0]

    assert summary["seeds"] == [1, 2]
    assert summary["seed_count"] == 2
    assert summary["curve"][0]["correct"] == 10
    assert summary["curve"][0]["accuracy"] == pytest.approx(0.5)
    assert summary["curve"][0]["seed_accuracy_mean"] == pytest.approx(0.5)
