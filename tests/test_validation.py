import pytest

from tinymem.data.babilong import parse_babilong_records
from tinymem.data.symbolic_world import OracleResult
from tinymem.data.validation import (
    measure_evidence_delay,
    require_evidence_outside_local_window,
)


def make_example():
    context = "Mary went to the garden." + ("x" * 20)
    return parse_babilong_records(
        [{"input": context, "question": "Where is Mary?", "target": "garden"}],
        task_id="qa1",
        split="test",
        source_name="1k.json",
    )[0]


def test_measure_evidence_delay_uses_latest_support_end() -> None:
    delay = measure_evidence_delay(make_example(), OracleResult("garden", (1,)))
    assert delay.characters == 20
    assert delay.trailing_evidence_facts == 0


def test_require_evidence_outside_local_window_checks_boundary() -> None:
    require_evidence_outside_local_window(
        make_example(), OracleResult("garden", (1,)), local_window_characters=20
    )
    with pytest.raises(ValueError, match="inside the local window"):
        require_evidence_outside_local_window(
            make_example(), OracleResult("garden", (1,)), local_window_characters=21
        )
