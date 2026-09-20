"""Behavioral checks for bounded delta-fact report aggregation."""

from copy import deepcopy
from dataclasses import asdict
import json

import pytest

from tinymem.research.delta_fact_report import (
    _article_only_diagnostics,
    _bootstrap_difference,
    _comparison,
    _validate_donor_rows,
    _expected_case_metadata,
    _validate_expected_case_metadata,
    _validate_row,
    _summary,
    _validate_case_coverage,
    _validate_cross_cell_rows,
)
from tinymem.research.delta_fact_data import build_dataset


def _row(episode: str, entity: int, *, endpoint: int = 8, condition: str = "repeat",
         target: int | None = 0, scope: str = "unspoken", before: bool = False,
         after: bool = False, truth: int = 0) -> dict:
    correct = before if endpoint == 8 else after
    return {
        "key": f"{episode}/write{endpoint}/entity{entity}", "episode_id": episode,
        "prefix_id": "test/p0000", "wording": "familiar", "condition": condition,
        "target": target, "after_write": endpoint, "entity": entity, "scope": scope,
        "truth_bit": truth, "answer": "bathroom", "probe_bit": truth if correct else 1 - truth, "probe_correct": correct,
        "reads": {mode: {"prediction": "bathroom" if correct else "office", "generated_ids": [1],
                           "input_positions": 3, "memory_positions": 2,
                           "native_envelope_tokens": 1, "correct": correct}
                  for mode in ("memory", "zero", "donor")},
        "donor_prefix": "test/p0001", "donor_truth_agrees": True,
        "base_full_text_correct": correct, "explicit_correct": True,
    }


def test_change_contrast_retains_initial_and_final_accuracy_context():
    cells = {}
    for writer, initial, final in (("delta", False, False), ("gated", True, False)):
        cells[(writer, 8, "11")] = [_row("e0", 1, endpoint=8, before=initial, after=final),
                                    _row("e0", 1, endpoint=16, before=initial, after=final)]
    result = _comparison(cells, width=8, wording="familiar", condition="repeat", scope="unspoken",
                         method="lm", left_writer="delta", right_writer="gated", endpoint=16,
                         settings={"bootstrap_samples": 10, "bootstrap_seed": 8237})
    assert result["mean_seed_difference_pp"] == 100
    assert result["accuracy_context_by_seed"]["11"]["delta"] == {
        "n": 1, "before_correct": 0, "after_correct": 0}


def test_row_validation_covers_read_metadata_probe_and_explicit_baseline():
    row = _row("e0", 0, scope="target")
    bad_probe = deepcopy(row)
    bad_probe["probe_bit"] = bad_probe["truth_bit"]
    with pytest.raises(ValueError, match="probe"):
        _validate_row(bad_probe)
    bad_read = deepcopy(row)
    bad_read["reads"]["memory"].pop("generated_ids")
    with pytest.raises(ValueError, match="read metadata"):
        _validate_row(bad_read)
    bad_explicit = deepcopy(row)
    bad_explicit["explicit_correct"] = False
    with pytest.raises(ValueError, match="explicit"):
        _validate_row(bad_explicit)
    bad_positions = deepcopy(row)
    bad_positions["reads"]["memory"]["memory_positions"] = 0
    with pytest.raises(ValueError, match="read metadata"):
        _validate_row(bad_positions)
    bad_correctness = deepcopy(row)
    bad_correctness["reads"]["memory"]["correct"] = True
    with pytest.raises(ValueError, match="correctness"):
        _validate_row(bad_correctness)


def test_article_only_diagnostic_does_not_relabel_wrong_rooms():
    row = _row("e0", 0, scope="target", before=False, after=False)
    row["reads"]["memory"]["prediction"] = "The bathroom"
    row["reads"]["zero"]["prediction"] = "The kitchen"
    row["reads"]["donor"]["prediction"] = "The bathroom because I remember it"
    reference = {row["key"]: {"base_full_text": {"prediction": "The bathroom"}}}
    result = _article_only_diagnostics([row], reference, "familiar")
    assert result["lm"] == {"n": 1, "article_only_mismatches": 1}
    assert result["zero"]["article_only_mismatches"] == 0
    assert result["donor"]["article_only_mismatches"] == 0
    assert result["base_full_text"]["article_only_mismatches"] == 1


def test_donor_truth_agreement_uses_the_declared_no_write_prefix():
    row = _row("e0", 1)
    _validate_donor_rows([row], {("test/p0001", "familiar"): (0, 0, 0, 0)})
    row["donor_truth_agrees"] = False
    with pytest.raises(ValueError, match="donor_truth_agrees"):
        _validate_donor_rows([row], {("test/p0001", "familiar"): (0, 0, 0, 0)})


def test_summary_counts_repairs_damage_and_null_conditional_rates():
    rows = [
        _row("e0", 0, before=True, after=True),
        _row("e0", 1, before=True, after=False),
        _row("e0", 2, before=False, after=True),
        _row("e0", 3, before=False, after=False),
    ]
    before = rows
    after = [dict(row, key=row["key"].replace("write8", "write16"), after_write=16,
                  reads={mode: {"correct": value} for mode, value in row["reads"].items()},
                  probe_correct=row["probe_correct"], base_full_text_correct=row["base_full_text_correct"])
             for row in rows]
    for row, value in zip(after, (True, False, True, False), strict=True):
        row["reads"] = {mode: {"correct": value} for mode in ("memory", "zero", "donor")}
        row["probe_correct"] = value
        row["base_full_text_correct"] = value
    result = _summary(before + after, condition="repeat", wording="familiar", scope="unspoken",
                      endpoint=16, method="lm")
    assert (result["n"], result["before_correct"], result["after_correct"]) == (4, 2, 2)
    assert (result["retained_correct"], result["remained_wrong"], result["damage"], result["repair"]) == (1, 1, 1, 1)
    assert result["change_pp"] == 0

    all_correct = [_row("e1", entity, before=True, after=True) for entity in range(4)]
    all_after = [dict(row, key=row["key"].replace("write8", "write16"), after_write=16)
                 for row in all_correct]
    result = _summary(all_correct + all_after, condition="repeat", wording="familiar", scope="unspoken",
                      endpoint=16, method="lm")
    assert result["rates"]["repair"] is None
    assert result["rates"]["remained_wrong"] is None


def test_bootstrap_keeps_constant_paired_prefix_effect_exact():
    left = {(prefix, seed): 0.5 for seed in (11, 12, 13) for prefix in ("p0", "p1", "p2")}
    right = {(prefix, seed): 0.25 for seed in (11, 12, 13) for prefix in ("p0", "p1", "p2")}
    result = _bootstrap_difference(left, right, resamples=101, seed=8237)
    assert result["per_seed_difference_pp"] == {"11": 25.0, "12": 25.0, "13": 25.0}
    assert result["mean_seed_difference_pp"] == 25.0
    assert result["interval_pp"] == [25.0, 25.0]
    assert result["confidence"] == 0.99


def test_cross_cell_validation_rejects_missing_or_tampered_pairs():
    rows = []
    for endpoint in (8, 9, 16):
        rows.extend(_row("e0", entity, endpoint=endpoint, after=endpoint == 16)
                    for entity in range(4))
    _validate_case_coverage(rows)

    with pytest.raises(ValueError, match="coverage"):
        _validate_case_coverage(rows[:-1])

    missing = rows[:-1]
    with pytest.raises(ValueError, match="matching case keys"):
        _validate_cross_cell_rows([rows, missing])

    tampered = deepcopy(rows)
    tampered[-1]["truth_bit"] = 1
    # Endpoint 16 is unspoken and must retain the prefix truth.
    with pytest.raises(ValueError, match="metadata"):
        _validate_cross_cell_rows([rows, tampered])


def test_cross_cell_validation_preserves_weak_cell_coverage():
    rows = []
    for endpoint in (8, 9, 16):
        rows.extend(_row("e0", entity, endpoint=endpoint, after=endpoint == 16)
                    for entity in range(4))
    weak = deepcopy(rows)
    weak[0]["reads"]["memory"]["correct"] = False
    _validate_cross_cell_rows([rows, weak])


def test_row_schema_rejects_missing_field_even_with_extra_field_and_accepts_null_residual(tmp_path):
    row = _row("e0", 0, scope="target")
    row["last_residual_norm"] = None
    _validate_row(row)
    missing = dict(row)
    missing.pop("probe_correct")
    missing["unexpected"] = True
    with pytest.raises(ValueError, match="missing required"):
        _validate_row(missing)


def test_expected_dataset_panel_rejects_an_omitted_branch(tmp_path):
    dataset = build_dataset(seed=13, train_prefixes=16, validation_prefixes=16, test_prefixes=16)
    (tmp_path / "dataset.json").write_text(json.dumps(asdict(dataset)))
    expected = _expected_case_metadata(tmp_path, {"evaluation_prefix_limit": 1})
    rows = [
        {"episode_id": key[0], "after_write": key[1], "entity": key[2],
         "prefix_id": value[0], "wording": value[1], "condition": value[2],
         "target": value[3], "scope": value[4], "truth_bit": value[5], "answer": value[6]}
        for key, value in expected.items()
    ]
    omitted = [row for row in rows if row["condition"] != "repeat"]
    with pytest.raises(ValueError, match="declared dataset panel"):
        _validate_expected_case_metadata(omitted, expected)
