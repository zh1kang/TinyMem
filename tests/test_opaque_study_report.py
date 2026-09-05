import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.report_opaque_study import ANALYSIS_SOURCES, DISPLAY_METHODS, build_report, correct_counts, validate_analysis_sources


SEEDS = (1337, 2027, 4099)


def fixture():
    frozen = {"protocol": "opaque_qa1_fixed_byte_comparison_v1", "stored_bytes": 66,
              "statistics": {"worlds": 128}, "seeds": list(SEEDS), "bootstrap_resamples": 100000,
              "bootstrap_seed": 20260905, "statistical_comparisons": [],
              "source_sha256": {path: hashlib.sha256(path.encode()).hexdigest() for path in ANALYSIS_SOURCES}}
    accuracy = []
    for method, condition, _, _ in DISPLAY_METHODS:
        for category, total in (("opaque_qa1_known", 1024), ("opaque_qa1_missing", 128)):
            if category.endswith("missing"):
                counts = dict.fromkeys(SEEDS, 128)
            elif method == "query_pool":
                counts = {1337: 768, 2027: 776, 4099: 760}
            elif condition in ("fingerprint", "full_history"):
                counts = dict.fromkeys(SEEDS, 1024)
            else:
                counts = dict.fromkeys(SEEDS, 0 if condition == "drop" else 512)
            values = {str(seed): count / total for seed, count in counts.items()}
            accuracy.append({"method": method, "condition": f"opaque:{condition}", "category": category,
                             "worlds": 128, "queries_per_checkpoint": total, "per_seed_accuracy": values,
                             "mean_accuracy": statistics.fmean(values.values())})
    rows = {(row["method"], row["condition"], row["category"]): row for row in accuracy}
    comparisons = []
    right_sides = [["mean_pool", "opaque:normal"], *[["baseline", f"opaque:{name}"] for name in (
        "recent_native", "recent_vocabulary", "latest_vocabulary", "latest_template")]]
    for family, category in (("known_superiority", "opaque_qa1_known"), ("absent_noninferiority", "opaque_qa1_missing")):
        for right in right_sides:
            expected = {"family": family, "left": ["query_pool", "opaque:normal"], "right": right,
                        "category": category, "confidence": 0.99}
            frozen["statistical_comparisons"].append(expected)
            left_values = rows[("query_pool", "opaque:normal", category)]["per_seed_accuracy"]
            right_values = rows[(*right, category)]["per_seed_accuracy"]
            gains = {str(seed): left_values[str(seed)] - right_values[str(seed)] for seed in SEEDS}
            difference = statistics.fmean(gains.values())
            comparisons.append(expected | {"difference": difference, "lower": difference - 0.01,
                                           "upper": difference + 0.01, "histories": 128,
                                           "optimization_seeds": list(SEEDS), "resamples": 100000,
                                           "bootstrap_seed": 20260905, "per_seed_difference": gains})
    return frozen, {"accuracy": accuracy, "comparisons": comparisons,
                    "provenance": {"source_sha256": dict(frozen["source_sha256"])}}


def test_report_uses_frozen_design_and_keeps_references_separate():
    frozen, summary = fixture()
    report = build_report(frozen, summary)
    assert report["assessment"]["overall_result"] == "positive"
    assert report["assessment"]["practical_positive"]
    assert report["positive_known_gaps_in_every_seed"]
    assert len(report["accuracy"]) == 9 and len(report["known_contrasts"]) == 5
    assert "fingerprint" not in report["known_contrasts"]
    learned = report["accuracy"][0]
    assert learned["known_optimization_seed_sd"] == statistics.stdev([768 / 1024, 776 / 1024, 760 / 1024])
    assert report["accuracy"][2]["known_optimization_seed_sd"] is None
    assert report["known_contrasts"]["mean_pool"]["optimization_difference_sd"] == learned["known_optimization_seed_sd"]
    assert report["absent_contrasts"]["mean_pool"]["optimization_difference_sd"] == 0
    assert report["absent_per_seed_gains"]["mean_pool"] == dict.fromkeys(map(str, SEEDS), 0)
    frozen["bootstrap_seed"] = 17
    with pytest.raises(ValueError, match="declared"):
        build_report(frozen, summary)


def test_report_rejects_consistent_contrasts_that_disagree_with_the_accuracy_table():
    frozen, summary = fixture()
    summary["comparisons"][0]["difference"] += 0.01
    summary["comparisons"][0]["per_seed_difference"] = {
        seed: value + 0.01 for seed, value in summary["comparisons"][0]["per_seed_difference"].items()}
    with pytest.raises(ValueError, match="accuracy table"):
        build_report(frozen, summary)


@pytest.mark.parametrize("change", ["missing_comparison", "duplicate_accuracy", "wrong_category", "wrong_condition",
                                    "wrong_mean", "bad_world_count", "baseline_not_deterministic"])
def test_invalid_report_records_fail(change):
    frozen, summary = fixture()
    if change == "missing_comparison":
        summary["comparisons"].pop()
    elif change == "duplicate_accuracy":
        summary["accuracy"].append(summary["accuracy"][0])
    elif change == "wrong_category":
        summary["comparisons"][0]["category"] = "opaque_qa1_missing"
    elif change == "wrong_condition":
        summary["comparisons"][0]["right"] = ["mean_pool", "short:normal"]
    elif change == "wrong_mean":
        summary["accuracy"][0]["mean_accuracy"] = 0.5
    elif change == "bad_world_count":
        summary["accuracy"][0]["worlds"] = 128.0
    else:
        row = next(row for row in summary["accuracy"] if row["condition"] == "opaque:full_history")
        row["per_seed_accuracy"]["1337"] = 0.5
        row["mean_accuracy"] = statistics.fmean(row["per_seed_accuracy"].values())
    with pytest.raises(ValueError):
        build_report(frozen, summary)


@pytest.mark.parametrize("value", [True, float("nan"), 1.1, "0.5", 0.50001])
def test_invalid_or_fractional_counts_fail(value):
    row = {"queries_per_checkpoint": 128, "per_seed_accuracy": dict.fromkeys(map(str, SEEDS), value), "mean_accuracy": value}
    with pytest.raises(ValueError):
        correct_counts(row, total=128, seeds=SEEDS)


def test_analysis_sources_accept_absolute_relative_aliases():
    frozen, summary = fixture()
    summary["provenance"]["source_sha256"] = {
        str(Path(path).resolve()): value for path, value in frozen["source_sha256"].items()
    }
    frozen["source_sha256"]["src/tinymem/evaluation/association_study.py"] = "unrelated frozen source"
    validate_analysis_sources(frozen, summary)


@pytest.mark.parametrize("change", ["wrong_hash", "missing_source", "extra_source", "duplicate_alias", "missing_frozen_source"])
def test_analysis_sources_reject_unbound_analysis(change):
    frozen, summary = fixture()
    sources = summary["provenance"]["source_sha256"]
    path = ANALYSIS_SOURCES[0]
    if change == "wrong_hash":
        sources[path] = "0" * 64
    elif change == "missing_source":
        sources.pop(path)
    elif change == "extra_source":
        sources["extra.py"] = "0" * 64
    elif change == "duplicate_alias":
        sources[str(Path(path).resolve())] = sources[path]
    else:
        frozen["source_sha256"].pop(path)
    with pytest.raises(ValueError, match="source"):
        validate_analysis_sources(frozen, summary)


def test_cli_writes_figures_from_synthetic_fixture_only(tmp_path):
    frozen, summary = fixture()
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(frozen))
    summary["provenance"]["study_protocol_sha256"] = hashlib.sha256(protocol.read_bytes()).hexdigest()
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary))
    output = tmp_path / "report"
    command = [sys.executable, "scripts/report_opaque_study.py", "--study-protocol", str(protocol),
               "--summary", str(summary_path), "--output", str(output),
               "--figure-label", "synthetic fixture - not research results"]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout)["overall_result"] == "positive"
    report = json.loads((output / "report.json").read_text())
    assert report["figure_label"] == "synthetic fixture - not research results"
    assert report["provenance"]["summary_sha256"] == hashlib.sha256(summary_path.read_bytes()).hexdigest()
    for name in ("accuracy.png", "accuracy.svg", "known_contrasts.png", "known_contrasts.svg"):
        assert (output / name).stat().st_size > 1000
    repeated = subprocess.run(command, capture_output=True, text=True)
    assert repeated.returncode != 0 and "FileExistsError" in repeated.stderr
