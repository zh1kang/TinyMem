"""Exercise paired reporting against sealed, real tiny-reader runs."""
import json

import pytest

from test_readout_runner import tiny_reader
from test_readout_experiment import inputs
from tinymem.research.readout_experiment import run_arm


def test_real_paired_report(tmp_path, tiny_reader):
    from tinymem.research.readout_paired_report import build_paired_report

    paths = []
    for arm in ("affine", "gelu"):
        path = tmp_path / arm
        run_arm(tiny_reader, inputs(tiny_reader), path, kind=arm, seed=17,
                steps=1, learning_rate=.001, weight_decay=.01,
                max_new_tokens=1,
                input_identity={"evidence_kind": "tiny_random_cpu_test"})
        paths.append(path)
    report = build_paired_report(paths, resamples=20, bootstrap_seed=7)
    assert report["seeds"] == [17]
    assert len(report["runs"]) == 2
    assert report["persistent_bytes"] == 66
    for run in report["runs"]:
        costs = run["costs"]
        assert costs["elapsed_seconds"] >= 0
        assert costs["shared_parameter_bytes"] == 4 * sum(costs["shared_parameters"].values())
        assert costs["persistent_bytes_per_history"] == 66
        assert costs["training_peak_memory_bytes"] is None
        assert costs["temporary_vector_bytes_max"] > 0
    assert report["full_text_qualification"] == "rule_not_frozen"
    assert set(report["comparisons"]) == {"train", "development"}
    assert report["comparisons"]["development"]["known"]["seed_sd"] is None
    assert report["training_marginal_answer_metrics"]["train"]["queries"] == 20
    assert report["training_marginal_answer_metrics"]["development"]["queries"] == 20
    assert report["training_marginal_answer_mode"]
    for split in ("train", "development"):
        for arm in ("affine", "gelu"):
            controls = report["control_comparisons"][split][arm]
            assert set(controls) == {"zero", "no_memory", "shuffled"}
            for reference, comparison in controls.items():
                assert comparison["contrast"] == {
                    "condition": "normal", "reference": reference,
                }
                assert comparison["metrics"]["known"]["seed_sd"] is None
    json.dumps(report, allow_nan=False)
    from tinymem.research.readout_paired_report import write_paired_report

    destination = tmp_path / "report"
    write_paired_report(report, destination)
    assert json.loads((destination / "report.json").read_text()) == report
    markdown = (destination / "report.md").read_text()
    assert "rule_not_frozen" in markdown
    assert "66" in markdown
    assert "unmeasured" in markdown
    assert "Jointly trained arms" in markdown
    with pytest.raises(FileExistsError):
        write_paired_report(report, destination)
    with pytest.raises(ValueError, match="both arms"):
        build_paired_report(paths[:1], resamples=20)
