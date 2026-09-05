import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tinymem.research.study_runtime import PORTABLE_SOURCES, REPOSITORY


def test_rsync_filter_keeps_code_and_complete_inputs_only(tmp_path):
    if shutil.which("rsync") is None:
        pytest.skip("rsync is not installed")
    source, target = tmp_path / "source", tmp_path / "target"
    study = "artifacts/predictions/opaque_memory_study_20260905"
    included = [".git/HEAD", "src/tinymem/__init__.py", "scripts/della.slurm", "data/raw/pretrained/qwen3-1.7b/model.safetensors",
                f"{study}/protocol.json", f"{study}/query_pool_seed_1337/results.json",
                "artifacts/predictions/evaluate_opaque_memory.py",
                "artifacts/predictions/opaque_qa1_data_20260905/confirmation.json"]
    excluded = [".venv/bin/python", ".env", "src/__pycache__/module.pyc", "logs/job.out",
                "data/raw/pretrained/qwen3-1.7b/.cache/file", f"{study}/confirmation/baselines/predictions.jsonl",
                f"{study}/mean_pool_seed_9999/metrics.jsonl", "artifacts/predictions/opaque_training_fit_20260905/states.json"]
    for name in included + excluded:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    subprocess.run(["rsync", "-a", "--filter", f"merge {REPOSITORY / 'scripts/della-rsync.filter'}",
                    str(source) + "/", str(target) + "/"], check=True, capture_output=True, text=True)
    assert all((target / name).is_file() for name in included)
    assert not any((target / name).exists() for name in excluded)
    relative_target = tmp_path / "relative-target"
    subprocess.run(["rsync", "-aR", "--filter", f"merge {REPOSITORY / 'scripts/della-rsync.filter'}",
                    "data/", "artifacts/", str(relative_target) + "/"], cwd=source,
                   check=True, capture_output=True, text=True)
    assert (relative_target / included[3]).is_file()
    assert (relative_target / included[4]).is_file()
    assert not any((relative_target / name).exists() for name in excluded)


def test_slurm_scripts_parse_and_unknown_stage_fails():
    subprocess.run(["bash", "-n", "scripts/della.slurm", "scripts/della_run.sh"], cwd=REPOSITORY, check=True)
    result = subprocess.run(["bash", "scripts/della_run.sh", "typo", "artifacts/predictions/unused-test"],
                            cwd=REPOSITORY, capture_output=True, text=True)
    assert result.returncode == 2 and "usage:" in result.stderr
    assert not (REPOSITORY / "artifacts/predictions/unused-test").exists()


@pytest.fixture
def relocated(tmp_path):
    root = tmp_path / "TinyMem"
    shutil.copytree(REPOSITORY / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    for name in PORTABLE_SOURCES:
        if name.startswith("scripts/"):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPOSITORY / name, path)
    return root


def test_relocated_cli_resolves_historical_paths_and_refuses_optimized_python(relocated):
    environment = os.environ | {"PYTHONPATH": str(relocated / "src")}
    code = "from tinymem.research.study_runtime import sha256; print(sha256('/Users/caleb/TinyMem/src/tinymem/research/pretrained.py'))"
    result = subprocess.run([sys.executable, "-c", code], cwd=relocated, env=environment,
                            capture_output=True, text=True, check=True)
    assert len(result.stdout.strip()) == 64
    result = subprocess.run([sys.executable, "-O", "-c", "from tinymem.research.study_runtime import check_repository; check_repository()"],
                            cwd=relocated, env=environment, capture_output=True, text=True)
    assert result.returncode != 0 and "do not use python -O" in result.stderr


@pytest.mark.parametrize("module", ["train", "baselines", "memory", "training_fit", "oracle", "smoke"])
def test_portable_cli_exposes_device_without_loading_model(module):
    pytest.importorskip("peft")
    pytest.importorskip("transformers")
    result = subprocess.run([sys.executable, "-m", f"scripts.opaque.{module}", "--help"],
                            cwd=REPOSITORY, capture_output=True, text=True, check=True)
    assert "--device {cuda,mps,cpu}" in result.stdout


def test_relocated_synthetic_predictions_aggregate_and_report(relocated):
    pytest.importorskip("peft")
    pytest.importorskip("transformers")
    environment = os.environ | {"PYTHONPATH": str(relocated / "src")}
    command = [sys.executable, "-c", "import json; from tinymem.research.study_runtime import execution_record, prepare_device; print(json.dumps(execution_record(prepare_device('cpu'))))"]
    execution = json.loads(subprocess.run(command, cwd=relocated, env=environment, capture_output=True,
                                          text=True, check=True).stdout)

    def save(path, value, *, lines=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in value) if lines else json.dumps(value))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    seeds = [1337, 2027, 4099]
    writers = ["query_pool", "mean_pool"]
    run_names = [f"training/{writer}_seed_{seed}" for writer in writers for seed in seeds]
    baseline_conditions = [f"opaque:{name}" for name in (
        "drop", "recent_native", "recent_vocabulary", "latest_vocabulary", "latest_template", "fingerprint", "full_history")]
    comparisons = []
    for family, category in (("known_superiority", "opaque_qa1_known"), ("absent_noninferiority", "opaque_qa1_missing")):
        for comparator in ("mean_pool", "recent_native", "recent_vocabulary", "latest_vocabulary", "latest_template"):
            comparisons.append({"family": family, "category": category, "left": ["query_pool", "opaque:normal"],
                                "right": ["mean_pool", "opaque:normal"] if comparator == "mean_pool" else ["baseline", f"opaque:{comparator}"],
                                "confidence": 0.99})
    (relocated / "historical.py").write_text("# synthetic fixture, not experiment evidence\n")
    frozen = {"protocol": "opaque_qa1_fixed_byte_comparison_v1", "stored_bytes": 66,
              "runs": run_names, "seeds": seeds, "writers": writers, "statistics": {"worlds": 128},
              "bootstrap_resamples": 20, "bootstrap_seed": 13, "statistical_comparisons": comparisons,
              "baseline_conditions": baseline_conditions, "memory_conditions": ["opaque:normal"],
              "data_protocol_sha256": "fixture-data", "reader_gate_results_sha256": "fixture-reader",
              "vocabulary_sha256": "fixture-vocabulary", "source_sha256": {
                  "/Users/caleb/TinyMem/historical.py": hashlib.sha256((relocated / "historical.py").read_bytes()).hexdigest()}}
    study_hash = save(relocated / "study.json", frozen)

    def predictions(conditions, *, learned=False):
        rows = []
        for condition in conditions:
            for world in range(128):
                for query in range(9):
                    absent = query == 8
                    answer = "unknown" if absent else "kitchen"
                    predicted = answer if learned or absent or condition.endswith(("full_history", "fingerprint")) else "office"
                    rows.append({"condition": condition, "world_id": f"world-{world}", "case_id": f"world-{world}:query-{query}",
                                 "context": "synthetic context", "question": f"where is entity-{query}?", "answer": answer,
                                 "category": "opaque_qa1_missing" if absent else "opaque_qa1_known",
                                 "prediction": predicted, "exact_match": predicted == answer})
        return rows

    def evaluation(name, protocol, rows):
        directory = relocated / name
        protocol.update(execution=execution, split="confirmation", study_protocol_sha256=study_hash,
                        data_protocol_sha256="fixture-data", adapter_sha256={"adapter": "fixture"}, split_sha256="fixture-split")
        save(directory / "protocol.json", protocol)
        save(directory / "results.json", {"predictions_sha256": save(directory / "predictions.jsonl", rows, lines=True),
                                           "states_sha256": save(directory / "states.json", [])})

    evaluation("baseline", {"reader_gate_results_sha256": "fixture-reader", "vocabulary_sha256": "fixture-vocabulary"}, predictions(baseline_conditions))
    evaluations = []
    for writer in writers:
        for seed in seeds:
            run = f"training/{writer}_seed_{seed}"
            training_hash = save(relocated / run / "protocol.json", {"study_protocol_sha256": study_hash, "writer_kind": writer, "seed": seed})
            result_hash = save(relocated / run / "results.json", {"checkpoint_sha256": "fixture-checkpoint"})
            name = f"evaluations/{writer}_seed_{seed}"
            evaluation(name, {"training_run": run, "training_protocol_sha256": training_hash,
                              "training_results_sha256": result_hash, "checkpoint_sha256": "fixture-checkpoint",
                              "writer_kind": writer, "seed": seed}, predictions(["opaque:normal"], learned=writer == "query_pool"))
            evaluations.append(name)
    command = [sys.executable, "-m", "scripts.opaque.aggregate", "--study-protocol", "study.json",
               "--baseline-run", "baseline", "--memory-runs", *evaluations, "--output", "analysis"]
    completed = subprocess.run(command, cwd=relocated, env=environment, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    report = json.loads((relocated / "analysis/report/report.json").read_text())
    assert report["assessment"]["overall_result"] == "positive"
    assert report["provenance"]["execution"] == execution
    assert "scripts/opaque/aggregate.py" in report["provenance"]["source_sha256"] or any(
        name.endswith("scripts/opaque/aggregate.py") for name in report["provenance"]["source_sha256"])
    assert all((relocated / f"analysis/report/{name}").stat().st_size > 0 for name in
               ("accuracy.png", "accuracy.svg", "known_contrasts.png", "known_contrasts.svg"))
    changed = relocated / evaluations[0] / "protocol.json"
    protocol = json.loads(changed.read_text())
    protocol["execution"]["runtime"]["device"] = "mps"
    save(changed, protocol)
    command[-1] = "mixed-output"
    result = subprocess.run(command, cwd=relocated, env=environment, capture_output=True, text=True)
    assert result.returncode != 0 and "do not mix" in result.stderr
    assert not (relocated / "mixed-output").exists()
    protocol["execution"] = execution
    save(changed, protocol)
    (relocated / evaluations[0] / "results.json").unlink()
    command[-1] = "partial-output"
    result = subprocess.run(command, cwd=relocated, env=environment, capture_output=True, text=True)
    assert result.returncode != 0
    assert not (relocated / "partial-output").exists()
