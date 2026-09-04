import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def load_report_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts/aggregate_conversational_qa.py"
    spec = importlib.util.spec_from_file_location("aggregate_conversational_qa", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def fine_tune_document(checkpoint: str, *, seed: int = 1) -> dict[str, object]:
    return {
        "status": "development_single_seed_conversational_qa",
        "memory": {"compressor": "mean", "memory_update": "fifo"},
        "training_delay": {"max_bytes": 0},
        "seed": seed,
        "checkpoint": checkpoint,
        "controlled_after": {"overall": {"exact_accuracy": 0.5}},
        "training_history": {"answer_losses": [1.0, 0.5]},
    }


def test_discover_finetunes_rejects_duplicate_config_seed(tmp_path: Path) -> None:
    report = load_report_module()
    write_json(tmp_path / "a/results.json", fine_tune_document("a.pt"))
    write_json(tmp_path / "b/results.json", fine_tune_document("b.pt"))

    with pytest.raises(ValueError, match="duplicate fine-tune config and seed"):
        report.discover_finetunes((tmp_path,))


def test_discover_holdouts_rejects_duplicate_checkpoint_condition(
    tmp_path: Path,
) -> None:
    report = load_report_module()
    fine_tune = report.FineTuneRun(
        config="mean/fifo",
        seed=1,
        checkpoint="model.pt",
        results_path=tmp_path / "fine-tune.json",
        validation_exact=0.5,
        answer_losses=(1.0,),
    )
    evaluation = {
        "overall": {"exact_accuracy": 0.5},
        "by_task": {},
        "predictions": [],
    }
    document = {
        "checkpoint": "model.pt",
        "memory_condition": "normal",
        "selected_window": 512,
        "evaluation": evaluation,
    }
    write_json(tmp_path / "a/results.json", document)
    write_json(tmp_path / "b/results.json", document)

    with pytest.raises(ValueError, match="duplicate holdout checkpoint"):
        report.discover_holdouts(tmp_path, {fine_tune.checkpoint: fine_tune})
