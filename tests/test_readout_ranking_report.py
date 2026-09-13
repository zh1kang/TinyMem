import json

import pytest

from scripts.diagnose_readout_ranking import CANDIDATES, ranking_metrics, summarize, write_json
from scripts.report_readout_ranking import build_report
from tinymem.research.update_protocol import file_sha256


def fixture_runs(tmp_path, monkeypatch):
    from scripts import report_readout_ranking
    source = tmp_path / "source"
    dirs = []
    monkeypatch.setattr(report_readout_ranking, "verify_run", lambda path: {"synthetic": True})
    monkeypatch.setattr(report_readout_ranking, "HISTORY_COUNT", 2)
    for arm in ("affine", "gelu"):
        for seed in (1337, 2027, 4099):
            original = source / f"{arm}_seed_{seed}"
            original.mkdir(parents=True)
            write_json(original / "complete.json", {"synthetic": True})
            directory = tmp_path / f"{arm}_{seed}"
            directory.mkdir()
            protocol = dict.fromkeys(("runtime", "reader_parameters_sha256", "original_sources", "script_sha256",
                "selection_salt", "state_encoding_policy", "donors", "candidate_ids", "gold_ce_tolerance", "score", "ties", "checkpoint_policy", "scope"), "synthetic")
            protocol.update(arm=arm, seed=seed, source_seal={"synthetic": True},
                source_complete_sha256=file_sha256(original / "complete.json"),
                selected_training_histories=["h0", "h1"], conditions=["normal", "shuffled", "zero"])
            protocol["gold_ce_tolerance"] = 3e-6
            records = []
            for history in ("h0", "h1"):
                for condition in ("normal", "shuffled", "zero"):
                    for i in range(10):
                        gold = "office" if i < 8 else "unknown"
                        winner = gold if condition == "normal" and history == "h0" else "bathroom"
                        scores = {name: -1.0 if name == winner else -3.0 for name in CANDIDATES}
                        records.append({"history_id": history, "case_id": f"{history}-{i}", "condition": condition,
                            "category": "update_known" if i < 8 else "update_missing", "answer": gold,
                            "scores": scores, **ranking_metrics(scores, gold), "saved_correct": False,
                            "gold_ce_replay_error": 0.0})
            write_json(directory / "protocol.json", protocol)
            (directory / "scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
            write_json(directory / "summary.json", summarize(records))
            write_json(directory / "complete.json", {"kind": "readout_ranking_complete_v1", "records": 60,
                "max_gold_ce_replay_error": 0.0,
                "files": {name: file_sha256(directory / name) for name in ("protocol.json", "scores.jsonl", "summary.json")}})
            dirs.append(directory)
    return source, dirs


def test_whole_history_pairing_keeps_seed_repeats_together(tmp_path, monkeypatch):
    source, dirs = fixture_runs(tmp_path, monkeypatch)
    report = build_report(source, dirs)
    metric = report["contrasts"]["affine"]["shuffled"]["update_known"]["top1_correct"]
    assert metric["mean_difference"] == 0.5
    assert metric["seed_sd"] == 0
    # Two histories with differences [1, 0]; a paired two-draw bootstrap spans [0, 1].
    assert metric["interval95"] == [0.0, 1.0]
    with pytest.raises(ValueError, match="all six"):
        build_report(source, dirs[:-1])
    with pytest.raises(ValueError, match="duplicate arm"):
        build_report(source, dirs + dirs[:1])
    (dirs[0] / "scores.jsonl").write_text("tampered\n")
    with pytest.raises(ValueError, match="artifact changed"):
        build_report(source, dirs)
