from dataclasses import asdict

import pytest
import torch

from test_readout_runner import tiny_reader
from test_memory_updates import episode
from tinymem.research.readout_runner import encode_before
from scripts.export_readout_content import targets_from_history
from scripts.readout_content_validation import replay_targets, validate_exports
from scripts.readout_content_probe import fit_heads, predict, metrics, positive_control, room_counts


def test_fixed_decoder_recovers_separate_synthetic_holdout():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = positive_control()
    finally:
        torch.set_num_threads(old_threads)
    assert result["passed"]
    assert result["metrics"]["all_eight_correct"] == 1
    assert result["model"]["max_gradient"] <= 1e-5


def test_targets_follow_visible_first_appearance_not_question_order(tiny_reader):
    row = asdict(encode_before(tiny_reader, episode(7)))
    first = targets_from_history(tiny_reader.tokenizer, row)
    assert first == replay_targets(tiny_reader.tokenizer, row)
    row["queries"] = list(reversed(row["queries"]))
    assert targets_from_history(tiny_reader.tokenizer, row) == first
    assert len(first) == 8 and all(0 <= v < 6 for v in first)
    row["queries"][0]["answer"] = "bathroom" if row["queries"][0]["answer"] != "bathroom" else "office"
    with pytest.raises(ValueError, match="replay disagrees"):
        targets_from_history(tiny_reader.tokenizer, row)


def test_count_control_drops_position_information():
    first = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 1]])
    permuted = first.flip(1)
    assert torch.equal(room_counts(first), room_counts(permuted))
    assert torch.equal(room_counts(first)[:, :6], torch.tensor([[2, 2, 1, 1, 1, 1]]) / 8)
    assert torch.count_nonzero(room_counts(first)[:, 6:]) == 0


def test_constant_features_remain_finite_and_prediction_does_not_refit():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        x = torch.ones(24, 16)
        y = (torch.arange(24).reshape(-1, 1) % 6).expand(-1, 8)
        model = fit_heads(x, y)
        expected_mean, expected_scale = model["mean"][:], model["scale"][:]
        result = predict(model, torch.full((3, 16), 100.0))
    finally:
        torch.set_num_threads(old_threads)
    assert torch.isfinite(result).all()
    assert model["mean"] == expected_mean and model["scale"] == expected_scale
    assert metrics(result, torch.zeros(3, 8, dtype=torch.long))["accuracy"] in (0.0, 1.0)


def test_validation_requires_all_six_before_accessing_data(tmp_path):
    import json
    declaration = tmp_path / "declaration.json"
    declaration.write_text(json.dumps({"gold_ce_tolerance": 3e-6}))
    with pytest.raises(ValueError, match="all six"):
        validate_exports([], tmp_path, declaration, tmp_path)


def test_export_validation_rejects_resealed_wrong_labels_and_replay(tiny_reader, tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from safetensors.torch import save_file
    from scripts import readout_content_validation as validation
    from tinymem.research.update_protocol import file_sha256

    monkeypatch.setattr(validation, "SPLIT_COUNTS", {"train": 2, "development": 2})
    monkeypatch.setattr(validation, "verify_run", lambda _: {"fixture": True})
    monkeypatch.setattr(validation.AutoTokenizer, "from_pretrained", lambda *a, **k: tiny_reader.tokenizer)
    source_root, token_dir = tmp_path / "sources", tmp_path / "tokenizer"
    source_root.mkdir()
    token_dir.mkdir()
    token_files = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json", "config.json"):
        (token_dir / name).write_text("fixture")
        token_files[name] = {"sha256": file_sha256(token_dir / name)}
    declaration = {"gold_ce_tolerance": 3e-6, "state_encoding_policy": "fixture",
        "source_complete_sha256": {}, "source_sha256": {name: file_sha256(Path(validation.__file__).with_name(name))
            for name in ("readout_content_probe.py", "readout_content_validation.py", "export_readout_content.py")}}
    encoded = [asdict(encode_before(tiny_reader, episode(i))) for i in range(4)]
    splits = {"train": encoded[:2], "development": encoded[2:]}
    metadata_rows = {split: [{"history_id": r["history_id"], "source_group_ids": r["source_group_ids"],
                             "targets": replay_targets(tiny_reader.tokenizer, r)} for r in rows]
                     for split, rows in splits.items()}
    runtime = {"device": "cpu", "torch_version": "fixture", "cuda_version": None, "device_name": "cpu",
               "reader_dtype": "torch.float32", "deterministic_algorithms": True}
    directories = []
    for arm in ("affine", "gelu"):
        for seed in (1337, 2027, 4099):
            name = f"{arm}_seed_{seed}"
            source = source_root / name
            source.mkdir()
            (source / "complete.json").write_text(json.dumps({"fixture": True}))
            declaration["source_complete_sha256"][name] = file_sha256(source / "complete.json")
            protocol = {"arm": arm, "seed": seed, **runtime, "reader_parameters_sha256": "fixture",
                        "source_sha256": {}, "input_identity": {"reader": {"snapshot": {"files": token_files}}}}
            (source / "protocol.json").write_text(json.dumps(protocol))
            (source / "encodings.json").write_text(json.dumps(splits))
            predictions = [{"split": split, "history_id": row["history_id"], "case_id": q["case_id"],
                            "answer": q["answer"], "phase": "final", "condition": "normal", "answer_ce": 0.5}
                           for split, rows in splits.items() for row in rows for q in row["queries"]]
            (source / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in predictions))
            directory = tmp_path / name
            directory.mkdir()
            save_file({split: torch.zeros(2, 16) for split in splits}, str(directory / "states.safetensors"))
            replay = [{k: r[k] for k in ("split", "history_id", "case_id", "answer", "answer_ce")}
                      | {"saved_answer_ce": 0.5, "error": 0.0} for r in predictions]
            (directory / "replay.jsonl").write_text("".join(json.dumps(r) + "\n" for r in replay))
            metadata = {"arm": arm, "seed": seed, "runtime": runtime, "reader_parameters_sha256": "fixture",
                "state_encoding_policy": "fixture", "source_complete_sha256": declaration["source_complete_sha256"][name],
                "source_seal": {"fixture": True}, "export_script_sha256": declaration["source_sha256"]["export_readout_content.py"],
                "rows": metadata_rows}
            (directory / "metadata.json").write_text(json.dumps(metadata))
            directories.append(directory)
    declaration_path = tmp_path / "declaration.json"
    declaration_path.write_text(json.dumps(declaration))

    def reseal(directory):
        seal = {"kind": "readout_content_export_v1", "records": 40, "max_replay_error": 0.0,
                "files": {name: file_sha256(directory / name) for name in ("states.safetensors", "metadata.json", "replay.jsonl")}}
        (directory / "complete.json").write_text(json.dumps(seal))

    for directory in directories:
        metadata = json.loads((directory / "metadata.json").read_text())
        metadata["declaration_sha256"] = file_sha256(declaration_path)
        (directory / "metadata.json").write_text(json.dumps(metadata))
        reseal(directory)
    assert len(validate_exports(directories, source_root, declaration_path, token_dir)) == 6
    target = directories[0]
    original_metadata = (target / "metadata.json").read_text()
    metadata = json.loads(original_metadata)
    metadata["rows"]["train"][0]["targets"][0] = (metadata["rows"]["train"][0]["targets"][0] + 1) % 6
    (target / "metadata.json").write_text(json.dumps(metadata))
    reseal(target)
    with pytest.raises(ValueError, match="labels differ"):
        validate_exports(directories, source_root, declaration_path, token_dir)
    (target / "metadata.json").write_text(original_metadata)
    (target / "replay.jsonl").write_text("")
    reseal(target)
    with pytest.raises(ValueError, match="replay coverage"):
        validate_exports(directories, source_root, declaration_path, token_dir)
