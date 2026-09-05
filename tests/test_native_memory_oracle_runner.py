import hashlib
import json
import sys

import pytest
import torch

from scripts import fit_native_memory_oracle as runner
from tinymem.research.memory_prompt import NativeMemoryExample
from tinymem.research.native_memory_oracle import QA1_LOCATIONS, NativeMemoryOracle
from tinymem.research.pretrained import PretrainedReader


def adaptation_fixture(tmp_path):
    adaptation = tmp_path / "adaptation"
    adapter = adaptation / "step_000400"
    adapter.mkdir(parents=True)
    rows = [{"case": {"case_id": answer, "category": "babi_qa1", "history_id": answer, "context": answer,
                      "question": f"where-{index % 3}?", "answer": answer}} for index, answer in enumerate(QA1_LOCATIONS)]
    manifest = json.dumps({"train": rows, "development": "must not evaluate this split"})
    (adaptation / "data_manifest.json").write_text(manifest)
    manifest_hash = hashlib.sha256(manifest.encode()).hexdigest()
    protocol = json.dumps({"data_manifest_sha256": manifest_hash})
    (adaptation / "protocol.json").write_text(protocol)
    (adaptation / "results.json").write_text(json.dumps({"best_checkpoint": str(adapter)}))
    (adapter / "reader_adapter_protocol.json").write_text(json.dumps({
        "training_manifest_sha256": manifest_hash,
        "training_protocol_sha256": hashlib.sha256(protocol.encode()).hexdigest(),
    }))
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        (adapter / name).write_text("test-only placeholder")
    return adaptation


def test_runner_fits_only_training_saves_final_and_reports_unique_grid(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    safetensors = pytest.importorskip("safetensors.torch")
    adaptation = adaptation_fixture(tmp_path)
    output = tmp_path / "run"
    model = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(
        vocab_size=16, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=32,
    ))
    model.generation_config.eos_token_id = list(range(16))

    class Tokenizer:
        def decode(self, ids, *, skip_special_tokens):
            return "unknown"

    reader = PretrainedReader(model, Tokenizer())
    monkeypatch.setattr(runner, "verify_qwen_snapshot", lambda path: {"test": True})
    monkeypatch.setattr(runner, "load_qwen_reader", lambda *args, **kwargs: reader)
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", lambda model, *args, **kwargs: model)

    def encode(reader, case):
        index = QA1_LOCATIONS.index(case.answer)
        return NativeMemoryExample(case.case_id, (1, 3), (9, 10, 11), (4, 5 + index % 3), (index + 6, 2))

    monkeypatch.setattr(runner, "encode_memory_example", encode)
    monkeypatch.setattr(sys, "argv", ["fit", "--adaptation-run", str(adaptation), "--output", str(output), "--device", "cpu", "--steps", "2"])
    runner.main()
    protocol = json.loads((output / "protocol.json").read_text())
    result = json.loads((output / "results.json").read_text())
    predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert len(predictions) == 54 and len(metrics) == 2
    assert all(len(row["case_answer_ce"]) == 6 for row in metrics)
    assert result["unique_grid_count"] == 18 and result["grid_count"] == 36
    assert protocol["unique_query_representatives"] == [0, 1, 2]
    assert protocol["strict_success"]["minimum_unique_grid_donor_answer_matches"] == 15
    assert result["by_condition"]["drop"]["unknown"] == 6 and not result["strict_fit_success"]
    assert not result["development_or_test_or_external_evaluated"]
    assert protocol["state_tensor_bytes_per_materialized_code"] == 66
    assert (output / "protocol.json").stat().st_mtime_ns <= (output / "initial_oracle.safetensors").stat().st_mtime_ns
    checkpoint, = output.glob("step_*.safetensors")
    assert runner.sha256(checkpoint) == result["checkpoint_sha256"]
    oracle = NativeMemoryOracle(6, 16)
    oracle.load_state_dict(safetensors.load_file(checkpoint))
    assert all(parameter.grad is None for parameter in model.parameters())
    assert result["logical_training_forward_tokens"] == 2 * 6 * (2 + 2 + 2 + 2 - 1)
    with pytest.raises(FileExistsError, match="overwrite"):
        runner.main()


def test_bad_provenance_fails_before_model_loading(tmp_path, monkeypatch):
    pytest.importorskip("peft")
    adaptation = adaptation_fixture(tmp_path)
    (adaptation / "data_manifest.json").write_text("{}")
    monkeypatch.setattr(runner, "load_qwen_reader", lambda *args, **kwargs: pytest.fail("must reject before loading"))
    monkeypatch.setattr(sys, "argv", ["fit", "--adaptation-run", str(adaptation), "--output", str(tmp_path / "run"), "--device", "cpu"])
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.main()
