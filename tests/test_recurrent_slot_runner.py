import json
import sys

import pytest
import torch

from scripts.run_recurrent_slot_qa import main
from tinymem.evaluation.wikitext_checkpoint import ByteMemorySpec, build_byte_memory_decoder
from tinymem.model.config import ExperimentConfig, ModelConfig, MemoryConfig, StreamConfig
from tinymem.model.recurrent_slot_decoder import RecurrentSlotDecoder
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.training.checkpointing import load_checkpoint, save_checkpoint


def parent_checkpoint(tmp_path):
    config = ExperimentConfig(
        model=ModelConfig(vocab_size=260, d_model=8, n_layers=1, n_heads=2, d_ff=16, max_local_tokens=128, dropout=0.0),
        stream=StreamConfig(segment_length=128, local_window=128),
        memory=MemoryConfig(n_slots=8, code_dim=8),
    )
    model = build_byte_memory_decoder(config, ByteMemorySpec(), segment_length=128)
    path = tmp_path / "parent.pt"
    save_checkpoint(path, model=model, config=config, extra={"architecture": "segmented_continuous_wikitext_byte_lm", "selected_window": 128, "tokenizer": "utf8_bytes_v1"})
    return path


def test_runner_saves_consistent_config_and_reloadable_checkpoint(tmp_path, monkeypatch):
    parent = parent_checkpoint(tmp_path)
    output = tmp_path / "runs"
    monkeypatch.setattr(sys, "argv", ["run", "--checkpoint", str(parent), "--steps", "1", "--batch-size", "2", "--train-examples", "4", "--validation-examples", "4", "--memory-width", "4", "--slots", "1", "--artifact-root", str(output)])
    main()
    result_path, = output.glob("*/results.json")
    result = json.loads(result_path.read_text())
    payload = torch.load(result_path.parent / "checkpoint.pt", weights_only=True)
    assert payload["config"]["stream"]["segment_length"] == 64
    assert payload["config"]["memory"]["n_slots"] == 1
    assert payload["config"]["memory"]["code_dim"] == payload["config"]["model"]["d_model"] == 8
    assert result["protocol"]["persistent_state_shape"] == [1, 4]
    assert result["protocol"]["reader_interface_width"] == 8
    reader = DecoderOnlyTransformer(ModelConfig(**payload["config"]["model"]))
    model = RecurrentSlotDecoder(reader, memory_width=4, slots=1, segment_length=64)
    load_checkpoint(result_path.parent / "checkpoint.pt", model=model)
    assert result["protocol"]["persistent_history_state_bytes_per_stream"] == 17
    assert set(result["evaluation"]) == {"normal", "drop", "zero", "shuffle", "without_correction"}
    assert (result_path.parent / "data_manifest.json").stat().st_mtime <= (result_path.parent / "checkpoint.pt").stat().st_mtime


def test_generation_capacity_is_rejected_before_any_run_or_training(tmp_path, monkeypatch):
    parent = parent_checkpoint(tmp_path)
    output = tmp_path / "runs"
    monkeypatch.setattr(sys, "argv", ["run", "--checkpoint", str(parent), "--steps", "1", "--train-examples", "4", "--validation-examples", "4", "--max-new-tokens", "100", "--artifact-root", str(output)])
    with pytest.raises(ValueError, match="generation"):
        main()
    assert not output.exists()
