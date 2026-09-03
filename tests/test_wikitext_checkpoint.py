from pathlib import Path

import pytest
import torch

from tinymem.evaluation.wikitext_checkpoint import load_wikitext_checkpoint
from tinymem.memory.continuous import MeanPoolMemoryCompressor
from tinymem.memory.recurrent_memory import RecurrentMemoryBank
from tinymem.model.config import (
    ExperimentConfig,
    MemoryConfig,
    ModelConfig,
    StreamConfig,
)
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.training.checkpointing import save_checkpoint


def make_checkpoint(path: Path, *, architecture: str) -> SegmentedContinuousDecoder:
    config = ExperimentConfig(
        model=ModelConfig(
            vocab_size=260,
            d_model=8,
            n_layers=1,
            n_heads=2,
            d_ff=16,
            max_local_tokens=8,
            dropout=0.0,
        ),
        stream=StreamConfig(segment_length=4, local_window=8),
        memory=MemoryConfig(n_slots=2, code_dim=8, codes_per_write=2),
    )
    decoder = SegmentedContinuousDecoder(
        DecoderOnlyTransformer(config.model),
        MeanPoolMemoryCompressor(8),
        RecurrentMemoryBank(capacity=2, model_width=8),
        segment_length=4,
    )
    save_checkpoint(
        path,
        model=decoder,
        config=config,
        extra={
            "architecture": architecture,
            "selected_window": 4,
            "tokenizer": "utf8_bytes_v1",
        },
    )
    return decoder


def test_load_wikitext_checkpoint_restores_decoder(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    expected = make_checkpoint(
        path,
        architecture="segmented_continuous_wikitext_byte_lm",
    )

    loaded = load_wikitext_checkpoint(path, device="cpu")

    assert loaded.selected_window == 4
    assert loaded.config.model.vocab_size == 260
    for name, parameter in expected.state_dict().items():
        torch.testing.assert_close(loaded.decoder.state_dict()[name], parameter)


def test_load_wikitext_checkpoint_rejects_other_architectures(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.pt"
    make_checkpoint(path, architecture="other")

    with pytest.raises(ValueError, match="not a WikiText"):
        load_wikitext_checkpoint(path, device="cpu")


def test_load_wikitext_checkpoint_rejects_segment_metadata_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.pt"
    make_checkpoint(
        path,
        architecture="segmented_continuous_wikitext_byte_lm",
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["extra"]["selected_window"] = 2
    torch.save(payload, path)

    with pytest.raises(ValueError, match="does not match"):
        load_wikitext_checkpoint(path, device="cpu")
