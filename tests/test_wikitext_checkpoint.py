from pathlib import Path

import pytest
import torch

from tinymem.evaluation.wikitext_checkpoint import (
    ByteMemorySpec,
    build_byte_memory_decoder,
    load_wikitext_checkpoint,
)
from tinymem.memory.continuous import (
    MeanPoolMemoryCompressor,
    MultiSlotAttentionMemoryCompressor,
)
from tinymem.memory.recurrent_memory import (
    GatedRecurrentMemoryBank,
    RecurrentMemoryBank,
)
from tinymem.model.config import (
    ExperimentConfig,
    MemoryConfig,
    ModelConfig,
    StreamConfig,
)
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.training.checkpointing import save_checkpoint


def make_config(*, codes_per_write: int = 2) -> ExperimentConfig:
    return ExperimentConfig(
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
        memory=MemoryConfig(n_slots=2, code_dim=8, codes_per_write=codes_per_write),
    )


def make_checkpoint(path: Path, *, architecture: str) -> SegmentedContinuousDecoder:
    """Write a legacy checkpoint with no memory metadata, as early runs did."""
    config = make_config()
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


def make_spec_checkpoint(
    path: Path,
    spec: ByteMemorySpec,
    *,
    extra_overrides: dict[str, object] | None = None,
) -> SegmentedContinuousDecoder:
    config = make_config(codes_per_write=spec.summaries_per_segment)
    decoder = build_byte_memory_decoder(config, spec, segment_length=4)
    extra = {
        "architecture": "segmented_continuous_wikitext_byte_lm",
        "selected_window": 4,
        "tokenizer": "utf8_bytes_v1",
        **spec.to_metadata(),
        **(extra_overrides or {}),
    }
    save_checkpoint(path, model=decoder, config=config, extra=extra)
    return decoder


def test_load_wikitext_checkpoint_restores_decoder(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    expected = make_checkpoint(
        path,
        architecture="segmented_continuous_wikitext_byte_lm",
    )

    loaded = load_wikitext_checkpoint(path, device="cpu")

    assert loaded.selected_window == 4
    assert loaded.architecture == "segmented_continuous_wikitext_byte_lm"
    assert loaded.config.model.vocab_size == 260
    assert loaded.memory_spec == ByteMemorySpec()
    assert isinstance(loaded.decoder.compressor, MeanPoolMemoryCompressor)
    assert type(loaded.decoder.bank) is RecurrentMemoryBank
    for name, parameter in expected.state_dict().items():
        torch.testing.assert_close(loaded.decoder.state_dict()[name], parameter)


def test_load_wikitext_checkpoint_rejects_other_architectures(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.pt"
    make_checkpoint(path, architecture="other")

    with pytest.raises(ValueError, match="not a supported"):
        load_wikitext_checkpoint(path, device="cpu")


def test_load_wikitext_checkpoint_accepts_conversational_descendant(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.pt"
    make_checkpoint(
        path,
        architecture="segmented_continuous_conversational_qa",
    )

    loaded = load_wikitext_checkpoint(path, device="cpu")

    assert loaded.architecture == "segmented_continuous_conversational_qa"


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


def test_load_wikitext_checkpoint_restores_multislot_gated_memory(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.pt"
    spec = ByteMemorySpec(
        compressor="multislot_attention",
        summaries_per_segment=2,
        memory_update="gated",
        write_threshold=0.7,
    )
    expected = make_spec_checkpoint(path, spec)

    loaded = load_wikitext_checkpoint(path, device="cpu")

    assert loaded.memory_spec == spec
    assert isinstance(
        loaded.decoder.compressor,
        MultiSlotAttentionMemoryCompressor,
    )
    assert loaded.decoder.compressor.summary_slots == 2
    assert isinstance(loaded.decoder.bank, GatedRecurrentMemoryBank)
    assert loaded.decoder.bank.write_threshold == 0.7
    for name, parameter in expected.state_dict().items():
        torch.testing.assert_close(loaded.decoder.state_dict()[name], parameter)


def test_load_wikitext_checkpoint_rejects_unknown_compressor(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.pt"
    make_spec_checkpoint(
        path,
        ByteMemorySpec(),
        extra_overrides={"compressor": "discrete"},
    )

    with pytest.raises(ValueError, match="compressor must be one of"):
        load_wikitext_checkpoint(path, device="cpu")


def test_load_wikitext_checkpoint_rejects_gated_without_threshold(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.pt"
    make_spec_checkpoint(
        path,
        ByteMemorySpec(),
        extra_overrides={"memory_update": "gated"},
    )

    with pytest.raises(ValueError, match="missing its write threshold"):
        load_wikitext_checkpoint(path, device="cpu")


def test_load_wikitext_checkpoint_rejects_summary_count_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.pt"
    make_spec_checkpoint(
        path,
        ByteMemorySpec(
            compressor="multislot_attention",
            summaries_per_segment=2,
        ),
        extra_overrides={"summaries_per_segment": 1},
    )

    with pytest.raises(ValueError, match="does not match its memory config"):
        load_wikitext_checkpoint(path, device="cpu")


def test_byte_memory_spec_rejects_inconsistent_values() -> None:
    with pytest.raises(ValueError, match="exactly one summary"):
        ByteMemorySpec(compressor="mean", summaries_per_segment=2)
    with pytest.raises(ValueError, match="must not have a write threshold"):
        ByteMemorySpec(memory_update="fifo", write_threshold=0.5)
    with pytest.raises(TypeError, match="require a real write threshold"):
        ByteMemorySpec(memory_update="gated")
    with pytest.raises(ValueError, match="must not exceed memory n_slots"):
        build_byte_memory_decoder(
            make_config(codes_per_write=2),
            ByteMemorySpec(
                compressor="multislot_attention",
                summaries_per_segment=3,
            ),
            segment_length=4,
        )
