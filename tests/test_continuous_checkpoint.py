from pathlib import Path

import pytest

from tinymem.data.vocabulary import ControlledVocabulary
from tinymem.evaluation.continuous_checkpoint import (
    GATED_MULTISLOT_ARCHITECTURE,
    TOKEN_GATED_MULTISLOT_ARCHITECTURE,
    load_continuous_checkpoint,
)
from tinymem.memory.continuous import MultiSlotAttentionMemoryCompressor
from tinymem.memory.recurrent_memory import GatedRecurrentMemoryBank
from tinymem.memory.write_gate import TokenSegmentWriteGate
from tinymem.model.config import (
    ExperimentConfig,
    MemoryConfig,
    ModelConfig,
    StreamConfig,
)
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.training.checkpointing import save_checkpoint


def write_checkpoint(
    path: Path,
    *,
    architecture: str,
    write_threshold: float | None = None,
    memory_position_mode: str = "absolute",
    write_gate_kernel_size: int = 3,
) -> ExperimentConfig:
    vocabulary = ControlledVocabulary([" ", "Mary", "kitchen"])
    config = ExperimentConfig(
        model=ModelConfig(
            vocab_size=len(vocabulary),
            d_model=8,
            n_layers=1,
            n_heads=2,
            d_ff=16,
            max_local_tokens=4,
        ),
        stream=StreamConfig(segment_length=2, local_window=4),
        memory=MemoryConfig(
            n_slots=4,
            codebook_size=8,
            code_dim=8,
            codes_per_write=2,
        ),
    )
    decoder = SegmentedContinuousDecoder(
        DecoderOnlyTransformer(config.model),
        MultiSlotAttentionMemoryCompressor(8, summary_slots=2),
        GatedRecurrentMemoryBank(
            capacity=4,
            model_width=8,
            write_threshold=(
                0.5 if write_threshold is None else write_threshold
            ),
        ),
        segment_length=2,
        write_gate=(
            TokenSegmentWriteGate(8, kernel_size=write_gate_kernel_size)
            if architecture == TOKEN_GATED_MULTISLOT_ARCHITECTURE
            else None
        ),
        memory_position_mode=memory_position_mode,
    )
    extra: dict[str, object] = {
        "architecture": architecture,
        "vocabulary": list(vocabulary.id_to_token),
    }
    if write_threshold is not None:
        extra["write_threshold"] = write_threshold
    if memory_position_mode != "absolute":
        extra["memory_position_mode"] = memory_position_mode
    save_checkpoint(
        path,
        model=decoder,
        step=17,
        config=config,
        extra=extra,
    )
    return config


def test_load_continuous_checkpoint_reconstructs_trained_architecture(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.pt"
    config = write_checkpoint(
        path,
        architecture=GATED_MULTISLOT_ARCHITECTURE,
    )

    loaded = load_continuous_checkpoint(path, device="cpu")

    assert loaded.config == config
    assert loaded.step == 17
    assert loaded.decoder.segment_length == 2
    assert loaded.decoder.compressor.summary_slots == 2
    assert loaded.decoder.bank.capacity == 4
    assert loaded.decoder.bank.write_threshold == 0.5
    assert loaded.decoder.write_gate is None
    assert loaded.decoder.memory_position_mode == "absolute"


def test_load_continuous_checkpoint_reconstructs_token_gate_and_threshold(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.pt"
    write_checkpoint(
        path,
        architecture=TOKEN_GATED_MULTISLOT_ARCHITECTURE,
        write_threshold=0.81,
        memory_position_mode="virtual",
        write_gate_kernel_size=11,
    )

    loaded = load_continuous_checkpoint(path, device="cpu")

    assert isinstance(loaded.decoder.write_gate, TokenSegmentWriteGate)
    assert loaded.decoder.bank.write_threshold == 0.81
    assert loaded.decoder.memory_position_mode == "virtual"
    assert loaded.decoder.write_gate.kernel_size == 11


def test_load_continuous_checkpoint_rejects_unknown_architecture(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.pt"
    write_checkpoint(path, architecture="unknown")

    with pytest.raises(ValueError, match="unsupported.*architecture"):
        load_continuous_checkpoint(path, device="cpu")
