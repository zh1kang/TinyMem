from pathlib import Path

import pytest

from tinymem.data.vocabulary import ControlledVocabulary
from tinymem.evaluation.continuous_checkpoint import (
    GATED_MULTISLOT_ARCHITECTURE,
    load_continuous_checkpoint,
)
from tinymem.memory.continuous import MultiSlotAttentionMemoryCompressor
from tinymem.memory.recurrent_memory import GatedRecurrentMemoryBank
from tinymem.model.config import (
    ExperimentConfig,
    MemoryConfig,
    ModelConfig,
    StreamConfig,
)
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.training.checkpointing import save_checkpoint


def write_checkpoint(path: Path, *, architecture: str) -> ExperimentConfig:
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
        GatedRecurrentMemoryBank(capacity=4, model_width=8),
        segment_length=2,
    )
    save_checkpoint(
        path,
        model=decoder,
        step=17,
        config=config,
        extra={
            "architecture": architecture,
            "vocabulary": list(vocabulary.id_to_token),
        },
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


def test_load_continuous_checkpoint_rejects_unknown_architecture(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.pt"
    write_checkpoint(path, architecture="unknown")

    with pytest.raises(ValueError, match="unsupported.*architecture"):
        load_continuous_checkpoint(path, device="cpu")
