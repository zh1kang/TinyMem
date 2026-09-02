from pathlib import Path

import torch

from tinymem.data.vocabulary import ControlledVocabulary
from tinymem.evaluation.base_checkpoint import load_base_checkpoint
from tinymem.model.config import (
    ExperimentConfig,
    MemoryConfig,
    ModelConfig,
    MTPConfig,
    StreamConfig,
)
from tinymem.model.multi_token_prediction import MultiTokenPredictionHeads
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.training.checkpointing import save_checkpoint


def test_load_base_checkpoint_restores_trained_mtp_heads(tmp_path: Path) -> None:
    vocabulary = ControlledVocabulary([" ", "Mary", "kitchen"])
    config = ExperimentConfig(
        model=ModelConfig(
            vocab_size=len(vocabulary),
            d_model=8,
            n_layers=1,
            n_heads=2,
            d_ff=16,
            max_local_tokens=16,
        ),
        stream=StreamConfig(segment_length=8, local_window=16),
        memory=MemoryConfig(code_dim=8),
        mtp=MTPConfig(enabled=True, horizons=(2, 3, 4), loss_weight=0.2),
    )
    model = DecoderOnlyTransformer(config.model)
    heads = MultiTokenPredictionHeads(8, len(vocabulary), (2, 3, 4))
    with torch.no_grad():
        for index, head in enumerate(heads.heads, start=1):
            head.weight.fill_(float(index))
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model=model,
        step=21,
        config=config,
        extra={
            "vocabulary": list(vocabulary.id_to_token),
            "mtp_horizons": [2, 3, 4],
            "mtp_head_state": heads.state_dict(),
        },
    )

    loaded = load_base_checkpoint(path, device="cpu")

    assert loaded.mtp_heads is not None
    assert loaded.step == 21
    for expected, actual in zip(
        heads.parameters(),
        loaded.mtp_heads.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)
