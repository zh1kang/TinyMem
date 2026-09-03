"""Reconstruct frozen segmented decoders from WikiText checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from tinymem.memory.continuous import MeanPoolMemoryCompressor
from tinymem.memory.recurrent_memory import RecurrentMemoryBank
from tinymem.model.config import ExperimentConfig
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.tokenization.byte_tokenizer import ByteTokenizer
from tinymem.training.checkpointing import load_checkpoint


@dataclass(frozen=True)
class LoadedWikiTextCheckpoint:
    """Hold a reconstructed frozen decoder and its experiment config."""

    decoder: SegmentedContinuousDecoder
    config: ExperimentConfig
    selected_window: int


def load_wikitext_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str,
) -> LoadedWikiTextCheckpoint:
    """Load and validate a segmented byte-language-model checkpoint."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError("checkpoint must contain an experiment configuration")
    extra = payload.get("extra")
    if not isinstance(extra, dict):
        raise ValueError("checkpoint must contain architecture metadata")
    if extra.get("architecture") != "segmented_continuous_wikitext_byte_lm":
        raise ValueError("checkpoint is not a WikiText segmented language model")
    if extra.get("tokenizer") != "utf8_bytes_v1":
        raise ValueError("checkpoint does not use the expected byte tokenizer")
    selected_window = extra.get("selected_window")
    if isinstance(selected_window, bool) or not isinstance(selected_window, int):
        raise ValueError("checkpoint is missing its selected local window")

    config = ExperimentConfig.from_dict(payload["config"])
    if config.model.vocab_size != ByteTokenizer.vocab_size:
        raise ValueError("checkpoint vocabulary does not match the byte tokenizer")
    if not 0 < selected_window <= config.model.max_local_tokens:
        raise ValueError("checkpoint selected window is outside the local window")
    if selected_window != config.stream.segment_length:
        raise ValueError(
            "checkpoint selected window does not match its stream configuration"
        )

    decoder = SegmentedContinuousDecoder(
        DecoderOnlyTransformer(config.model),
        MeanPoolMemoryCompressor(config.model.d_model),
        RecurrentMemoryBank(
            capacity=config.memory.n_slots,
            model_width=config.model.d_model,
        ),
        segment_length=selected_window,
    ).to(device)
    load_checkpoint(
        checkpoint_path,
        model=decoder,
        map_location=device,
    )
    return LoadedWikiTextCheckpoint(
        decoder=decoder,
        config=config,
        selected_window=selected_window,
    )
