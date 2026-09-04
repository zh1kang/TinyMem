"""Reconstruct frozen segmented decoders from WikiText checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import torch

from tinymem.memory.continuous import (
    AttentionPoolMemoryCompressor,
    MeanPoolMemoryCompressor,
    MultiSlotAttentionMemoryCompressor,
)
from tinymem.memory.recurrent_memory import (
    GatedRecurrentMemoryBank,
    RecurrentMemoryBank,
)
from tinymem.model.config import ExperimentConfig
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.tokenization.byte_tokenizer import ByteTokenizer
from tinymem.training.checkpointing import load_checkpoint


SUPPORTED_BYTE_MEMORY_ARCHITECTURES = frozenset(
    {
        "segmented_continuous_wikitext_byte_lm",
        "segmented_continuous_conversational_qa",
    }
)
SUPPORTED_COMPRESSORS = ("mean", "attention", "multislot_attention")
SUPPORTED_MEMORY_UPDATES = ("fifo", "gated")
DEFAULT_WRITE_THRESHOLD = 0.5


@dataclass(frozen=True)
class ByteMemorySpec:
    """Describe the compressor and bank family of a byte-level decoder.

    WikiText checkpoints written before this record existed used a mean-pool
    compressor with a FIFO bank, so those values are the defaults.
    """

    compressor: str = "mean"
    summaries_per_segment: int = 1
    memory_update: str = "fifo"
    write_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.compressor not in SUPPORTED_COMPRESSORS:
            raise ValueError(
                f"compressor must be one of {SUPPORTED_COMPRESSORS}, "
                f"got {self.compressor!r}"
            )
        if isinstance(self.summaries_per_segment, bool) or not isinstance(
            self.summaries_per_segment,
            Integral,
        ):
            raise TypeError("summaries_per_segment must be an integer")
        if self.summaries_per_segment <= 0:
            raise ValueError("summaries_per_segment must be positive")
        if (
            self.compressor != "multislot_attention"
            and self.summaries_per_segment != 1
        ):
            raise ValueError(
                "single-slot compressors write exactly one summary per segment"
            )
        if self.memory_update not in SUPPORTED_MEMORY_UPDATES:
            raise ValueError(
                f"memory_update must be one of {SUPPORTED_MEMORY_UPDATES}, "
                f"got {self.memory_update!r}"
            )
        if self.memory_update == "fifo":
            if self.write_threshold is not None:
                raise ValueError("fifo memory updates must not have a write threshold")
        else:
            if isinstance(self.write_threshold, bool) or not isinstance(
                self.write_threshold,
                Real,
            ):
                raise TypeError("gated memory updates require a real write threshold")
            if not 0 < self.write_threshold <= 1:
                raise ValueError("write_threshold must be in (0, 1]")

    def to_metadata(self) -> dict[str, object]:
        """Return the checkpoint ``extra`` fields that identify this spec."""
        return {
            "compressor": self.compressor,
            "summaries_per_segment": int(self.summaries_per_segment),
            "memory_update": self.memory_update,
            "write_threshold": (
                float(self.write_threshold)
                if self.write_threshold is not None
                else None
            ),
        }

    @classmethod
    def from_metadata(cls, extra: dict[str, object]) -> "ByteMemorySpec":
        """Read the spec from checkpoint metadata, defaulting to mean-pool FIFO."""
        if not isinstance(extra, dict):
            raise TypeError("extra must be a dictionary")
        compressor = extra.get("compressor", "mean")
        if not isinstance(compressor, str):
            raise ValueError("checkpoint compressor must be a string")
        memory_update = extra.get("memory_update", "fifo")
        if not isinstance(memory_update, str):
            raise ValueError("checkpoint memory_update must be a string")
        write_threshold = extra.get("write_threshold")
        if memory_update == "gated" and write_threshold is None:
            raise ValueError("gated checkpoint is missing its write threshold")
        return cls(
            compressor=compressor,
            summaries_per_segment=extra.get("summaries_per_segment", 1),
            memory_update=memory_update,
            write_threshold=write_threshold,
        )


def build_byte_memory_decoder(
    config: ExperimentConfig,
    spec: ByteMemorySpec,
    *,
    segment_length: int,
) -> SegmentedContinuousDecoder:
    """Build the byte-level segmented decoder described by a config and spec."""
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    if not isinstance(spec, ByteMemorySpec):
        raise TypeError("spec must be a ByteMemorySpec")
    if spec.summaries_per_segment > config.memory.n_slots:
        raise ValueError("summaries_per_segment must not exceed memory n_slots")

    width = config.model.d_model
    if spec.compressor == "mean":
        compressor = MeanPoolMemoryCompressor(width)
    elif spec.compressor == "attention":
        compressor = AttentionPoolMemoryCompressor(width)
    else:
        compressor = MultiSlotAttentionMemoryCompressor(
            width,
            summary_slots=spec.summaries_per_segment,
        )
    if spec.memory_update == "fifo":
        bank = RecurrentMemoryBank(
            capacity=config.memory.n_slots,
            model_width=width,
        )
    else:
        assert spec.write_threshold is not None
        bank = GatedRecurrentMemoryBank(
            capacity=config.memory.n_slots,
            model_width=width,
            write_threshold=float(spec.write_threshold),
        )
    return SegmentedContinuousDecoder(
        DecoderOnlyTransformer(config.model),
        compressor,
        bank,
        segment_length=segment_length,
    )


@dataclass(frozen=True)
class LoadedWikiTextCheckpoint:
    """Hold a reconstructed frozen decoder and its experiment config."""

    decoder: SegmentedContinuousDecoder
    config: ExperimentConfig
    selected_window: int
    architecture: str
    memory_spec: ByteMemorySpec


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
    architecture = extra.get("architecture")
    if architecture not in SUPPORTED_BYTE_MEMORY_ARCHITECTURES:
        raise ValueError("checkpoint is not a supported segmented byte model")
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
    memory_spec = ByteMemorySpec.from_metadata(extra)
    if (
        "summaries_per_segment" in extra
        and memory_spec.summaries_per_segment != config.memory.codes_per_write
    ):
        raise ValueError(
            "checkpoint summaries_per_segment does not match its memory config"
        )

    decoder = build_byte_memory_decoder(
        config,
        memory_spec,
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
        architecture=architecture,
        memory_spec=memory_spec,
    )
