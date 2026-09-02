"""Load trained continuous-memory decoders for frozen evaluation."""

from dataclasses import dataclass
from pathlib import Path

import torch

from tinymem.data.vocabulary import SPECIAL_TOKENS, ControlledVocabulary
from tinymem.memory.continuous import MultiSlotAttentionMemoryCompressor
from tinymem.memory.recurrent_memory import GatedRecurrentMemoryBank
from tinymem.memory.write_gate import TokenSegmentWriteGate
from tinymem.model.config import ExperimentConfig
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.training.checkpointing import CHECKPOINT_FORMAT_VERSION


GATED_MULTISLOT_ARCHITECTURE = (
    "segmented_continuous_multislot_attention_pool_gated_update"
)
TOKEN_GATED_MULTISLOT_ARCHITECTURE = (
    "segmented_continuous_multislot_attention_pool_token_gated_update"
)


@dataclass(frozen=True)
class LoadedContinuousCheckpoint:
    """Hold a reconstructed decoder and its immutable training metadata."""

    decoder: SegmentedContinuousDecoder
    config: ExperimentConfig
    vocabulary: ControlledVocabulary
    step: int
    architecture: str


def load_continuous_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str,
) -> LoadedContinuousCheckpoint:
    """Reconstruct the supported continuous decoder without training it."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"continuous checkpoint does not exist: {checkpoint_path}"
        )
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise ValueError("continuous checkpoint must contain a dictionary")
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format version")

    config_values = payload.get("config")
    if not isinstance(config_values, dict):
        raise ValueError("continuous checkpoint must contain an experiment config")
    config = ExperimentConfig.from_dict(config_values)

    extra = payload.get("extra")
    if not isinstance(extra, dict):
        raise ValueError("continuous checkpoint must contain extra metadata")
    architecture = extra.get("architecture")
    if architecture not in (
        GATED_MULTISLOT_ARCHITECTURE,
        TOKEN_GATED_MULTISLOT_ARCHITECTURE,
    ):
        raise ValueError(
            "unsupported continuous checkpoint architecture: "
            f"{architecture!r}"
        )
    tokens = extra.get("vocabulary")
    if not isinstance(tokens, list) or not all(
        isinstance(token, str) for token in tokens
    ):
        raise ValueError("continuous checkpoint must contain its vocabulary")
    if tuple(tokens[: len(SPECIAL_TOKENS)]) != SPECIAL_TOKENS:
        raise ValueError("continuous checkpoint has invalid special tokens")
    vocabulary = ControlledVocabulary(tokens[len(SPECIAL_TOKENS) :])
    if vocabulary.id_to_token != tuple(tokens):
        raise ValueError("continuous checkpoint vocabulary is not in standard order")
    write_threshold = extra.get("write_threshold", 0.5)
    if isinstance(write_threshold, bool) or not isinstance(
        write_threshold,
        (int, float),
    ):
        raise ValueError("continuous checkpoint has an invalid write threshold")
    if not 0 < write_threshold <= 1:
        raise ValueError("continuous checkpoint has an invalid write threshold")

    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("continuous checkpoint must contain model state")
    queries = state.get("compressor.queries")
    if not isinstance(queries, torch.Tensor) or queries.ndim != 2:
        raise ValueError("continuous checkpoint has invalid compressor queries")
    if queries.shape[1] != config.model.d_model:
        raise ValueError("compressor query width does not match the model")

    decoder = SegmentedContinuousDecoder(
        DecoderOnlyTransformer(config.model),
        MultiSlotAttentionMemoryCompressor(
            config.model.d_model,
            summary_slots=queries.shape[0],
        ),
        GatedRecurrentMemoryBank(
            capacity=config.memory.n_slots,
            model_width=config.model.d_model,
            write_threshold=float(write_threshold),
        ),
        segment_length=config.stream.segment_length,
        write_gate=(
            TokenSegmentWriteGate(config.model.d_model)
            if architecture == TOKEN_GATED_MULTISLOT_ARCHITECTURE
            else None
        ),
    )
    decoder.load_state_dict(state)
    decoder.to(device)
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("continuous checkpoint has an invalid training step")
    return LoadedContinuousCheckpoint(
        decoder=decoder,
        config=config,
        vocabulary=vocabulary,
        step=step,
        architecture=architecture,
    )
