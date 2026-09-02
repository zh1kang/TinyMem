"""Load base decoder checkpoints with optional MTP heads."""

from dataclasses import dataclass
from pathlib import Path

import torch

from tinymem.data.vocabulary import SPECIAL_TOKENS, ControlledVocabulary
from tinymem.model.config import ExperimentConfig
from tinymem.model.multi_token_prediction import MultiTokenPredictionHeads
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.training.checkpointing import CHECKPOINT_FORMAT_VERSION


@dataclass(frozen=True)
class LoadedBaseCheckpoint:
    """Hold a reconstructed base decoder and optional auxiliary heads."""

    model: DecoderOnlyTransformer
    mtp_heads: MultiTokenPredictionHeads | None
    config: ExperimentConfig
    vocabulary: ControlledVocabulary
    step: int


def load_base_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str,
) -> LoadedBaseCheckpoint:
    """Restore a base decoder and every trained MTP parameter."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"base checkpoint does not exist: {checkpoint_path}"
        )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("base checkpoint must contain a dictionary")
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format version")
    config_values = payload.get("config")
    if not isinstance(config_values, dict):
        raise ValueError("base checkpoint must contain an experiment config")
    config = ExperimentConfig.from_dict(config_values)
    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("base checkpoint must contain model state")
    extra = payload.get("extra")
    if not isinstance(extra, dict):
        raise ValueError("base checkpoint must contain extra metadata")

    tokens = extra.get("vocabulary")
    if not isinstance(tokens, list) or not all(
        isinstance(token, str) for token in tokens
    ):
        raise ValueError("base checkpoint must contain its vocabulary")
    if tuple(tokens[: len(SPECIAL_TOKENS)]) != SPECIAL_TOKENS:
        raise ValueError("base checkpoint has invalid special tokens")
    vocabulary = ControlledVocabulary(tokens[len(SPECIAL_TOKENS) :])
    if vocabulary.id_to_token != tuple(tokens):
        raise ValueError("base checkpoint vocabulary is not in standard order")

    raw_horizons = extra.get("mtp_horizons", [])
    if not isinstance(raw_horizons, list) or any(
        isinstance(horizon, bool) or not isinstance(horizon, int)
        for horizon in raw_horizons
    ):
        raise ValueError("base checkpoint has invalid MTP horizons")
    horizons = tuple(raw_horizons)
    if config.mtp.enabled and horizons != config.mtp.horizons:
        raise ValueError("checkpoint MTP horizons do not match its config")
    if not config.mtp.enabled and horizons:
        raise ValueError("checkpoint contains MTP heads while MTP is disabled")

    model = DecoderOnlyTransformer(config.model)
    model.load_state_dict(state)
    mtp_heads = None
    mtp_state = extra.get("mtp_head_state")
    if horizons:
        if not isinstance(mtp_state, dict):
            raise ValueError("MTP checkpoint must contain auxiliary head state")
        mtp_heads = MultiTokenPredictionHeads(
            config.model.d_model,
            config.model.vocab_size,
            horizons,
        )
        mtp_heads.load_state_dict(mtp_state)
    elif mtp_state is not None:
        raise ValueError("checkpoint has MTP head state without MTP horizons")

    model.to(device)
    if mtp_heads is not None:
        mtp_heads.to(device)
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("base checkpoint has an invalid training step")
    return LoadedBaseCheckpoint(
        model=model,
        mtp_heads=mtp_heads,
        config=config,
        vocabulary=vocabulary,
        step=step,
    )
