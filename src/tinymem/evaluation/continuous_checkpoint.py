"""Load trained continuous-memory decoders for frozen evaluation."""

from dataclasses import dataclass
from pathlib import Path

import torch

from tinymem.data.vocabulary import SPECIAL_TOKENS, ControlledVocabulary
from tinymem.memory.controller import AdaptiveWriteController
from tinymem.memory.continuous import MultiSlotAttentionMemoryCompressor
from tinymem.memory.discrete_compressor import DiscreteMemoryCompressor
from tinymem.memory.recurrent_memory import GatedRecurrentMemoryBank
from tinymem.memory.write_gate import TokenSegmentWriteGate
from tinymem.model.config import ExperimentConfig
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.multi_token_prediction import MultiTokenPredictionHeads
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.training.checkpointing import CHECKPOINT_FORMAT_VERSION


GATED_MULTISLOT_ARCHITECTURE = (
    "segmented_continuous_multislot_attention_pool_gated_update"
)
TOKEN_GATED_MULTISLOT_ARCHITECTURE = (
    "segmented_continuous_multislot_attention_pool_token_gated_update"
)
DISCRETE_TOKEN_GATED_ARCHITECTURE = (
    "segmented_discrete_gumbel_codebook_token_gated_update"
)
ADAPTIVE_MULTISLOT_ARCHITECTURE = (
    "segmented_continuous_multislot_attention_pool_adaptive_gated_update"
)
ADAPTIVE_DISCRETE_ARCHITECTURE = (
    "segmented_discrete_gumbel_codebook_adaptive_gated_update"
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
        DISCRETE_TOKEN_GATED_ARCHITECTURE,
        ADAPTIVE_MULTISLOT_ARCHITECTURE,
        ADAPTIVE_DISCRETE_ARCHITECTURE,
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
    memory_position_mode = extra.get("memory_position_mode", "absolute")
    if memory_position_mode not in ("absolute", "virtual"):
        raise ValueError(
            "continuous checkpoint has an invalid memory position mode"
        )
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
    raw_mtp_horizons = extra.get("mtp_horizons", [])
    if not isinstance(raw_mtp_horizons, list) or any(
        isinstance(horizon, bool) or not isinstance(horizon, int)
        for horizon in raw_mtp_horizons
    ):
        raise ValueError("continuous checkpoint has invalid MTP horizons")
    mtp_horizons = tuple(raw_mtp_horizons)
    if config.mtp.enabled and mtp_horizons != config.mtp.horizons:
        raise ValueError("checkpoint MTP horizons do not match its config")
    if not config.mtp.enabled and mtp_horizons:
        raise ValueError("checkpoint contains MTP heads while MTP is disabled")
    is_discrete = architecture in (
        DISCRETE_TOKEN_GATED_ARCHITECTURE,
        ADAPTIVE_DISCRETE_ARCHITECTURE,
    )
    is_adaptive = architecture in (
        ADAPTIVE_MULTISLOT_ARCHITECTURE,
        ADAPTIVE_DISCRETE_ARCHITECTURE,
    )
    queries = state.get(
        "compressor.summarizer.queries"
        if is_discrete
        else "compressor.queries"
    )
    if not isinstance(queries, torch.Tensor) or queries.ndim != 2:
        raise ValueError("continuous checkpoint has invalid compressor queries")
    if queries.shape[1] != config.model.d_model:
        raise ValueError("compressor query width does not match the model")
    write_gate_kernel_size = 3
    if architecture in (
        TOKEN_GATED_MULTISLOT_ARCHITECTURE,
        DISCRETE_TOKEN_GATED_ARCHITECTURE,
    ):
        pattern_weight = state.get("write_gate.patterns.weight")
        if (
            not isinstance(pattern_weight, torch.Tensor)
            or pattern_weight.ndim != 3
            or pattern_weight.shape[:2]
            != (config.model.d_model, config.model.d_model)
            or pattern_weight.shape[2] <= 0
            or pattern_weight.shape[2] % 2 == 0
        ):
            raise ValueError("continuous checkpoint has invalid write gate patterns")
        write_gate_kernel_size = pattern_weight.shape[2]

    controller_hidden_width = config.model.d_model
    if is_adaptive:
        controller_input_weight = state.get(
            "write_controller.input_projection.weight"
        )
        controller_action_weight = state.get(
            "write_controller.action_projection.weight"
        )
        if (
            not isinstance(controller_input_weight, torch.Tensor)
            or controller_input_weight.ndim != 2
            or controller_input_weight.shape[1]
            != 2 * config.model.d_model + 1
            or not isinstance(controller_action_weight, torch.Tensor)
            or controller_action_weight.shape
            != (2, controller_input_weight.shape[0])
        ):
            raise ValueError(
                "continuous checkpoint has an invalid adaptive controller"
            )
        controller_hidden_width = controller_input_weight.shape[0]

    if is_discrete:
        codebook_weight = state.get("compressor.codebook.embedding.weight")
        if (
            not isinstance(codebook_weight, torch.Tensor)
            or codebook_weight.ndim != 2
            or codebook_weight.shape[1] != config.model.d_model
            or codebook_weight.shape[0] != config.memory.codebook_size
        ):
            raise ValueError("continuous checkpoint has an invalid codebook")
        compressor = DiscreteMemoryCompressor(
            config.model.d_model,
            codebook_size=codebook_weight.shape[0],
            summary_slots=queries.shape[0],
        )
    else:
        compressor = MultiSlotAttentionMemoryCompressor(
            config.model.d_model,
            summary_slots=queries.shape[0],
        )

    decoder = SegmentedContinuousDecoder(
        DecoderOnlyTransformer(config.model),
        compressor,
        GatedRecurrentMemoryBank(
            capacity=config.memory.n_slots,
            model_width=config.model.d_model,
            write_threshold=float(write_threshold),
        ),
        segment_length=config.stream.segment_length,
        write_gate=(
            TokenSegmentWriteGate(
                config.model.d_model,
                kernel_size=write_gate_kernel_size,
            )
            if architecture
            in (
                TOKEN_GATED_MULTISLOT_ARCHITECTURE,
                DISCRETE_TOKEN_GATED_ARCHITECTURE,
            )
            else None
        ),
        write_controller=(
            AdaptiveWriteController(
                config.model.d_model,
                hidden_width=controller_hidden_width,
            )
            if is_adaptive
            else None
        ),
        mtp_heads=(
            MultiTokenPredictionHeads(
                config.model.d_model,
                config.model.vocab_size,
                mtp_horizons,
            )
            if mtp_horizons
            else None
        ),
        memory_position_mode=memory_position_mode,
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
