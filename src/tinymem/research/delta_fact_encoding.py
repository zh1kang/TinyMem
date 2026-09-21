"""Frozen statement features and native readout labels for delta fact episodes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib

import torch

from tinymem.data.reader_gate import ReaderCase
from tinymem.research.adapted_readout import frozen_history_features
from tinymem.research.delta_fact_data import ENTITIES, ROOM_PAIRS, Episode, parse_statement, replay
from tinymem.research.delta_fact_training import Endpoint, TrainingExample
from tinymem.research.memory_prompt import encode_memory_example
from tinymem.research.pretrained import PretrainedReader
from tinymem.research.adapted_readout import ReadoutQuery


FeatureCache = Mapping[str, torch.Tensor]


def _require_unadapted_reader(reader: PretrainedReader) -> None:
    if hasattr(reader.model, "peft_config"):
        raise ValueError("feature cache requires the unadapted base reader")


def _statement_case(statement_text: str) -> ReaderCase:
    parsed = parse_statement(statement_text)
    identity = hashlib.sha256(statement_text.encode("utf-8")).hexdigest()
    return ReaderCase(
        case_id=f"delta-fact-statement:{identity}",
        category="update_known",
        history_id=f"delta-fact-statement:{identity}",
        context=statement_text,
        question=f"Where is {ENTITIES[parsed.entity]}?",
        answer=ROOM_PAIRS[parsed.entity][parsed.value],
    )


def build_feature_cache(
    reader: PretrainedReader, episodes: Sequence[Episode],
) -> dict[str, torch.Tensor]:
    """Encode each unique statement independently at position zero.

    The reader must still be the frozen base reader.
    Each returned feature matrix is a detached, contiguous CPU FP32 copy.
    """

    _require_unadapted_reader(reader)
    if isinstance(episodes, Episode):
        episodes = (episodes,)
    elif not isinstance(episodes, Sequence) or isinstance(episodes, (str, bytes)) or not episodes:
        raise ValueError("episodes must be a nonempty sequence")
    statements: dict[str, None] = {}
    for episode in episodes:
        if not isinstance(episode, Episode):
            raise TypeError("episodes must contain Episode values")
        for statement in (*episode.prefix, *episode.tail):
            statements.setdefault(statement.text, None)

    cache: dict[str, torch.Tensor] = {}
    for statement_text in statements:
        native = encode_memory_example(reader, _statement_case(statement_text))
        features = frozen_history_features(reader, native.history_ids)
        if (features.ndim != 2 or features.shape[0] == 0 or features.dtype != torch.float32
                or features.device.type != "cpu" or features.requires_grad or features.grad_fn is not None
                or not torch.isfinite(features).all()):
            raise ValueError("statement features must be detached CPU FP32 token matrices")
        cache[statement_text] = features.detach().cpu().contiguous().clone()
    return cache


def _cached_feature(feature_cache: FeatureCache, text: str, reader_width: int) -> torch.Tensor:
    if text not in feature_cache:
        raise ValueError(f"feature cache is missing statement: {text!r}")
    feature = feature_cache[text]
    if (not isinstance(feature, torch.Tensor) or feature.ndim != 2 or feature.shape[0] == 0
            or feature.shape[1] != reader_width or feature.dtype != torch.float32
            or feature.device.type != "cpu" or feature.requires_grad or feature.grad_fn is not None):
        raise ValueError("feature cache values must be detached CPU FP32 token matrices")
    if not torch.isfinite(feature).all():
        raise ValueError("feature cache values must be finite")
    return feature.detach().clone().contiguous()


def _queries(
    reader: PretrainedReader, episode: Episode, statements: Sequence, after_write: int,
) -> tuple[tuple[int, ...], tuple[ReadoutQuery, ...]]:
    world = replay(statements)
    if len(world) != len(ENTITIES) or any(value is None for value in world):
        raise ValueError("episode replay must define all four entities")
    context = "\n".join(statement.text for statement in statements)
    target = "none" if episode.target is None else str(episode.target)
    branch_id = f"{episode.prefix_id}/{episode.condition}-{target}"
    queries: list[ReadoutQuery] = []
    before_ids: tuple[int, ...] | None = None
    for entity, value in enumerate(world):
        if value is None:
            raise ValueError("episode replay must define all four entities")
        case = ReaderCase(
            case_id=f"{branch_id}:write-{after_write}:entity-{entity}",
            category="update_known",
            history_id=f"{branch_id}:write-{after_write}",
            context=context,
            question=f"Where is {ENTITIES[entity]}?",
            answer=ROOM_PAIRS[entity][value],
        )
        native = encode_memory_example(reader, case)
        if before_ids is None:
            before_ids = native.before_ids
        elif native.before_ids != before_ids:
            raise ValueError("native questions do not share a stable before prompt")
        queries.append(ReadoutQuery(
            case.case_id, case.category, case.answer, native.after_ids, native.answer_ids,
        ))
    if before_ids is None:
        raise ValueError("episode must produce four native questions")
    return before_ids, tuple(queries)


def encode_episode(
    reader: PretrainedReader, episode: Episode, feature_cache: FeatureCache,
) -> TrainingExample:
    """Encode one episode with independent replay labels at each write endpoint."""

    if not isinstance(episode, Episode):
        raise TypeError("episode must be an Episode")
    if not isinstance(feature_cache, Mapping):
        raise TypeError("feature_cache must be a mapping from statement text to features")
    if len(episode.prefix) != 8 or len(episode.tail) not in (0, 8):
        raise ValueError("episodes must contain an eight-statement prefix and optional eight-statement tail")
    statements = (*episode.prefix, *episode.tail)
    features = tuple(_cached_feature(feature_cache, statement.text, reader.model.config.hidden_size)
                     for statement in statements)

    endpoints: list[Endpoint] = []
    before_ids: tuple[int, ...] | None = None
    for after_write in (8, 16) if episode.tail else (8,):
        current = statements[:after_write]
        endpoint_before_ids, queries = _queries(reader, episode, current, after_write)
        if before_ids is None:
            before_ids = endpoint_before_ids
        elif endpoint_before_ids != before_ids:
            raise ValueError("native endpoints do not share a stable before prompt")
        endpoints.append(Endpoint(after_write, queries))

    if before_ids is None:
        raise ValueError("episode must produce at least one endpoint")
    return TrainingExample(episode.id, before_ids, features, tuple(endpoints), split=episode.split)
