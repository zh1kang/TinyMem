"""State-supervised training for the fixed 258-byte delta writer.

The writer sees only the current statement's frozen reader features and its
own preceding state.
Oracle states are targets, never inputs to the primary rollout.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from tinymem.memory.delta_slots import DeltaSlotWriter

FACT_CELLS: tuple[int, ...] = (0, 8, 16, 24)


@dataclass(frozen=True)
class DistilledExample:
    """One history with one target state for every actual write."""

    episode_id: str
    split: str
    features: tuple[torch.Tensor, ...]
    targets: tuple[torch.Tensor, ...]
    truth: tuple[tuple[int | None, ...], ...]


def _check_writer(writer: DeltaSlotWriter, *, training: bool) -> None:
    if not isinstance(writer, DeltaSlotWriter):
        raise TypeError("writer must be a DeltaSlotWriter")
    if writer.reader_width <= 0 or writer.memory_width != 32 or writer.key_width != 8:
        raise ValueError("writer must use the declared 32-wide, key-width-8 state")
    parameters = tuple(writer.parameters())
    if any(parameter.dtype != torch.float32 for parameter in parameters):
        raise TypeError("writer parameters must use float32")
    if {parameter.device.type for parameter in parameters} != {"cpu"}:
        raise ValueError("distilled writer training is CPU-only")
    if not training and (
        any(module.training for module in writer.modules())
        or any(parameter.requires_grad or parameter.grad is not None for parameter in parameters)
    ):
        raise ValueError("diagnostic writer must be frozen and in evaluation mode")


def _check_feature(feature: object, width: int) -> torch.Tensor:
    if (not isinstance(feature, torch.Tensor) or feature.ndim != 2 or feature.shape[0] == 0
            or feature.shape[1] != width or feature.device.type != "cpu"
            or feature.dtype != torch.float32 or feature.requires_grad or feature.grad_fn is not None):
        raise ValueError("features must be detached CPU FP32 token matrices")
    if not bool(torch.isfinite(feature).all()):
        raise ValueError("features must be finite")
    return feature


def _check_target(target: object, writer: DeltaSlotWriter) -> torch.Tensor:
    if (not isinstance(target, torch.Tensor) or target.shape != (1, 2, writer.memory_width)
            or target.device.type != "cpu" or target.dtype != torch.float32
            or target.requires_grad or target.grad_fn is not None):
        raise ValueError("targets must be detached CPU FP32 [1, 2, 32] states")
    if not bool(torch.isfinite(target).all()):
        raise ValueError("targets must be finite")
    return target


def _check_example(example: DistilledExample, writer: DeltaSlotWriter, *, training: bool) -> None:
    if not isinstance(example, DistilledExample):
        raise TypeError("examples must contain DistilledExample values")
    if training and example.split != "train":
        raise ValueError("optimization accepts training examples only")
    if not example.episode_id or not example.features or len(example.features) != len(example.targets):
        raise ValueError("an example needs one target for every feature sequence")
    if len(example.truth) != len(example.features):
        raise ValueError("an example needs one truth tuple for every write")
    for feature, target, truth in zip(example.features, example.targets, example.truth):
        _check_feature(feature, writer.reader_width)
        _check_target(target, writer)
        if len(truth) != 4 or any(value is not None and value not in (0, 1) for value in truth):
            raise ValueError("truth must contain four binary values or None")


def _state_sse(state: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    residual = state - target
    total = residual.square().sum()
    off_fact = residual.reshape(-1).square()[
        [index for index in range(residual.numel()) if index not in FACT_CELLS]
    ].sum()
    return total, off_fact


def _rollout(
    writer: DeltaSlotWriter,
    example: DistilledExample,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return mean per-history write SSE and off-fact SSE."""

    state = writer.empty(1)
    total = state.values.new_zeros(())
    off_fact = state.values.new_zeros(())
    for feature, target in zip(example.features, example.targets):
        valid = torch.ones((1, feature.shape[0]), dtype=torch.bool, device="cpu")
        state = writer(state, feature.unsqueeze(0), valid)
        state_loss, off_loss = _state_sse(state.values, target)
        total = total + state_loss
        off_fact = off_fact + off_loss
    writes = len(example.features)
    return total / writes, off_fact / writes, writes


def rollout_loss(writer: DeltaSlotWriter, examples: Sequence[DistilledExample]) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the batch loss before an optimizer step.

    Each history receives equal weight.
    Within a history, every actual write receives equal weight.
    Each write is a sum over all 64 stored coordinates, so unused coordinates
    are supervised instead of diluted by an additional coordinate mean.
    """

    if isinstance(examples, (str, bytes)) or not isinstance(examples, Sequence) or not examples:
        raise ValueError("a training batch must contain examples")
    _check_writer(writer, training=True)
    for example in examples:
        _check_example(example, writer, training=True)
    losses = [_rollout(writer, example)[:2] for example in examples]
    return torch.stack([item[0] for item in losses]).mean(), torch.stack([item[1] for item in losses]).mean()


def train_batch(
    writer: DeltaSlotWriter,
    examples: Sequence[DistilledExample],
    optimizer: torch.optim.Optimizer,
    *,
    clip_norm: float = 1.0,
) -> dict[str, float | int | bool]:
    """Optimize a batch of writer trajectories on CPU FP32."""

    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    if type(clip_norm) is not float or clip_norm <= 0:
        raise ValueError("clip_norm must be a positive float")
    loss, off_fact_loss = rollout_loss(writer, examples)
    parameters = tuple(writer.parameters())
    optimized = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if len(optimized) != len(parameters) or {id(p) for p in optimized} != {id(p) for p in parameters}:
        raise ValueError("optimizer must own exactly the writer parameters")
    optimizer.zero_grad(set_to_none=True)
    if not bool(torch.isfinite(loss)):
        raise ValueError("nonfinite state loss")
    loss.backward()
    if any(parameter.grad is None for parameter in parameters):
        raise ValueError("every writer parameter must receive a gradient")
    if any(not bool(torch.isfinite(parameter.grad).all()) for parameter in parameters):
        raise ValueError("nonfinite writer gradient")
    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, clip_norm, error_if_nonfinite=True)
    optimizer.step()
    if any(not bool(torch.isfinite(parameter).all()) for parameter in parameters):
        raise ValueError("optimizer produced nonfinite writer parameters")
    writes = sum(len(example.features) for example in examples)
    return {
        "state_loss": float(loss.detach()),
        "off_fact_loss": float(off_fact_loss.detach()),
        "gradient_norm": float(gradient_norm),
        "histories": len(examples),
        "write_calls": writes,
        "attached_state_gradients": True,
        "persistent_bytes": 2 * writer.memory_width * 4 + 2,
    }


def _direct_bits(values: torch.Tensor) -> list[int | None]:
    flat = values.detach().reshape(-1)
    return [None if float(flat[index]) == 0.0 else int(float(flat[index]) > 0.0)
            for index in FACT_CELLS]


def _truth_from_episode(episode: Any, count: int) -> tuple[int | None, ...]:
    from tinymem.studies.delta.data import replay

    statements = tuple(statement.text for statement in (*episode.prefix, *episode.tail)[:count])
    return tuple(replay(statements))


@torch.no_grad()
def trajectory_diagnostics(
    writer: DeltaSlotWriter,
    episodes: Sequence[Any],
    feature_cache: Mapping[str, torch.Tensor],
) -> list[dict[str, Any]]:
    """Measure learned rollouts and detached one-step oracle-state transitions."""

    _check_writer(writer, training=False)
    if isinstance(episodes, (str, bytes)) or not isinstance(episodes, Sequence) or not episodes:
        raise ValueError("episodes must be a nonempty sequence")
    if not isinstance(feature_cache, Mapping):
        raise TypeError("feature_cache must be a mapping")
    from tinymem.studies.delta.data import Episode
    from tinymem.studies.oracle.state import _state_for_texts

    rows: list[dict[str, Any]] = []
    for episode in episodes:
        if not isinstance(episode, Episode):
            raise TypeError("episodes must contain Episode values")
        statements = (*episode.prefix, *episode.tail)
        state = writer.empty(1)
        for index, statement in enumerate(statements, start=1):
            feature = _check_feature(feature_cache.get(statement.text), writer.reader_width)
            target = _state_for_texts(tuple(item.text for item in statements[:index]))
            target_values = target.values.detach().clone().contiguous()
            state = writer(state, feature.unsqueeze(0), torch.ones((1, feature.shape[0]), dtype=torch.bool))
            state_sse, off_fact_sse = _state_sse(state.values, target_values)
            if index == 1:
                one_step_state_sse, one_step_off_fact_sse = state_sse, off_fact_sse
            else:
                previous_target = _state_for_texts(tuple(item.text for item in statements[:index - 1]))
                one_step = writer(
                    previous_target,
                    feature.unsqueeze(0),
                    torch.ones((1, feature.shape[0]), dtype=torch.bool),
                )
                one_step_state_sse, one_step_off_fact_sse = _state_sse(one_step.values, target_values)
            truth = _truth_from_episode(episode, index)
            bits = _direct_bits(state.values)
            known = [(bit, expected) for bit, expected in zip(bits, truth) if expected is not None]
            rows.append({
                "episode_id": episode.id,
                "prefix_id": episode.prefix_id,
                "wording": episode.wording,
                "condition": episode.condition,
                "target": episode.target,
                "after_write": index,
                "state_sse": float(state_sse),
                "off_fact_sse": float(off_fact_sse),
                "direct_bits": bits,
                "truth": list(truth),
                "known_fact_correct": sum(bit == expected for bit, expected in known),
                "known_fact_count": len(known),
                "one_step_state_sse": float(one_step_state_sse),
                "one_step_off_fact_sse": float(one_step_off_fact_sse),
            })
    return rows


def build_example(episode: Any, feature_cache: Mapping[str, torch.Tensor], writer: DeltaSlotWriter) -> DistilledExample:
    """Build a train or validation trajectory with oracle targets."""

    from tinymem.studies.delta.data import Episode, replay
    from tinymem.studies.oracle.state import _state_for_texts

    if not isinstance(episode, Episode):
        raise TypeError("episode must be an Episode")
    statements = (*episode.prefix, *episode.tail)
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    truths: list[tuple[int | None, ...]] = []
    for index, statement in enumerate(statements, start=1):
        feature = _check_feature(feature_cache.get(statement.text), writer.reader_width)
        features.append(feature.detach().clone().contiguous())
        oracle_state = _state_for_texts(tuple(item.text for item in statements[:index]))
        targets.append(oracle_state.values.detach().clone().contiguous())
        truths.append(tuple(replay(tuple(item.text for item in statements[:index]))))
    return DistilledExample(episode.id, episode.split, tuple(features), tuple(targets), tuple(truths))


def aggregate_validation(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Aggregate diagnostics by condition for descriptive validation curves."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["condition"]), []).append(row)
    result: dict[str, dict[str, float | int]] = {}
    for condition, values in grouped.items():
        total_known = sum(int(row["known_fact_count"]) for row in values)
        total_correct = sum(int(row["known_fact_correct"]) for row in values)
        result[condition] = {
            "writes": len(values),
            "mean_state_sse": sum(float(row["state_sse"]) for row in values) / len(values),
            "mean_off_fact_sse": sum(float(row["off_fact_sse"]) for row in values) / len(values),
            "direct_known_fact_accuracy": total_correct / total_known if total_known else 0.0,
            "known_fact_count": total_known,
        }
    return result
