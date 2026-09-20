"""Bounded state collection and descriptive evaluation for phase-two writers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import torch

from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.delta_fact_data import Episode, parse_statement, replay


@dataclass(frozen=True)
class StateRecord:
    episode_id: str
    prefix_id: str
    split: str
    wording: str
    condition: str
    target: int | None
    after_write: int
    values: torch.Tensor
    valid: torch.Tensor
    truth: tuple[int, ...]
    state_norm: float | None = None
    last_update_norm: float | None = None
    last_residual_norm: float | None = None


def _frozen_writer(writer: torch.nn.Module) -> None:
    if not isinstance(writer, torch.nn.Module):
        raise TypeError("writer must be a torch module")
    if writer.training or any(module.training for module in writer.modules()):
        raise ValueError("writer must be frozen and in evaluation mode")
    parameters = tuple(writer.parameters())
    devices = {parameter.device for parameter in parameters}
    if (any(parameter.dtype != torch.float32 or parameter.requires_grad or parameter.grad is not None
            for parameter in parameters) or len(devices) > 1):
        raise ValueError("writer must be frozen FP32 with no gradients on one device")


def _feature(cache: Mapping[str, torch.Tensor], text: str, width: int) -> torch.Tensor:
    if text not in cache:
        raise ValueError(f"feature cache is missing statement: {text!r}")
    feature = cache[text]
    if (not isinstance(feature, torch.Tensor) or feature.ndim != 2 or feature.shape[0] == 0
            or feature.shape[1] != width or feature.device.type != "cpu"
            or feature.dtype != torch.float32 or feature.requires_grad or feature.grad_fn is not None):
        raise ValueError("feature cache values must be detached CPU FP32 token matrices")
    if not bool(torch.isfinite(feature).all()):
        raise ValueError("feature cache values must be finite")
    return feature.detach().clone().contiguous()


def _owned_state(state: LatentSlotState, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    if (state.values.shape != (1, 2, width) or state.valid.shape != (1, 2)
            or state.values.dtype != torch.float32 or state.valid.dtype != torch.bool
            or state.values.device != state.valid.device):
        raise ValueError("writer state must be FP32 [1, 2, width] with boolean validity")
    values = state.values.detach().to("cpu").contiguous().clone()
    valid = state.valid.detach().to("cpu").contiguous().clone()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("writer state must be finite")
    return values, valid


def _truth(statements: Sequence[Any]) -> tuple[int, ...]:
    labels = replay(tuple(statement.text if hasattr(statement, "text") else statement
                          for statement in statements))
    if any(value is None for value in labels):
        raise ValueError("episode replay must define all four facts")
    return tuple(int(value) for value in labels)


def collect_states(
    writer: torch.nn.Module,
    episodes: Sequence[Episode],
    feature_cache: Mapping[str, torch.Tensor],
) -> tuple[StateRecord, ...]:
    """Replay each episode through a frozen writer and own every endpoint state."""
    _frozen_writer(writer)
    if isinstance(episodes, (str, bytes)) or not isinstance(episodes, Sequence) or not episodes:
        raise ValueError("episodes must be a nonempty sequence")
    if not isinstance(feature_cache, Mapping):
        raise TypeError("feature_cache must be a mapping")
    if not hasattr(writer, "reader_width") or not hasattr(writer, "memory_width"):
        raise ValueError("writer must expose reader_width and memory_width")
    width = writer.memory_width
    snapshots = {name: value.detach().clone() for name, value in writer.state_dict().items()}
    records: list[StateRecord] = []
    device = next(writer.parameters()).device
    with torch.inference_mode():
        for episode in episodes:
            if not isinstance(episode, Episode):
                raise TypeError("episodes must contain Episode values")
            if len(episode.prefix) != 8 or len(episode.tail) not in (0, 8):
                raise ValueError("episodes must contain an eight-statement prefix and optional tail")
            state = writer.empty(1)
            statements = (*episode.prefix, *episode.tail)
            for index, statement in enumerate(statements, start=1):
                feature = _feature(feature_cache, statement.text, writer.reader_width)
                source_state = state
                before_values, before_valid = source_state.values.detach().clone(), source_state.valid.detach().clone()
                hidden = feature.to(device).unsqueeze(0)
                valid = torch.ones((1, feature.shape[0]), dtype=torch.bool, device=device)
                residual_norm = None
                if hasattr(writer, "encode_statement"):
                    key, value, _beta = writer.encode_statement(hidden, valid)
                    old_values = torch.where(source_state.valid.unsqueeze(-1), source_state.values,
                                             torch.zeros_like(source_state.values))
                    old_matrix = old_values.reshape(1, writer.key_width, writer.value_width)
                    residual = value - torch.bmm(key.unsqueeze(1), old_matrix).squeeze(1)
                    residual_norm = float(torch.linalg.vector_norm(residual).cpu())
                state = writer(source_state, hidden, valid)
                if (not torch.equal(hidden, feature.to(device).unsqueeze(0))
                        or not torch.equal(source_state.values, before_values)
                        or not torch.equal(source_state.valid, before_valid)):
                    raise ValueError("writer mutated its input state or feature")
                if index not in (8, 9, 16) or (index in (9, 16) and not episode.tail):
                    continue
                values, state_valid = _owned_state(state, width)
                truth = _truth(statements[:index])
                old_values = torch.where(source_state.valid.unsqueeze(-1), before_values,
                                         torch.zeros_like(before_values))
                update_norm = float(torch.linalg.vector_norm(state.values - old_values).cpu())
                state_norm = float(torch.linalg.vector_norm(
                    torch.where(state.valid.unsqueeze(-1), state.values, torch.zeros_like(state.values))
                ).cpu())
                records.append(StateRecord(
                    episode.id, episode.prefix_id, episode.split, episode.wording,
                    episode.condition, episode.target, index, values, state_valid, truth,
                    state_norm, update_norm, residual_norm,
                ))
    if any(not torch.equal(value, snapshots[name]) for name, value in writer.state_dict().items()):
        raise ValueError("writer parameters or buffers changed during collection")
    return tuple(records)


def _record_matrix(records: Sequence[StateRecord]) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], int]:
    selected: dict[tuple[str, str], StateRecord] = {}
    for record in records:
        if not isinstance(record, StateRecord):
            raise TypeError("records must contain StateRecord values")
        if record.split != "train":
            raise ValueError("probe fitting accepts training records only")
        if record.wording != "familiar":
            raise ValueError("probe fitting accepts familiar prefix records only")
        if record.after_write != 8:
            continue
        if (record.values.ndim != 3 or record.values.shape[:2] != (1, 2)
                or record.values.device.type != "cpu" or record.values.dtype != torch.float32
                or record.values.requires_grad or record.values.grad_fn is not None
                or record.valid.shape != (1, 2) or record.valid.device.type != "cpu"
                or record.valid.dtype != torch.bool or not bool(record.valid.all())):
            raise ValueError("probe records must own valid CPU FP32 states")
        if record.values.shape[-1] not in (8, 32) or not bool(torch.isfinite(record.values).all()):
            raise ValueError("probe records must contain finite width-8 or width-32 states")
        if (len(record.truth) != 4 or any(type(value) is not int or value not in (0, 1)
                                          for value in record.truth)):
            raise ValueError("probe records must contain four binary truth labels")
        key = (record.prefix_id, record.wording)
        old = selected.get(key)
        if old is not None:
            if (not torch.equal(old.values, record.values) or not torch.equal(old.valid, record.valid)
                    or old.truth != record.truth):
                raise ValueError("duplicate prefix rows disagree")
        else:
            selected[key] = record
    ordered = [selected[key] for key in sorted(selected)]
    if len(ordered) < 4:
        raise ValueError("at least four training prefixes are required")
    x = np.asarray([row.values.detach().cpu().numpy().reshape(-1) for row in ordered], dtype=np.float64)
    y = np.asarray([row.truth for row in ordered], dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(ordered), 4) or not np.isfinite(x).all():
        raise ValueError("records must contain finite four-bit training states")
    prefixes = tuple(sorted({row.prefix_id for row in ordered}))
    groups = np.asarray([prefixes.index(row.prefix_id) % 4 for row in ordered], dtype=np.int64)
    if set(groups.tolist()) != set(range(4)):
        raise ValueError("four nonempty deterministic prefix folds are required")
    return x, y, tuple(str(group) for group in groups), len(prefixes)


def _fit(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, Any]:
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale == 0, 1.0, scale)
    z = (x - mean) / scale
    bias = y.mean(0)
    normal = z.T @ z / (4 * len(x)) + alpha * np.eye(x.shape[1])
    rhs = z.T @ (y - bias) / (4 * len(x))
    weights = np.linalg.solve(normal, rhs)
    if not np.isfinite(weights).all():
        raise ValueError("nonfinite fitted coefficients")
    return {"mean": mean, "scale": scale, "weights": weights, "bias": bias, "alpha": float(alpha)}


def _scores(model: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    return ((x - np.asarray(model["mean"])) / np.asarray(model["scale"])) @ np.asarray(model["weights"]) + np.asarray(model["bias"])


def fit_probe(records: Sequence[StateRecord]) -> dict[str, Any]:
    """Fit a train-only, grouped four-fold ridge probe over prefix states."""
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence) or not records:
        raise ValueError("records must be a nonempty sequence")
    x, y, group_strings, prefix_count = _record_matrix(records)
    groups = np.asarray([int(value) for value in group_strings])
    alphas = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
    cv: list[dict[str, Any]] = []
    for alpha in alphas:
        fold_rows = []
        pooled_errors = []
        for fold in range(4):
            train, test = groups != fold, groups == fold
            fold_model = _fit(x[train], y[train], alpha)
            error = _scores(fold_model, x[test]) - y[test]
            pooled_errors.append(error)
            fold_rows.append({"fold": fold, "mse": float(np.square(error).mean()), "n": int(error.size)})
        cv.append({"alpha": alpha, "mse": float(np.square(np.concatenate(pooled_errors)).mean()), "folds": fold_rows})
    minimum = min(row["mse"] for row in cv)
    chosen = max(row["alpha"] for row in cv if np.isclose(row["mse"], minimum, rtol=1e-12, atol=1e-15))
    model = _fit(x, y, chosen)
    result: dict[str, Any] = {
        "feature_width": int(x.shape[1] // 2), "n_train": int(len(x)),
        "n_prefixes": int(prefix_count),
        "alpha": float(chosen), "mean": model["mean"].tolist(), "scale": model["scale"].tolist(),
        "weights": model["weights"].tolist(), "bias": model["bias"].tolist(),
        "cv": cv, "groups": [int(group) for group in groups], "threshold": 0.5,
    }
    return result


def predict_probe(model: Mapping[str, Any], values_batch: torch.Tensor | np.ndarray) -> np.ndarray:
    """Return four raw fact scores from flattened or two-slot states."""
    if not isinstance(model, Mapping):
        raise TypeError("model must be a mapping")
    values = values_batch.detach().cpu().numpy() if isinstance(values_batch, torch.Tensor) else np.asarray(values_batch)
    if not np.issubdtype(values.dtype, np.number) or values.ndim not in (2, 3):
        raise ValueError("values_batch must have shape [N, 2, width] or [N, 2 * width]")
    if values.ndim == 3 and values.shape[1] == 2:
        x = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    elif values.ndim == 2:
        x = np.asarray(values, dtype=np.float64)
    else:
        raise ValueError("values_batch must have shape [N, 2, width] or [N, 2 * width]")
    if not np.isfinite(x).all():
        raise ValueError("values_batch must be finite")
    scores = _scores(model, x)
    if scores.shape[1] != 4:
        raise ValueError("probe model must predict four bits")
    return scores


def transition_counts(before_correct: Sequence[bool], after_correct: Sequence[bool]) -> dict[str, int | float | None]:
    before = np.asarray(before_correct, dtype=bool)
    after = np.asarray(after_correct, dtype=bool)
    if before.shape != after.shape or before.ndim == 0:
        raise ValueError("before_correct and after_correct must have equal non-scalar shape")
    old = before.reshape(-1)
    new = after.reshape(-1)
    n = int(old.size)
    before_n, after_n = int(old.sum()), int(new.sum())
    damage, repair = int(np.sum(old & ~new)), int(np.sum(~old & new))
    retained, remained_wrong = int(np.sum(old & new)), int(np.sum(~old & ~new))
    return {
        "n": n, "before_correct": before_n, "after_correct": after_n,
        "retained_correct": retained, "remained_wrong": remained_wrong,
        "damage": damage, "repair": repair,
        "damage_rate": damage / before_n if before_n else None,
        "repair_rate": repair / (n - before_n) if n - before_n else None,
        "change_pp_all": 100.0 * (after_n - before_n) / n if n else None,
    }


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {"n": int(array.size), "mean": float(array.mean()) if array.size else 0.0,
            "min": float(array.min()) if array.size else 0.0, "max": float(array.max()) if array.size else 0.0}


def delta_geometry(writer: torch.nn.Module, feature_cache: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Describe delta-writer keys for same-entity and cross-entity statements."""
    _frozen_writer(writer)
    if not hasattr(writer, "encode_statement"):
        raise TypeError("delta_geometry requires a writer exposing encode_statement")
    entries: list[tuple[int, int, str, torch.Tensor]] = []
    betas: list[float] = []
    key_norms: list[float] = []
    device = next(writer.parameters()).device
    with torch.inference_mode():
        for text in feature_cache:
            parsed = parse_statement(text)
            hidden = _feature(feature_cache, text, writer.reader_width).to(device).unsqueeze(0)
            valid = torch.ones((1, hidden.shape[1]), dtype=torch.bool, device=device)
            key, _value, beta = writer.encode_statement(hidden, valid)
            entries.append((parsed.entity, parsed.value, text, key[0].detach().cpu().clone()))
            betas.append(float(beta[0]))
            key_norms.append(float(torch.linalg.vector_norm(key[0])))

    def cosine_pairs(predicate: Any) -> list[float]:
        values = []
        for left_row, right_row in combinations(entries, 2):
            if predicate(left_row, right_row):
                values.append(float(torch.nn.functional.cosine_similarity(
                    left_row[3].unsqueeze(0), right_row[3].unsqueeze(0)).item()))
        return values

    same_fact = cosine_pairs(lambda left, right: left[:2] == right[:2] and left[2] != right[2])
    opposite = cosine_pairs(lambda left, right: left[0] == right[0] and left[1] != right[1])
    cross_entity = cosine_pairs(lambda left, right: left[0] != right[0])
    same_summary, opposite_summary = _summary(same_fact), _summary(opposite)
    cross_summary = _summary(cross_entity)
    return {
        "same_fact_different_wording": same_summary,
        "same_entity_opposite_value": opposite_summary,
        "cross_entity": cross_summary,
        "beta": _summary(betas),
        "key_norm": _summary(key_norms), "zero_keys": sum(value == 0 for value in key_norms),
    }
