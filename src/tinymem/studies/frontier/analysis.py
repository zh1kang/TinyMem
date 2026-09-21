"""Pure-data analysis for the storage-frontier evaluation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

_TASKS = ("qa1", "qa2", "qa3", "qa4", "qa5")
_MODES = ("learned", "compressed_recent", "compressed_diverse", "dictionary_recent")
_CONTROLS = {"zero", "donor", "full_text"}
_BUDGETS = (64, 256, 1024)
_LEVELS = (0, 2)
_SEEDS = (27001, 27002, 27003)


def _number(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return value


def _validate_row(row: Mapping[str, object]) -> None:
    required = {
        "id", "story", "task", "answer", "mode", "budget", "level", "cell", "seed",
        "correct", "normalized_correct", "payload_bytes",
    }
    if not isinstance(row, Mapping) or not required <= row.keys():
        raise ValueError("each row must contain the complete storage-evaluation schema")
    for field in ("id", "story", "task", "answer", "mode"):
        if not isinstance(row[field], str) or not row[field].strip():
            raise ValueError(f"{field} must be a nonempty string")
    if row["task"] not in _TASKS or row["mode"] not in (*_MODES, *_CONTROLS):
        raise ValueError("unknown task or evaluation mode")
    if type(row["seed"]) is not int or row["seed"] not in _SEEDS:
        raise ValueError("seed must be one of the three declared optimization seeds")
    if type(row["level"]) is not int or row["level"] not in _LEVELS:
        raise ValueError("level must be zero or two")
    if type(row["cell"]) is not int or not 0 <= row["cell"] <= 11:
        raise ValueError("cell must be an integer from zero through eleven")
    if type(row["correct"]) is not bool or type(row["normalized_correct"]) is not bool:
        raise ValueError("correctness fields must be booleans")
    budget, payload = row["budget"], row["payload_bytes"]
    if budget is not None and (isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0):
        raise ValueError("budget must be a positive integer or None")
    if payload is not None and (isinstance(payload, bool) or not isinstance(payload, int) or payload < 0):
        raise ValueError("payload_bytes must be a nonnegative integer or None")


def _metric_rows(rows: Mapping[tuple[str, int, str], Mapping[str, object]], metric: str) -> dict[str, float]:
    return {str(seed): float(sum(bool(row[metric]) for row in rows.values() if row["seed"] == seed)
                              / sum(row["seed"] == seed for row in rows.values())) for seed in _SEEDS}


def _stats(values: Mapping[str, float]) -> dict[str, object]:
    numbers = np.asarray(list(values.values()), dtype=float)
    return {
        "per_seed": dict(values),
        "mean": float(numbers.mean()),
        "min": float(numbers.min()),
        "max": float(numbers.max()),
    }


def _payload_stats(rows: Mapping[tuple[str, int, str], Mapping[str, object]]) -> dict[str, object]:
    by_seed = {str(seed): float(np.mean([row["payload_bytes"] for row in rows.values()
                                         if row["seed"] == seed])) for seed in _SEEDS}
    values = np.asarray([row["payload_bytes"] for row in rows.values()], dtype=float)
    return {"per_seed_mean": by_seed, "mean": float(values.mean()),
            "min": int(values.min()), "max": int(values.max())}


def _bootstrap(task_values: dict[str, tuple[np.ndarray, np.ndarray]], samples: int,
               seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.empty(samples, dtype=float)
    chunk_size = 256
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        task_draws = []
        for task in _TASKS:
            differences, counts = task_values[task]
            stories = len(differences)
            weights = rng.multinomial(stories, np.full(stories, 1 / stories), size=stop - start)
            task_draws.append((weights @ differences) / (weights @ counts))
        result[start:stop] = np.mean(np.stack(task_draws, axis=1), axis=1)
    return result


def _comparison_metric(
    left: Mapping[tuple[str, int, str], Mapping[str, object]],
    right: Mapping[tuple[str, int, str], Mapping[str, object]],
    metric: str, *, bootstrap_samples: int, seed: int,
) -> dict[str, object]:
    seed_diffs: dict[str, list[float]] = {}
    task_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    task_means: dict[str, float] = {}
    for task in _TASKS:
        questions = sorted({key[2] for key in left if key[0] == task})
        per_story: dict[str, list[float]] = {}
        for question in questions:
            values = [float(left[task, opt_seed, question][metric]) - float(right[task, opt_seed, question][metric])
                      for opt_seed in _SEEDS]
            story = str(left[task, _SEEDS[0], question]["story"])
            per_story.setdefault(story, []).append(float(np.mean(values)))
        stories = sorted(per_story)
        differences = np.asarray([sum(per_story[story]) for story in stories], dtype=float)
        counts = np.asarray([len(per_story[story]) for story in stories], dtype=float)
        task_values[task] = differences, counts
        task_means[task] = float(differences.sum() / counts.sum())
        for opt_seed in _SEEDS:
            values = [float(left[task, opt_seed, question][metric]) - float(right[task, opt_seed, question][metric])
                      for question in questions]
            seed_diffs.setdefault(str(opt_seed), []).append(float(np.mean(values)))
    seed_diffs = {opt_seed: float(np.mean(values)) for opt_seed, values in seed_diffs.items()}
    draws = _bootstrap(task_values, bootstrap_samples, seed)
    return {
        "task_difference": task_means,
        "seed_difference": _stats(seed_diffs),
        "mean_difference": float(np.mean(list(task_means.values()))),
        "bootstrap": {
            "bounds": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
            "confidence": 0.95, "resamples": bootstrap_samples, "seed": seed,
            "unit": "whole_story_paired_within_task",
            "interpretation": "conditional_on_observed_optimization_seeds",
        },
    }


def analyze(rows: list[dict], *, bootstrap_samples: int = 2000,
            seed: int = 2026091806, expected_per_task: int = 1000) -> dict:
    """Validate the sealed design and summarize every declared primary comparison."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("rows must be a nonempty list")
    _number(bootstrap_samples, "bootstrap_samples", minimum=2)
    _number(seed, "seed")
    _number(expected_per_task, "expected_per_task", minimum=1)
    for row in rows:
        _validate_row(row)

    primary = [row for row in rows if row["mode"] in _MODES]
    modes = tuple(sorted({row["mode"] for row in primary}, key=_MODES.index))
    if "learned" not in modes or not any(mode != "learned" for mode in modes):
        raise ValueError("primary rows must include all four declared methods")
    if set(modes) != set(_MODES):
        raise ValueError("primary rows must include all four declared methods")
    for row in primary:
        if row["budget"] not in _BUDGETS or row["payload_bytes"] is None:
            raise ValueError("primary rows require one declared budget and payload size")
        if row["payload_bytes"] > row["budget"]:
            raise ValueError("accepted primary payload exceeds its byte budget")

    indexed: dict[tuple[str, int, int, str, int], dict[tuple[str, int, str], dict]] = {}
    for row in primary:
        cell = (row["mode"], row["budget"], row["level"], row["task"], row["seed"])
        keyed = indexed.setdefault(cell, {})
        question = (row["task"], row["seed"], row["id"])
        if question in keyed:
            raise ValueError("duplicate primary question row")
        keyed[question] = row
    expected_cells = [(mode, budget, level, task, opt_seed)
                      for mode in modes for budget in _BUDGETS for level in _LEVELS
                      for task in _TASKS for opt_seed in _SEEDS]
    if set(indexed) != set(expected_cells):
        raise ValueError("primary coverage has missing or unexpected cells")
    for cell, keyed in indexed.items():
        if len(keyed) != expected_per_task:
            raise ValueError("missing or extra questions in a task and seed cell")

    canonical = {
        key: row for (mode, budget, level, _, _), values in indexed.items()
        if (mode, budget, level) == ("learned", _BUDGETS[0], _LEVELS[0])
        for key, row in values.items()
    }
    reference = {(key[0], key[2]): (row["story"], row["answer"]) for key, row in canonical.items()}
    for cell, keyed in indexed.items():
        task_ids = {identity for identity in reference if identity[0] == cell[3]}
        if {(key[0], key[2]) for key in keyed} != task_ids:
            raise ValueError("primary methods do not share question identities")
        for key, row in keyed.items():
            identity = (key[0], key[2])
            if reference[identity] != (row["story"], row["answer"]):
                raise ValueError("primary methods do not share question story and target identities")

    summaries = []
    for mode in modes:
        for budget in _BUDGETS:
            for level in _LEVELS:
                tasks = {}
                cell_rows = {key: row for (m, b, l, task, _), values in indexed.items()
                             if (m, b, l) == (mode, budget, level) for key, row in values.items()}
                for task in _TASKS:
                    rows_for_task = {key: row for key, row in cell_rows.items() if key[0] == task}
                    metrics = {metric: _stats(_metric_rows(rows_for_task, metric))
                               for metric in ("correct", "normalized_correct")}
                    tasks[task] = {"metrics": metrics, "payload_bytes": _payload_stats(rows_for_task)}
                macro = {}
                for metric in ("correct", "normalized_correct"):
                    seed_macro = {str(opt_seed): float(np.mean([
                        sum(bool(row[metric]) for key, row in cell_rows.items()
                            if key[0] == task and key[1] == opt_seed) / expected_per_task
                        for task in _TASKS])) for opt_seed in _SEEDS}
                    macro[metric] = _stats(seed_macro)
                summaries.append({
                    "mode": mode, "budget": budget, "level": level, "tasks": tasks,
                    "correct_accuracy": macro["correct"],
                    "normalized_accuracy": macro["normalized_correct"],
                    "payload_bytes": _payload_stats(cell_rows),
                })

    comparisons = []
    for mode in modes:
        if mode == "learned":
            continue
        for budget in _BUDGETS:
            for level in _LEVELS:
                left = {key: row for (m, b, l, _, _), values in indexed.items()
                        if (m, b, l) == ("learned", budget, level) for key, row in values.items()}
                right = {key: row for (m, b, l, _, _), values in indexed.items()
                         if (m, b, l) == (mode, budget, level) for key, row in values.items()}
                comparisons.append({
                    "reference": "learned", "condition": mode, "budget": budget, "level": level,
                    "direction": "learned_minus_explicit",
                    "metrics": {metric: _comparison_metric(left, right, metric,
                                                            bootstrap_samples=bootstrap_samples,
                                                            seed=seed)
                                for metric in ("correct", "normalized_correct")},
                })
    return {
        "schema": "storage_frontier_analysis_v1",
        "design": {"tasks": list(_TASKS), "modes": list(modes), "budgets": list(_BUDGETS),
                    "levels": list(_LEVELS), "seeds": list(_SEEDS),
                    "expected_per_task": expected_per_task},
        "summaries": summaries, "comparisons": comparisons,
        "controls_excluded": sorted({row["mode"] for row in rows if row["mode"] in _CONTROLS}),
    }
