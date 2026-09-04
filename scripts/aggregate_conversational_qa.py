#!/usr/bin/env python3
"""Aggregate byte-level conversational QA runs into tables and figures.

The script joins fine-tune runs, controlled holdouts, memory ablations, and
delay sweeps by checkpoint path, groups them by memory configuration and seed,
and writes a summary plus research figures.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from tinymem.data.babi import load_babi_file
from tinymem.data.correction_deletion import generate_update_examples
from tinymem.data.schema import ReasoningExample
from tinymem.evaluation.conversational_evidence import locate_evidence


TASK_ORDER = (
    "babi:qa1",
    "babi:qa2",
    "babi:qa3",
    "babi:qa4",
    "babi:qa5",
    "tinymem_updates:correction_deletion",
)
TASK_LABELS = {
    "babi:qa1": "qa1",
    "babi:qa2": "qa2",
    "babi:qa3": "qa3",
    "babi:qa4": "qa4",
    "babi:qa5": "qa5",
    "tinymem_updates:correction_deletion": "updates",
    "overall": "overall",
}
CONFIG_LABELS = {
    "mean/fifo": "mean-pool FIFO",
    "multislot_attention/gated": "multislot attention + gated",
    "mean/fifo/curriculum": "mean-pool FIFO, delayed curriculum",
    "multislot_attention/gated/curriculum": "multislot + gated, delayed curriculum",
}
CONFIG_COLORS = {
    "mean/fifo": "#2a78d6",
    "multislot_attention/gated": "#eb6834",
    "mean/fifo/curriculum": "#184f95",
    "multislot_attention/gated/curriculum": "#b0400f",
}
FAMILY_OF = {
    "mean/fifo": "mean/fifo",
    "mean/fifo/curriculum": "mean/fifo",
    "multislot_attention/gated": "multislot_attention/gated",
    "multislot_attention/gated/curriculum": "multislot_attention/gated",
}
ORDINAL_BLUES = ("#86b6ef", "#3987e5", "#184f95")
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e6e5e1"
DE_EMPHASIS = "#b8b7b1"


@dataclass(frozen=True)
class FineTuneRun:
    config: str
    seed: int
    checkpoint: str
    results_path: Path
    validation_exact: float
    answer_losses: tuple[float, ...]


@dataclass(frozen=True)
class HoldoutRun:
    config: str
    seed: int
    condition: str
    results_path: Path
    by_task: dict[str, float]
    predictions: tuple[dict[str, object], ...]
    selected_window: int


@dataclass(frozen=True)
class DelayRun:
    config: str
    seed: int
    results_path: Path
    curve: dict[str, dict[int, float]]
    in_window_fraction: dict[int, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--finetune-roots",
        type=Path,
        nargs="+",
        default=(
            Path("artifacts/predictions/conversational_qa_budget"),
            Path("artifacts/predictions/conversational_qa_multiseed"),
            Path("artifacts/predictions/conversational_qa_delayed_curriculum"),
        ),
    )
    parser.add_argument(
        "--holdout-root",
        type=Path,
        default=Path("artifacts/predictions/conversational_qa_holdout"),
    )
    parser.add_argument(
        "--delay-root",
        type=Path,
        default=Path("artifacts/predictions/conversational_qa_delay"),
    )
    parser.add_argument(
        "--diagnosis-manifest",
        type=Path,
        help="JSON list of {label, holdout_results, segment_length} entries",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/figures/conversational_qa"),
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _config_key(document: dict[str, object]) -> str:
    memory = document["memory"]
    key = f"{memory['compressor']}/{memory['memory_update']}"
    delay = document.get("training_delay") or {}
    if int(delay.get("max_bytes", 0) or 0) > 0:
        key += "/curriculum"
    return key


def discover_finetunes(roots: tuple[Path, ...]) -> list[FineTuneRun]:
    runs = []
    checkpoints: dict[str, Path] = {}
    experiment_seeds: dict[tuple[str, int], Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("results.json")):
            document = _load_json(path)
            if document.get("status") != "development_single_seed_conversational_qa":
                continue
            if "memory" not in document:
                continue
            config = _config_key(document)
            seed = int(document["seed"])
            checkpoint = str(document["checkpoint"])
            if checkpoint in checkpoints:
                raise ValueError(
                    "duplicate fine-tune checkpoint in report inputs: "
                    f"{checkpoints[checkpoint]} and {path}"
                )
            identity = (config, seed)
            if identity in experiment_seeds:
                raise ValueError(
                    "duplicate fine-tune config and seed in report inputs: "
                    f"{experiment_seeds[identity]} and {path}"
                )
            checkpoints[checkpoint] = path
            experiment_seeds[identity] = path
            runs.append(
                FineTuneRun(
                    config=config,
                    seed=seed,
                    checkpoint=checkpoint,
                    results_path=path,
                    validation_exact=float(
                        document["controlled_after"]["overall"]["exact_accuracy"]
                    ),
                    answer_losses=tuple(
                        document["training_history"]["answer_losses"]
                    ),
                )
            )
    return runs


def _task_accuracies(evaluation: dict[str, object]) -> dict[str, float]:
    by_task = {
        name: float(result["exact_accuracy"])
        for name, result in evaluation["by_task"].items()
    }
    by_task["overall"] = float(evaluation["overall"]["exact_accuracy"])
    return by_task


def discover_holdouts(
    root: Path,
    finetunes: dict[str, FineTuneRun],
) -> list[HoldoutRun]:
    runs = []
    seen: dict[tuple[str, str], Path] = {}
    if not root.exists():
        return runs
    for path in sorted(root.rglob("results.json")):
        document = _load_json(path)
        finetune = finetunes.get(str(document.get("checkpoint")))
        if finetune is None:
            continue
        condition = str(document.get("memory_condition", "normal"))
        identity = (finetune.checkpoint, condition)
        if identity in seen:
            raise ValueError(
                "duplicate holdout checkpoint and condition in report inputs: "
                f"{seen[identity]} and {path}"
            )
        seen[identity] = path
        runs.append(
            HoldoutRun(
                config=finetune.config,
                seed=finetune.seed,
                condition=condition,
                results_path=path,
                by_task=_task_accuracies(document["evaluation"]),
                predictions=tuple(document["evaluation"]["predictions"]),
                selected_window=int(document.get("selected_window", 0)),
            )
        )
    return runs


def discover_delays(
    root: Path,
    finetunes: dict[str, FineTuneRun],
) -> list[DelayRun]:
    runs = []
    seen: dict[str, Path] = {}
    if not root.exists():
        return runs
    for path in sorted(root.rglob("results.json")):
        document = _load_json(path)
        finetune = finetunes.get(str(document.get("checkpoint")))
        if finetune is None:
            continue
        if finetune.checkpoint in seen:
            raise ValueError(
                "duplicate delay sweep checkpoint in report inputs: "
                f"{seen[finetune.checkpoint]} and {path}"
            )
        seen[finetune.checkpoint] = path
        curve = {
            condition: {
                int(delay): float(result["overall"]["exact_accuracy"])
                for delay, result in by_delay.items()
            }
            for condition, by_delay in document["curve"].items()
        }
        window = int(document["selected_window"])
        first_condition = next(iter(document["curve"].values()))
        in_window_fraction = {
            int(delay): sum(
                int(prediction["prompt_bytes"]) <= window
                for prediction in result["predictions"]
            )
            / len(result["predictions"])
            for delay, result in first_condition.items()
        }
        runs.append(
            DelayRun(
                config=finetune.config,
                seed=finetune.seed,
                results_path=path,
                curve=curve,
                in_window_fraction=in_window_fraction,
            )
        )
    return runs


def load_holdout_examples(
    repository_root: Path,
    holdout_document: dict[str, object],
) -> dict[str, ReasoningExample]:
    """Rebuild the deterministic holdout examples referenced by a results file."""
    data_root = repository_root / "data/raw/tasks_1-20_v1-2/en-valid-10k"
    examples: dict[str, ReasoningExample] = {}
    for task in holdout_document["tasks"]:
        for example in load_babi_file(
            data_root / f"{task}_test.txt",
            task_id=task,
            split="test",
        ):
            examples[example.source_example_id] = example
    generation = holdout_document["update_generation"]
    for generated in generate_update_examples(
        split=generation["split"],
        count=int(generation["count"]),
        base_seed=int(generation["base_seed"]),
        deletion_rate=float(generation["deletion_rate"]),
        query_delay=int(generation["query_delay"]),
        distractor_count=int(generation["distractor_count"]),
        correction_counts=tuple(generation["correction_counts"]),
    ):
        examples[generated.example.source_example_id] = generated.example
    return examples


def evidence_split(
    predictions: tuple[dict[str, object], ...],
    examples: dict[str, ReasoningExample],
    *,
    segment_length: int,
) -> dict[str, tuple[int, int]]:
    """Return correct and total counts for in-window and beyond-window cases."""
    counts = {"in_window": [0, 0], "beyond_window": [0, 0]}
    for prediction in predictions:
        example = examples[str(prediction["source_example_id"])]
        placement = locate_evidence(example)
        if placement.prompt_bytes != int(prediction["prompt_bytes"]):
            raise ValueError("prompt bytes do not match the rebuilt example")
        bucket = (
            "in_window"
            if placement.in_final_segment(segment_length)
            else "beyond_window"
        )
        counts[bucket][0] += int(bool(prediction["exact_match"]))
        counts[bucket][1] += 1
    return {name: (value[0], value[1]) for name, value in counts.items()}


def majority_class_rates(
    predictions: tuple[dict[str, object], ...],
) -> dict[str, float]:
    """Return the majority-answer rate per task as a chance reference."""
    references: dict[str, list[str]] = defaultdict(list)
    for prediction in predictions:
        key = f"{prediction['dataset']}:{prediction['task_id']}"
        references[key].append(str(prediction["reference"]).casefold())
    rates = {}
    total_majority = 0
    for key, values in references.items():
        counts = defaultdict(int)
        for value in values:
            counts[value] += 1
        majority = max(counts.values())
        rates[key] = majority / len(values)
        total_majority += majority
    rates["overall"] = total_majority / len(predictions)
    return rates


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Return a 95% Wilson score interval."""
    if total == 0:
        return (0.0, 0.0)
    z = 1.96
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return (center - half_width, center + half_width)


def _style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color(GRID)
    axis.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    axis.yaxis.grid(True, color=GRID, linewidth=1.0)
    axis.set_axisbelow(True)


def _percent_axis(axis: plt.Axes, top: float = 1.0) -> None:
    axis.set_ylim(0, top)
    axis.set_yticks([tick / 100 for tick in range(0, int(top * 100) + 1, 20)])
    axis.set_yticklabels(
        [f"{tick}%" for tick in range(0, int(top * 100) + 1, 20)],
        color=TEXT_SECONDARY,
    )


def _finish(figure: plt.Figure, destination: Path, title: str, subtitle: str) -> None:
    figure.patch.set_facecolor(SURFACE)
    height = figure.get_size_inches()[1]
    figure.suptitle(
        title,
        x=0.01,
        y=1 - 0.12 / height,
        ha="left",
        va="top",
        fontsize=13,
        color=TEXT_PRIMARY,
    )
    figure.text(
        0.01,
        1 - 0.42 / height,
        subtitle,
        ha="left",
        va="top",
        fontsize=9.5,
        color=TEXT_SECONDARY,
    )
    figure.tight_layout(rect=(0, 0, 1, 1 - 0.62 / height))
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination.with_suffix(".png"), dpi=200, facecolor=SURFACE)
    figure.savefig(destination.with_suffix(".svg"), facecolor=SURFACE)
    plt.close(figure)


def plot_diagnosis(
    entries: list[dict[str, object]],
    chance: dict[str, float],
    destination: Path,
) -> None:
    tasks = list(TASK_ORDER) + ["overall"]
    figure, axis = plt.subplots(figsize=(11, 4.6))
    _style_axis(axis)
    width = 0.8 / len(entries)
    for index, entry in enumerate(entries):
        values = [entry["by_task"][task] for task in tasks]
        counts = [entry["counts"][task] for task in tasks]
        positions = [
            position + (index - (len(entries) - 1) / 2) * width
            for position in range(len(tasks))
        ]
        lower = []
        upper = []
        for value, (correct, total) in zip(values, counts, strict=True):
            low, high = wilson_interval(correct, total)
            lower.append(value - low)
            upper.append(high - value)
        axis.bar(
            positions,
            values,
            width=width * 0.9,
            color=entry["color"],
            label=entry["label"],
            yerr=[lower, upper],
            error_kw={"elinewidth": 0.8, "ecolor": TEXT_SECONDARY, "capsize": 0},
        )
        for position, value in zip(positions, values, strict=True):
            axis.text(
                position,
                value + 0.03,
                f"{100 * value:.0f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=TEXT_SECONDARY,
            )
    for position, task in enumerate(tasks):
        axis.hlines(
            chance[task],
            position - 0.42,
            position + 0.42,
            color=TEXT_SECONDARY,
            linewidth=1.0,
        )
    axis.hlines([], [], [], color=TEXT_SECONDARY, linewidth=1.0, label="majority answer")
    axis.set_xticks(range(len(tasks)))
    axis.set_xticklabels([TASK_LABELS[task] for task in tasks], color=TEXT_PRIMARY)
    _percent_axis(axis, 0.8)
    axis.set_ylabel("exact match", color=TEXT_SECONDARY)
    axis.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)
    _finish(
        figure,
        destination,
        "Controlled test evaluation: window and fine-tune budget lift the byte model off the class prior",
        "6,000 bAbI qa1-qa5 and update test examples, seed 1337, 95% Wilson intervals; "
        "short marks show the majority-answer rate",
    )


def plot_evidence_window(
    entries: list[dict[str, object]],
    destination: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 4.2))
    _style_axis(axis)
    buckets = (("in_window", "evidence inside the final segment", "#2a78d6"),
               ("beyond_window", "evidence beyond the final segment", "#eb6834"))
    width = 0.36
    for offset, (bucket, label, color) in zip((-0.5, 0.5), buckets, strict=True):
        positions = [index + offset * width for index in range(len(entries))]
        values = []
        errors = ([], [])
        for entry in entries:
            correct, total = entry["evidence"][bucket]
            value = correct / total if total else float("nan")
            values.append(value)
            low, high = wilson_interval(correct, total)
            errors[0].append(value - low if total else 0)
            errors[1].append(high - value if total else 0)
        axis.bar(
            positions,
            values,
            width=width * 0.9,
            color=color,
            label=label,
            yerr=errors,
            error_kw={"elinewidth": 0.8, "ecolor": TEXT_SECONDARY, "capsize": 0},
        )
        for position, value, entry in zip(positions, values, entries, strict=True):
            total = entry["evidence"][bucket][1]
            if total == 0:
                axis.text(position, 0.01, "n=0", ha="center", va="bottom", fontsize=7.5, color=TEXT_SECONDARY)
                continue
            axis.text(
                position,
                value + 0.03,
                f"{100 * value:.0f}\nn={total:,}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=TEXT_SECONDARY,
            )
    axis.set_xticks(range(len(entries)))
    axis.set_xticklabels(
        [entry["label"].replace(", ", ",\n", 1) for entry in entries],
        color=TEXT_PRIMARY,
        fontsize=8.5,
    )
    _percent_axis(axis, 0.7)
    axis.set_ylabel("exact match", color=TEXT_SECONDARY)
    axis.legend(frameon=False, fontsize=8.5, loc="upper left")
    _finish(
        figure,
        destination,
        "Gains come from evidence inside the local segment, not from memory",
        "Holdout examples split by whether every supporting fact lies in the answer's local segment",
    )


def plot_multiseed(
    stats: dict[str, dict[str, dict[str, object]]],
    destination: Path,
) -> None:
    tasks = list(TASK_ORDER) + ["overall"]
    configs = [
        config
        for config in CONFIG_LABELS
        if config in stats and not config.endswith("/curriculum")
    ]
    figure, axis = plt.subplots(figsize=(11, 4.6))
    width = 0.8 / max(len(configs), 1)
    _style_axis(axis)
    width = 0.8 / max(len(configs), 1)
    for index, config in enumerate(configs):
        positions = [
            position + (index - (len(configs) - 1) / 2) * width
            for position in range(len(tasks))
        ]
        means = [stats[config][task]["mean"] for task in tasks]
        stds = [stats[config][task]["std"] for task in tasks]
        axis.bar(
            positions,
            means,
            width=width * 0.9,
            color=CONFIG_COLORS[config],
            label=f"{CONFIG_LABELS[config]} (n={stats[config]['overall']['n']} seeds)",
            yerr=stds,
            error_kw={"elinewidth": 0.8, "ecolor": TEXT_SECONDARY, "capsize": 0},
        )
        for position, task in zip(positions, tasks, strict=True):
            for value in stats[config][task]["values"]:
                axis.plot(
                    position + width * 0.42,
                    value,
                    marker="o",
                    markersize=4.5,
                    markerfacecolor=CONFIG_COLORS[config],
                    markeredgecolor=SURFACE,
                    markeredgewidth=1.2,
                    linestyle="none",
                    clip_on=False,
                )
        for position, task, mean, std in zip(positions, tasks, means, stds, strict=True):
            top = max([mean + std, *stats[config][task]["values"]])
            axis.text(
                position,
                top + 0.02,
                f"{100 * mean:.0f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=TEXT_SECONDARY,
            )
    axis.set_xticks(range(len(tasks)))
    axis.set_xticklabels([TASK_LABELS[task] for task in tasks], color=TEXT_PRIMARY)
    _percent_axis(axis, 0.8)
    axis.set_ylabel("exact match", color=TEXT_SECONDARY)
    axis.legend(frameon=False, fontsize=8.5, loc="upper left")
    _finish(
        figure,
        destination,
        "Memory family comparison across seeds on the controlled holdout",
        "Bars show the seed mean, whiskers one standard deviation, dots individual seeds; "
        "512-byte window, 4,000 steps x batch 16, lr 1e-3",
    )


def plot_ablation(
    ablation: dict[str, dict[str, dict[str, dict[str, object]]]],
    destination: Path,
) -> None:
    configs = [
        config
        for config in CONFIG_LABELS
        if config in ablation and not config.endswith("/curriculum")
    ]
    tasks = list(TASK_ORDER) + ["overall"]
    figure, axes = plt.subplots(1, len(configs), figsize=(5.5 * len(configs), 4.2), sharey=True)
    if len(configs) == 1:
        axes = [axes]
    for axis, config in zip(axes, configs, strict=True):
        _style_axis(axis)
        width = 0.38
        for offset, condition, color, label in (
            (-0.5, "normal", CONFIG_COLORS[config], "normal memory"),
            (0.5, "drop_at_query", DE_EMPHASIS, "memory dropped at query"),
        ):
            if condition not in ablation[config]:
                continue
            positions = [index + offset * width for index in range(len(tasks))]
            means = [ablation[config][condition][task]["mean"] for task in tasks]
            stds = [ablation[config][condition][task]["std"] for task in tasks]
            axis.bar(
                positions,
                means,
                width=width * 0.9,
                color=color,
                label=label,
                yerr=stds,
                error_kw={"elinewidth": 0.8, "ecolor": TEXT_SECONDARY, "capsize": 0},
            )
            for position, mean, std in zip(positions, means, stds, strict=True):
                axis.text(position, mean + std + 0.015, f"{100 * mean:.0f}", ha="center", va="bottom", fontsize=7, color=TEXT_SECONDARY)
        axis.set_title(CONFIG_LABELS[config], fontsize=10, color=TEXT_PRIMARY, loc="left")
        axis.set_xticks(range(len(tasks)))
        axis.set_xticklabels([TASK_LABELS[task] for task in tasks], color=TEXT_PRIMARY, fontsize=8.5)
        _percent_axis(axis, 0.8)
        axis.legend(frameon=False, fontsize=8.5, loc="upper left")
    axes[0].set_ylabel("exact match", color=TEXT_SECONDARY)
    _finish(
        figure,
        destination,
        "Causal memory test: hiding memory at the query barely changes controlled accuracy",
        "Same holdout, same checkpoints; whiskers show one standard deviation across seeds where more than one seed exists",
    )


def plot_delay(
    delays: dict[str, dict[str, dict[int, dict[str, object]]]],
    in_window: dict[str, dict[int, float]],
    chance: float,
    destination: Path,
) -> None:
    families = [
        family
        for family in ("mean/fifo", "multislot_attention/gated")
        if any(FAMILY_OF[config] == family for config in delays)
    ]
    regimes = [
        ("standard fine-tune", ""),
        ("delayed-recall curriculum", "/curriculum"),
    ]
    regimes = [
        regime
        for regime in regimes
        if any(f"{family}{regime[1]}" in delays for family in families)
    ]
    figure, axes = plt.subplots(
        len(regimes),
        len(families),
        figsize=(max(11.0, 5.5 * len(families)), 3.9 * len(regimes)),
        sharey=True,
        squeeze=False,
    )
    for row, (regime_label, suffix) in enumerate(regimes):
        for column, family in enumerate(families):
            axis = axes[row][column]
            _style_axis(axis)
            config = f"{family}{suffix}"
            axis.set_title(
                f"{CONFIG_LABELS[family]}, {regime_label}",
                fontsize=9.5,
                color=TEXT_PRIMARY,
                loc="left",
            )
            if config not in delays:
                axis.text(0.5, 0.5, "pending", transform=axis.transAxes, ha="center", color=TEXT_SECONDARY)
                continue
            fractions = in_window.get(config, {})
            for condition, color, label, width in (
                ("drop_at_query", DE_EMPHASIS, "memory dropped at query", 3.2),
                ("normal", CONFIG_COLORS[config], "normal memory", 1.8),
            ):
                if condition not in delays[config]:
                    continue
                points = sorted(delays[config][condition].items())
                x = list(range(len(points)))
                means = [stat["mean"] for _, stat in points]
                stds = [stat["std"] for _, stat in points]
                axis.plot(x, means, color=color, linewidth=width, solid_capstyle="round", label=label)
                axis.fill_between(
                    x,
                    [m - s for m, s in zip(means, stds, strict=True)],
                    [m + s for m, s in zip(means, stds, strict=True)],
                    color=color,
                    alpha=0.1,
                    linewidth=0,
                )
                axis.plot(x, means, marker="o", markersize=6, linestyle="none", markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.5)
                axis.set_xticks(x)
                axis.set_xticklabels(
                    [
                        f"{delay:,}\n{100 * fractions[delay]:.0f}% fit"
                        if delay in fractions
                        else f"{delay:,}"
                        for delay, _ in points
                    ],
                    color=TEXT_PRIMARY,
                    fontsize=8,
                )
            axis.axhline(chance, color=TEXT_SECONDARY, linewidth=1.0)
            axis.text(0.05, chance + 0.012, "majority answer", fontsize=7.5, color=TEXT_SECONDARY)
            if row == len(regimes) - 1:
                axis.set_xlabel(
                    "WikiText filler bytes before the question\n"
                    "(second line: share of prompts that fit one 512-byte segment)",
                    color=TEXT_SECONDARY,
                    fontsize=8.5,
                )
            _percent_axis(axis, 0.8)
            axis.legend(frameon=False, fontsize=8, loc="upper right")
        axes[row][0].set_ylabel("exact match on qa1", color=TEXT_SECONDARY)
    _finish(
        figure,
        destination,
        "Delayed recall: accuracy versus filler inserted before the question",
        "500 untouched qa1 test examples per delay, filler from the WikiText-2 validation split; "
        "band shows one standard deviation across seeds",
    )


def plot_training(finetunes: list[FineTuneRun], destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 4.2))
    _style_axis(axis)
    window = 100
    labelled: set[str] = set()
    for run in sorted(finetunes, key=lambda item: (item.config, item.seed)):
        losses = run.answer_losses
        smoothed = [
            statistics.fmean(losses[max(0, index - window) : index + 1])
            for index in range(0, len(losses), 25)
        ]
        steps = list(range(0, len(losses), 25))
        axis.plot(
            steps,
            smoothed,
            color=CONFIG_COLORS.get(run.config, DE_EMPHASIS),
            linewidth=1.6,
            alpha=0.85,
            label=CONFIG_LABELS.get(run.config, run.config) if run.config not in labelled else None,
        )
        labelled.add(run.config)
    axis.set_xlabel("fine-tune step", color=TEXT_SECONDARY)
    axis.set_ylabel("answer bytes cross-entropy (nats, 100-step mean)", color=TEXT_SECONDARY, fontsize=9)
    axis.set_ylim(0, 0.6)
    axis.legend(frameon=False, fontsize=8.5)
    _finish(
        figure,
        destination,
        "Answer loss keeps falling through 4,000 steps for every seed",
        "One line per seed; the earlier 1,000-step schedule stopped near 0.29 nats per byte",
    )


def summarize(values: list[float]) -> dict[str, object]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else float("nan"),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    finetunes = discover_finetunes(tuple(args.finetune_roots))
    by_checkpoint = {run.checkpoint: run for run in finetunes}
    holdouts = discover_holdouts(args.holdout_root, by_checkpoint)
    delays = discover_delays(args.delay_root, by_checkpoint)
    output_root = repository_root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    example_cache: dict[str, dict[str, ReasoningExample]] = {}

    def examples_for(document_path: Path) -> dict[str, ReasoningExample]:
        document = _load_json(document_path)
        key = json.dumps(document["update_generation"], sort_keys=True) + json.dumps(document["tasks"])
        if key not in example_cache:
            example_cache[key] = load_holdout_examples(repository_root, document)
        return example_cache[key]

    summary: dict[str, object] = {"finetunes": [], "holdouts": [], "delays": []}

    # Diagnosis figure from an explicit manifest.
    diagnosis_entries = []
    chance: dict[str, float] = {}
    if args.diagnosis_manifest is not None:
        manifest = json.loads(args.diagnosis_manifest.read_text(encoding="utf-8"))
        for index, entry in enumerate(manifest):
            document = _load_json(Path(entry["holdout_results"]))
            evaluation = document["evaluation"]
            predictions = tuple(evaluation["predictions"])
            counts = {
                name: (
                    round(result["exact_accuracy"] * result["count"]),
                    result["count"],
                )
                for name, result in evaluation["by_task"].items()
            }
            counts["overall"] = (
                round(evaluation["overall"]["exact_accuracy"] * evaluation["overall"]["count"]),
                evaluation["overall"]["count"],
            )
            colors = list(ORDINAL_BLUES) + [CONFIG_COLORS["multislot_attention/gated"]]
            color = entry.get("color") or colors[min(index, len(colors) - 1)]
            diagnosis_entries.append(
                {
                    "label": entry["label"],
                    "by_task": _task_accuracies(evaluation),
                    "counts": counts,
                    "color": color,
                    "evidence": evidence_split(
                        predictions,
                        examples_for(Path(entry["holdout_results"])),
                        segment_length=int(entry["segment_length"]),
                    ),
                }
            )
            if not chance:
                chance = majority_class_rates(predictions)
        plot_diagnosis(diagnosis_entries, chance, output_root / "fig1_holdout_diagnosis")
        plot_evidence_window(diagnosis_entries, output_root / "fig2_evidence_window")
        summary["diagnosis"] = [
            {key: value for key, value in entry.items() if key != "color"}
            for entry in diagnosis_entries
        ]
        summary["majority_answer_rate"] = chance

    # Multi-seed statistics on the controlled test evaluations.
    tasks = list(TASK_ORDER) + ["overall"]
    grouped: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for holdout in holdouts:
        for task in tasks:
            grouped[holdout.config][holdout.condition][task].append(holdout.by_task[task])
        summary["holdouts"].append(
            {
                "config": holdout.config,
                "seed": holdout.seed,
                "condition": holdout.condition,
                "results": str(holdout.results_path),
                "by_task": holdout.by_task,
            }
        )
    multiseed_stats = {
        config: {task: summarize(values) for task, values in conditions["normal"].items()}
        for config, conditions in grouped.items()
        if "normal" in conditions
    }
    if multiseed_stats:
        plot_multiseed(multiseed_stats, output_root / "fig3_multiseed_holdout")
    ablation_stats = {
        config: {
            condition: {task: summarize(values) for task, values in by_task.items()}
            for condition, by_task in conditions.items()
        }
        for config, conditions in grouped.items()
        if "drop_at_query" in conditions
    }
    if ablation_stats:
        plot_ablation(ablation_stats, output_root / "fig4_memory_ablation")
    summary["multiseed"] = multiseed_stats
    summary["ablation"] = ablation_stats

    # Delay sweeps.
    delay_grouped: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    in_window_grouped: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for run in delays:
        for condition, by_delay in run.curve.items():
            for delay, value in by_delay.items():
                delay_grouped[run.config][condition][delay].append(value)
        for delay, fraction in run.in_window_fraction.items():
            in_window_grouped[run.config][delay].append(fraction)
        summary["delays"].append(
            {
                "config": run.config,
                "seed": run.seed,
                "results": str(run.results_path),
                "curve": {condition: {str(k): v for k, v in by_delay.items()} for condition, by_delay in run.curve.items()},
            }
        )
    delay_stats = {
        config: {
            condition: {delay: summarize(values) for delay, values in by_delay.items()}
            for condition, by_delay in conditions.items()
        }
        for config, conditions in delay_grouped.items()
    }
    if delay_stats:
        plot_delay(
            delay_stats,
            {
                config: {delay: statistics.fmean(values) for delay, values in by_delay.items()}
                for config, by_delay in in_window_grouped.items()
            },
            chance.get("babi:qa1", 1 / 6) if chance else 1 / 6,
            output_root / "fig5_delay_curve",
        )
    summary["delay"] = {
        config: {condition: {str(k): v for k, v in by_delay.items()} for condition, by_delay in conditions.items()}
        for config, conditions in delay_stats.items()
    }

    if finetunes:
        plot_training(finetunes, output_root / "fig6_training_curves")
    summary["finetunes"] = [
        {
            "config": run.config,
            "seed": run.seed,
            "checkpoint": run.checkpoint,
            "validation_exact": run.validation_exact,
            "final_answer_loss": run.answer_losses[-1],
        }
        for run in finetunes
    ]

    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = ["# conversational QA report", ""]
    if diagnosis_entries:
        lines += ["## controlled test diagnosis (seed 1337)", "", "| run | " + " | ".join(TASK_LABELS[t] for t in tasks) + " | in-window | beyond |", "|" + " --- |" * (len(tasks) + 3)]
        for entry in diagnosis_entries:
            evidence = entry["evidence"]
            def rate(bucket: str) -> str:
                correct, total = evidence[bucket]
                return f"{100 * correct / total:.1f} (n={total})" if total else "n/a"
            lines.append(
                f"| {entry['label']} | "
                + " | ".join(f"{100 * entry['by_task'][t]:.1f}" for t in tasks)
                + f" | {rate('in_window')} | {rate('beyond_window')} |"
            )
        lines.append("")
    if multiseed_stats:
        lines += ["## multi-seed controlled test, normal memory (mean ± sd, exact %)", "", "| config | seeds | " + " | ".join(TASK_LABELS[t] for t in tasks) + " |", "|" + " --- |" * (len(tasks) + 2)]
        for config, stats in multiseed_stats.items():
            lines.append(
                f"| {CONFIG_LABELS.get(config, config)} | {stats['overall']['n']} | "
                + " | ".join(f"{100 * stats[t]['mean']:.1f} ± {100 * stats[t]['std']:.1f}" for t in tasks)
                + " |"
            )
        lines.append("")
    if ablation_stats:
        lines += ["## memory ablation (overall exact %, mean over seeds)", "", "| config | normal | drop at query | delta |", "| --- | --- | --- | --- |"]
        for config, conditions in ablation_stats.items():
            normal = conditions["normal"]["overall"]["mean"]
            dropped = conditions["drop_at_query"]["overall"]["mean"]
            lines.append(f"| {CONFIG_LABELS.get(config, config)} | {100 * normal:.1f} | {100 * dropped:.1f} | {100 * (dropped - normal):+.1f} |")
        lines.append("")
    if delay_stats:
        lines += ["## delayed recall on qa1 (exact %, mean over seeds)", ""]
        all_delays = sorted({d for c in delay_stats.values() for cond in c.values() for d in cond})
        lines += ["| config | condition | " + " | ".join(f"{d:,} B" for d in all_delays) + " |", "|" + " --- |" * (len(all_delays) + 2)]
        for config, conditions in delay_stats.items():
            for condition, by_delay in conditions.items():
                lines.append(
                    f"| {CONFIG_LABELS.get(config, config)} | {condition} | "
                    + " | ".join(f"{100 * by_delay[d]['mean']:.1f}" if d in by_delay else "" for d in all_delays)
                    + " |"
                )
        lines.append("")
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"figures: {output_root}")


if __name__ == "__main__":
    main()
