#!/usr/bin/env python
"""Render the README figures from the compact result tables in results/.

The CSV files are extracted from sealed study bundles; this script only draws
them. Run from the repository root:

    uv run --no-sync python scripts/make_figures.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"

COLOR = {
    "blue": "#1f77b4", "orange": "#e07b39", "green": "#2a9d8f",
    "red": "#c1443c", "grey": "#8c8c8c", "dark": "#333333",
}
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False,
})


def read(name: str) -> list[dict[str, str]]:
    with open(RESULTS / name, newline="") as handle:
        return list(csv.DictReader(handle))


def save(figure: plt.Figure, name: str) -> None:
    figure.tight_layout()
    figure.savefig(RESULTS / name, bbox_inches="tight")
    plt.close(figure)
    print("wrote", RESULTS / name)


def phase_one() -> None:
    rows = read("phase1.csv")
    seeds = sorted({row["seed"] for row in rows})
    arms = (("uniform", "Uniform", "o"), ("correction_weighted", "Correction weighted", "s"))
    panels = (
        ("Balanced refresh (all four facts rewritten)",
         (("balanced_probe_change_pp", "Probe", COLOR["blue"]),)),
        ("Single-fact repetition (unmentioned facts)",
         (("unspoken_probe_change_pp", "Probe", COLOR["blue"]), ("unspoken_lm_change_pp", "LM answer", COLOR["orange"]))),
    )
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for axis, (title, series) in zip(axes, panels, strict=True):
        axis.axhline(0, color=COLOR["dark"], linewidth=0.8)
        for column, label, color in series:
            for arm, arm_label, marker in arms:
                values = [float(next(r[column] for r in rows if r["seed"] == s and r["arm"] == arm)) for s in seeds]
                offset = -0.08 if arm == "uniform" else 0.08
                axis.plot([i + offset for i in range(len(seeds))], values, marker, color=color, markersize=7,
                          markerfacecolor=color if arm == "uniform" else "white", markeredgewidth=1.4,
                          label=f"{label}, {arm_label.lower()}")
        axis.set_xticks(range(len(seeds)), seeds)
        axis.set_xlabel("Writer seed")
        axis.set_title(title, fontsize=10)
    axes[0].set_ylabel("Change in recall (percentage points)")
    axes[1].legend(loc="upper right", fontsize=8)
    save(figure, "phase1.png")


def writer_collapse() -> None:
    rows = read("writer_collapse.csv")
    seeds = sorted({row["writer_seed"] for row in rows})
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for axis, metric in zip(axes, ("correction", "retention"), strict=True):
        control = [100 * float(next(r["accuracy"] for r in rows if r["writer_seed"] == s and r["arm"] == "control" and r["metric"] == metric)) for s in seeds]
        normalized = [100 * float(next(r["accuracy"] for r in rows if r["writer_seed"] == s and r["arm"] == "normalized" and r["metric"] == metric)) for s in seeds]
        for x, (a, b) in enumerate(zip(control, normalized, strict=True)):
            axis.plot([x, x], [a, b], color=COLOR["grey"], linewidth=1, zorder=1)
        axis.scatter(range(len(seeds)), control, color=COLOR["red"], s=32, zorder=2, label="Training-only state supervision")
        axis.scatter(range(len(seeds)), normalized, color=COLOR["green"], s=32, zorder=3, label="+ LayerNorm over 64 hidden coordinates")
        axis.axhline(50, color=COLOR["dark"], linewidth=0.8, linestyle="--")
        axis.text(-0.45, 51.5, "chance", fontsize=8, ha="left", color=COLOR["dark"])
        axis.set_xticks(range(len(seeds)), [s[-2:] for s in seeds], fontsize=8)
        axis.set_xlabel("Writer seed (41xx)")
        axis.set_title(f"{metric.capitalize()} accuracy", fontsize=10)
        axis.set_ylim(40, 102)
    axes[0].set_ylabel("Accuracy (%), three frozen readers pooled")
    axes[0].legend(loc="lower left", fontsize=8)
    save(figure, "writer_collapse.png")


def qa1_readout() -> None:
    rows = read("qa1_readout.csv")
    groups = (
        ("state_supervised", "State supervised\n(official test)", COLOR["green"]),
        ("fixed_supervised_keys", "Frozen learned keys,\nanswer-trained values", COLOR["blue"]),
        ("fixed_answer_only_keys", "Frozen answer-only keys,\nanswer-trained values", COLOR["orange"]),
        ("answer_only", "Answer only\n(development)", COLOR["red"]),
    )
    figure, axis = plt.subplots(figsize=(9, 3.8))
    oracle = [100 * float(r["oracle_accuracy"]) for r in rows]
    zero = [100 * float(r["zero_memory_accuracy"]) for r in rows]
    axis.axhspan(min(oracle), max(oracle), color=COLOR["green"], alpha=0.15, linewidth=0)
    axis.text(3.45, (min(oracle) + max(oracle)) / 2, "exact oracle state", fontsize=8, va="center", ha="right")
    axis.axhspan(min(zero), max(zero), color=COLOR["grey"], alpha=0.25, linewidth=0)
    axis.text(3.45, min(zero) - 4.5, "zero-memory control", fontsize=8, va="center", ha="right")
    for x, (key, label, color) in enumerate(groups):
        values = [100 * float(r["learned_accuracy"]) for r in rows if r["condition"] == key]
        jitter = [x + (i - (len(values) - 1) / 2) * 0.035 for i in range(len(values))]
        axis.scatter(jitter, values, color=color, s=30, zorder=3, edgecolor="white", linewidth=0.5)
        mean = sum(values) / len(values)
        axis.hlines(mean, x - 0.22, x + 0.22, color=COLOR["dark"], linewidth=1.6, zorder=4)
        axis.text(x + 0.25, mean, f"{mean:.1f}", fontsize=8, va="center")
    axis.set_xticks(range(len(groups)), [g[1] for g in groups], fontsize=8.5)
    axis.set_xlim(-0.5, 3.5)
    axis.set_ylim(0, 104)
    axis.set_ylabel("bAbI QA1 exact-match accuracy (%)")
    axis.set_title("258-byte delta state, frozen Qwen3-1.7B reader; one point per writer-reader cell, bar is the mean", fontsize=9)
    save(figure, "qa1_readout.png")


def storage_frontier() -> None:
    rows = read("storage_frontier.csv")
    methods = (
        ("learned", "Learned int8 slots", COLOR["blue"], "o"),
        ("compressed_recent", "Compressed recent text", COLOR["orange"], "s"),
        ("compressed_diverse", "Compressed diverse text", COLOR["green"], "^"),
        ("dictionary_recent", "Dictionary recent text", COLOR["red"], "D"),
    )
    budgets = sorted({int(r["budget_bytes"]) for r in rows})
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for axis, level, title in zip(axes, ("0", "2"), ("Clean official tasks 1-5", "With heavy text distractors"), strict=True):
        for key, label, color, marker in methods:
            by_budget = defaultdict(list)
            for r in rows:
                if r["method"] == key and r["noise_level"] == level:
                    by_budget[int(r["budget_bytes"])].append(100 * float(r["five_task_accuracy"]))
            mean = [sum(by_budget[b]) / len(by_budget[b]) for b in budgets]
            low = [mean[i] - min(by_budget[b]) for i, b in enumerate(budgets)]
            high = [max(by_budget[b]) - mean[i] for i, b in enumerate(budgets)]
            axis.errorbar(budgets, mean, yerr=[low, high], color=color, marker=marker, markersize=6,
                          linewidth=1.4, capsize=3, label=label)
        axis.set_xscale("log", base=2)
        axis.set_xticks(budgets, [f"{b:,}" for b in budgets])
        axis.minorticks_off()
        axis.set_xlabel("Retained payload bytes")
        axis.set_title(title, fontsize=10)
        axis.set_ylim(0, 104)
    axes[0].set_ylabel("Five-task exact-match accuracy (%)")
    axes[0].legend(loc="center right", fontsize=8)
    save(figure, "storage_frontier.png")


if __name__ == "__main__":
    phase_one()
    writer_collapse()
    qa1_readout()
    storage_frontier()
