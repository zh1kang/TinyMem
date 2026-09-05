#!/usr/bin/env python3
"""Apply frozen result rules and plot an aggregated opaque-association study."""

import argparse
import hashlib
import json
import math
import statistics
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from tinymem.evaluation.association_study import AssociationStudyDesign, assess_association_study
from tinymem.evaluation.paired_bootstrap import PairedAccuracyInterval


DISPLAY_METHODS = (
    ("query_pool", "normal", "query pool", "#217b68"),
    ("mean_pool", "normal", "trained mean/FIFO", "#4682a3"),
    ("baseline", "recent_native", "recent native IDs", "#8998a2"),
    ("baseline", "recent_vocabulary", "recent vocabulary", "#8998a2"),
    ("baseline", "latest_vocabulary", "latest vocabulary facts", "#8998a2"),
    ("baseline", "latest_template", "latest exact templates", "#8998a2"),
    ("baseline", "fingerprint", "fingerprint (not Qwen)", "#967aa1"),
    ("baseline", "drop", "no memory (0 B)", "#b5b5b5"),
    ("baseline", "full_history", "full history (unbounded)", "#d4b36a"),
)
ANALYSIS_SOURCES = (
    "artifacts/predictions/aggregate_opaque_study.py",
    "src/tinymem/evaluation/paired_bootstrap.py",
    "src/tinymem/evaluation/reader_gate.py",
    "src/tinymem/evaluation/longmemeval.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_analysis_sources(frozen: dict[str, Any], summary: dict[str, Any]) -> None:
    """Match recorded analysis sources after resolving relative-path aliases."""
    def normalized(values: dict[str, str]) -> dict[Path, str]:
        result = {Path(path).resolve(): value for path, value in values.items()}
        if len(result) != len(values):
            raise ValueError("duplicate analysis source path aliases")
        return result

    expected = normalized(frozen["source_sha256"])
    actual = normalized(summary["provenance"]["source_sha256"])
    required = {Path(path).resolve() for path in ANALYSIS_SOURCES}
    if set(actual) != required or any(path not in expected or actual[path] != expected[path] for path in required):
        raise ValueError("summary analysis sources differ from the frozen protocol")


def correct_counts(row: dict[str, Any], *, total: int, seeds: tuple[int, ...]) -> dict[int, int]:
    """Recover exact integer counts from validated per-checkpoint accuracy fractions."""
    values = row["per_seed_accuracy"]
    if (not isinstance(values, dict) or set(values) != {str(seed) for seed in seeds}
            or isinstance(row["queries_per_checkpoint"], bool) or not isinstance(row["queries_per_checkpoint"], int)
            or row["queries_per_checkpoint"] != total):
        raise ValueError("accuracy row must match the declared seeds and query count")
    counts = {}
    for seed in seeds:
        value = values[str(seed)]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("per-seed accuracies must be finite numeric fractions")
        count = round(value * total)
        if abs(value - count / total) > 1e-12:
            raise ValueError("accuracy does not represent an integer number of correct queries")
        counts[seed] = count
    if (isinstance(row["mean_accuracy"], bool) or not isinstance(row["mean_accuracy"], (int, float))
            or not math.isclose(row["mean_accuracy"], statistics.fmean(values.values()), rel_tol=0, abs_tol=1e-12)):
        raise ValueError("mean accuracy disagrees with per-seed values")
    return counts


def build_report(frozen: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    """Use expected metadata from the frozen protocol, never from an interval."""
    if frozen["protocol"] != "opaque_qa1_fixed_byte_comparison_v1" or frozen["stored_bytes"] != 66:
        raise ValueError("unsupported association study protocol")
    design = AssociationStudyDesign(frozen["statistics"]["worlds"], tuple(frozen["seeds"]),
                                    frozen["bootstrap_resamples"], frozen["bootstrap_seed"])
    if len(summary["comparisons"]) != len(frozen["statistical_comparisons"]):
        raise ValueError("comparison list is incomplete")
    known, absent = {}, {}
    gains_by_family = {"known_superiority": {}, "absent_noninferiority": {}}
    for actual, expected in zip(summary["comparisons"], frozen["statistical_comparisons"], strict=True):
        if any(actual[key] != value for key, value in expected.items()):
            raise ValueError("comparison differs from its frozen definition")
        if actual["family"] not in ("known_superiority", "absent_noninferiority"):
            continue
        if actual["left"] != ["query_pool", "opaque:normal"]:
            raise ValueError("primary comparison must evaluate opaque query pooling")
        category = "opaque_qa1_known" if actual["family"] == "known_superiority" else "opaque_qa1_missing"
        if actual["category"] != category:
            raise ValueError("primary comparison uses the wrong answer category")
        method, condition = actual["right"]
        if method not in ("baseline", "mean_pool") or not condition.startswith("opaque:") or (method == "mean_pool" and condition != "opaque:normal"):
            raise ValueError("primary comparator must use the opaque condition")
        name = condition.split(":")[1] if method == "baseline" else method
        evidence = {field.name: actual[field.name] for field in fields(PairedAccuracyInterval)}
        evidence["optimization_seeds"] = tuple(evidence["optimization_seeds"])
        destination = known if actual["family"] == "known_superiority" else absent
        if name in destination:
            raise ValueError("duplicate primary comparator")
        destination[name] = PairedAccuracyInterval(**evidence)
        gains = actual["per_seed_difference"]
        if set(gains) != {str(seed) for seed in design.seeds} or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not -1 <= value <= 1
            for value in gains.values()
        ):
            raise ValueError("paired gains must match the declared seeds and be finite fractions")
        if not math.isclose(statistics.fmean(gains.values()), actual["difference"], rel_tol=0, abs_tol=1e-12):
            raise ValueError("paired seed gains disagree with the reported difference")
        gains_by_family[actual["family"]][name] = gains
    rows = {(row["method"], row["condition"], row["category"]): row for row in summary["accuracy"]}
    if len(rows) != len(summary["accuracy"]):
        raise ValueError("duplicate accuracy row")
    accuracy = []
    counts_by_key = {}
    for method, condition, label, color in DISPLAY_METHODS:
        values = {"method": method, "condition": condition, "label": label, "color": color}
        for category, suffix, total in (("opaque_qa1_known", "known", design.worlds * 8),
                                        ("opaque_qa1_missing", "absent", design.worlds)):
            key = (method, f"opaque:{condition}", category)
            if (key not in rows or isinstance(rows[key]["worlds"], bool)
                    or not isinstance(rows[key]["worlds"], int) or rows[key]["worlds"] != design.worlds):
                raise ValueError("primary accuracy table is incomplete or has the wrong world count")
            counts = correct_counts(rows[key], total=total, seeds=design.seeds)
            if method == "baseline" and len(set(counts.values())) != 1:
                raise ValueError("deterministic reference scores must be identical across seed labels")
            counts_by_key[key] = counts
            values[suffix] = rows[key]["mean_accuracy"]
            values[f"{suffix}_per_seed"] = rows[key]["per_seed_accuracy"]
            values[f"{suffix}_optimization_seed_sd"] = (
                None if method == "baseline" else statistics.stdev(rows[key]["per_seed_accuracy"].values())
            )
        accuracy.append(values)
    for family, gains_by_comparator in gains_by_family.items():
        category = "opaque_qa1_known" if family == "known_superiority" else "opaque_qa1_missing"
        total = design.worlds * (8 if family == "known_superiority" else 1)
        left = counts_by_key[("query_pool", "opaque:normal", category)]
        for comparator, gains in gains_by_comparator.items():
            key = ("mean_pool", "opaque:normal", category) if comparator == "mean_pool" else ("baseline", f"opaque:{comparator}", category)
            right = counts_by_key[key]
            if any(not math.isclose(gains[str(seed)], (left[seed] - right[seed]) / total, rel_tol=0, abs_tol=1e-12) for seed in design.seeds):
                raise ValueError("paired gains disagree with the per-seed accuracy table")
    full_known = counts_by_key[("baseline", "opaque:full_history", "opaque_qa1_known")]
    full_absent = counts_by_key[("baseline", "opaque:full_history", "opaque_qa1_missing")]
    assessment = assess_association_study(
        known, absent, design=design, reader_known_correct=full_known[design.seeds[0]],
        reader_absent_correct=full_absent[design.seeds[0]],
        writer_absent_correct=counts_by_key[("query_pool", "opaque:normal", "opaque_qa1_missing")],
    )
    return {"design": asdict(design), "assessment": asdict(assessment), "accuracy": accuracy,
            "known_contrasts": {name: asdict(value) | {
                "optimization_difference_sd": statistics.stdev(gains_by_family["known_superiority"][name].values())
            } for name, value in known.items()},
            "absent_contrasts": {name: asdict(value) | {
                "optimization_difference_sd": statistics.stdev(gains_by_family["absent_noninferiority"][name].values())
            } for name, value in absent.items()},
            "known_per_seed_gains": gains_by_family["known_superiority"],
            "absent_per_seed_gains": gains_by_family["absent_noninferiority"],
            "positive_known_gaps_in_every_seed": all(value > 0 for gains in gains_by_family["known_superiority"].values() for value in gains.values()),
            "scope": "one_derived_task_at_66_bytes_not_a_storage_frontier_or_general_impossibility",
            "fingerprint": "handcrafted_approximate_reference_bypasses_Qwen_not_a_primary_matched_reader_comparator"}


def plot_report(report: dict[str, Any], output: Path, *, figure_label: str = "") -> None:
    rows = report["accuracy"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
    for axis, category, title in zip(axes, ("known", "absent"), ("known entities", "absent entity"), strict=True):
        for index, row in enumerate(rows):
            axis.barh(index, row[category] * 100, color=row["color"], height=0.62,
                      hatch="//" if row["condition"] in ("fingerprint", "drop", "full_history") else None)
            label_position = row[category] * 100
            if row["method"] != "baseline":
                values = list(row[f"{category}_per_seed"].values())
                axis.scatter([value * 100 for value in values], [index - 0.13, index, index + 0.13],
                             color="#172027", s=16, zorder=3)
                label_position = max(label_position, *(value * 100 for value in values))
            axis.text(min(label_position + 1.2, 103), index, f"{row[category] * 100:.1f}", va="center", fontsize=8)
        axis.set_xlim(0, 112)
        axis.set_xticks([0, 25, 50, 75, 100])
        axis.set_xlabel("exact accuracy (%)")
        axis.set_title(title)
        axis.set_axisbelow(True)
        axis.grid(axis="x", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(range(len(rows)), [row["label"] for row in rows])
    axes[0].invert_yaxis()
    prefix = f"{figure_label}\n" if figure_label else ""
    figure.suptitle(prefix + "opaque associations: 66-byte methods and separate references")
    figure.text(0.5, 0.02, "dots: three optimization seeds; hatched rows: separate references", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.04, 1, 0.91 if figure_label else 0.95))
    for suffix in ("png", "svg"):
        figure.savefig(output / f"accuracy.{suffix}", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4.2))
    for index, (name, row) in enumerate(report["known_contrasts"].items()):
        color = "#217b68" if row["lower"] > 0 else "#ac4949" if row["upper"] < 0 else "#687782"
        axis.hlines(index, row["lower"] * 100, row["upper"] * 100, color=color, linewidth=2.5)
        axis.scatter(row["difference"] * 100, index, color=color, s=32, zorder=3)
    axis.axvline(0, color="#666666", linestyle="--", linewidth=1)
    axis.set_yticks(range(len(report["known_contrasts"])), [name.replace("_", " ") for name in report["known_contrasts"]])
    axis.invert_yaxis()
    axis.set_xlabel("query-pool known accuracy minus comparator (percentage points)")
    axis.set_title(prefix + "99% paired-world intervals for the five primary comparisons")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", alpha=0.2)
    figure.text(0.5, 0.015, "conditional on these checkpoints; not uncertainty over the population of training seeds", ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    for suffix in ("png", "svg"):
        figure.savefig(output / f"known_contrasts.{suffix}", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-protocol", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure-label", default="")
    args = parser.parse_args()
    frozen, summary = (json.loads(path.read_text()) for path in (args.study_protocol, args.summary))
    if summary["provenance"]["study_protocol_sha256"] != sha256(args.study_protocol):
        raise ValueError("summary is not bound to this frozen study")
    validate_analysis_sources(frozen, summary)
    report = build_report(frozen, summary)
    report["figure_label"] = args.figure_label
    sources = [Path(__file__), Path("src/tinymem/evaluation/association_study.py")]
    report["provenance"] = {"study_protocol_sha256": sha256(args.study_protocol), "summary_sha256": sha256(args.summary),
                            "source_sha256": {str(path): sha256(path) for path in sources}}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    plot_report(report, args.output, figure_label=args.figure_label)
    for path in sources:
        (args.output / path.name).write_bytes(path.read_bytes())
    print(json.dumps(report["assessment"], indent=2), flush=True)


if __name__ == "__main__":
    main()
