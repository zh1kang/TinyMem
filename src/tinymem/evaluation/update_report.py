"""Verify sealed evaluations and produce descriptive, count-aware update reports."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from tinymem.evaluation.memory_updates import UpdatePrediction, score_update_episode
from tinymem.evaluation.update_aggregate import aggregate_updates
from tinymem.memory.packed_tokens import PackedTokenState
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.update_experiment import (
    SEEDS, confirmation_episodes, read_rows, verify_complete, verify_launch,
    verify_training, write_json,
)
from tinymem.research.update_protocol import file_sha256, load_development_data, read_json
from tinymem.research.update_runner import BASELINES, CONTROLS, NEURAL_METHODS, STAGES, check_state


def _states(row: dict, method: str) -> None:
    if method in CONTROLS:
        expected = None if method == "full_context" else 0
        if row["states"] != {} or row["persistent_bytes"] != expected:
            raise ValueError("capability control has unexpected persistent state")
        return
    if set(row["states"]) != set(STAGES):
        raise ValueError("state snapshots must cover every paired stage")
    sizes = []
    for snapshot in row["states"].values():
        if method in NEURAL_METHODS:
            if set(snapshot) != {"values", "valid", "nbytes"}:
                raise ValueError("learned state snapshot fields changed")
            if any(type(value) is not bool for values in snapshot["valid"] for value in values):
                raise ValueError("validity values must be booleans")
            state = LatentSlotState(torch.tensor(snapshot["values"], dtype=torch.float32),
                                    torch.tensor(snapshot["valid"], dtype=torch.bool))
        else:
            if set(snapshot) != {"payload", "nbytes"}:
                raise ValueError("packed state snapshot fields changed")
            if any(type(value) is not int or not 0 <= value <= 255 for values in snapshot["payload"] for value in values):
                raise ValueError("packed payload requires uint8 values")
            state = PackedTokenState(torch.tensor(snapshot["payload"], dtype=torch.uint8))
        check_state(state)
        if type(snapshot["nbytes"]) is not int or snapshot["nbytes"] != state.nbytes:
            raise ValueError("snapshot byte count disagrees with stored payload")
        sizes.append(state.nbytes)
    if type(row["persistent_bytes"]) is not int or row["persistent_bytes"] != max(sizes):
        raise ValueError("result byte count differs from its persistent states")


def verify_evaluation(launch_dir, launch, data, episodes, method, seed, split):
    name = method if seed is None else f"{method}_seed_{seed}"
    directory = launch_dir / "evaluations" / split / name
    complete = verify_complete(directory, {"protocol.json", "predictions.jsonl", "encodings.json"})
    protocol = read_json(directory / "protocol.json")
    checkpoint = None if seed is None else verify_training(launch_dir, launch, data, method, seed)
    expected = {"kind": "evaluation", "method": method, "seed": seed, "split": split,
                "launch_complete_sha256": file_sha256(launch_dir / "complete.json"),
                "evidence_kind": launch["evidence_kind"],
                "checkpoint_sha256": None if checkpoint is None else file_sha256(checkpoint)}
    if (any(protocol.get(key) != value for key, value in expected.items())
            or complete["kind"] != "evaluation" or complete["histories"] != len(episodes)):
        raise ValueError("evaluation provenance differs from the frozen launch")
    rows = read_rows(directory / "predictions.jsonl")
    gold = {row.episode_id: row for row in episodes}
    if len(rows) != len(gold) or {row["episode_id"] for row in rows} != set(gold):
        raise ValueError("evaluation must cover every expected history exactly once")
    encoded = read_json(directory / "encodings.json")
    if len(encoded) != len(gold) or {row["episode_id"] for row in encoded} != set(gold):
        raise ValueError("evaluation encodings do not cover expected histories")
    scored = []
    for row in rows:
        expected_reader = "handcrafted" if method == "fingerprint" else "shared_frozen_reader"
        if row["method"] != method or row["reader_kind"] != expected_reader:
            raise ValueError("prediction method or reader type changed")
        _states(row, method)
        predictions = [UpdatePrediction(item["case_id"], item["prediction"]) for item in row["predictions"]]
        result = score_update_episode(gold[row["episode_id"]], predictions)
        if row["metrics"] != result.to_dict():
            raise ValueError("cached metrics disagree with authoritative raw predictions")
        expected_stages = {case.case_id: stage for stage, cases in zip(STAGES,
            (gold[row["episode_id"]].before, *(b.queries for b in gold[row["episode_id"]].branches)), strict=True) for case in cases}
        if any(item["stage"] != expected_stages[item["case_id"]] for item in row["predictions"]):
            raise ValueError("prediction stage metadata changed")
        scored.append(result)
    return scored, {"directory": str(directory), "complete_sha256": file_sha256(directory / "complete.json"),
                    "shared_costs": {key: value for key, value in protocol.items() if key.startswith("shared_")}}


def _percent(value):
    return "undefined" if value is None else f"{value * 100:.2f}%"


def _interval_text(interval):
    bounds = interval["bounds"]
    if bounds is None:
        return f"undefined ({interval['undefined_resamples']}/{interval['resamples']} draws)"
    return f"[{100 * bounds[0]:.2f}, {100 * bounds[1]:.2f}]"


def render_report(report: dict) -> str:
    aggregate = report["aggregate"]
    lines = ["# Fixed-byte memory-update report", "", f"Evidence: **{report['evidence_kind']}**; split: **{report['split']}**.",
             "Synthetic fixtures are engineering checks, not model-quality results." if report["evidence_kind"] == "synthetic_fixture" else
             "These are measured outputs of the hash-verified frozen launch, not a claim of learned-memory superiority.", "",
             "## Accuracy and correction behavior", "",
             "| Method | Before known | Before absent | Correction target | Stale correction |", "|---|---:|---:|---:|---:|"]
    for method, family in aggregate["families"].items():
        rates = family["rates"]
        keys = ("before.known_accuracy", "before.absent_accuracy", "correction.target.update_accuracy", "correction.target.stale_answer")
        lines.append("| " + method + " | " + " | ".join(_percent(rates[key]["mean"]) for key in keys) + " |")
    lines += ["", "## Paired unchanged-fact outcomes", "",
              "| Method/event | Before | After | Conditional forgetting | Unconditional forgetting |", "|---|---:|---:|---:|---:|"]
    for method, family in aggregate["families"].items():
        for event in STAGES[1:]:
            rates = family["rates"]
            prefix = f"{event}.unchanged_known."
            keys = ("before_accuracy", "after_accuracy", "conditional_forgetting", "unconditional_forgetting")
            lines.append("| " + f"{method}/{event}" + " | " + " | ".join(_percent(rates[prefix + key]["mean"]) for key in keys) + " |")
    lines += ["", "## Seed variation and paired history uncertainty", "",
              f"Intervals are descriptive {100 * aggregate['bootstrap']['confidence']:g}% percentile intervals; difference bounds are percentage points.", "",
              "| Method | Before-known seed values | Seed SD (points) | Before-known history interval (%) |", "|---|---|---:|---|"]
    for method, family in aggregate["families"].items():
        rate = family["rates"]["before.known_accuracy"]
        seed_values = ", ".join(_percent(value) for value in rate["seed_values"].values())
        sd = "not applicable" if rate["seed_sd"] is None else f"{100 * rate['seed_sd']:.2f}"
        lines.append(f"| {method} | {seed_values} | {sd} | {_interval_text(rate['history_interval'])} |")
    lines += ["", "| Contrast | Outcome | Difference (points) | Paired history interval (points) |", "|---|---|---:|---|"]
    for contrast, rates in aggregate["contrasts"].items():
        for key in ("correction.target.update_accuracy", "correction.unchanged_known.after_accuracy", "correction.unchanged_known.conditional_forgetting"):
            rate = rates[key]
            difference = "undefined" if rate["difference"] is None else f"{100 * rate['difference']:.2f}"
            lines.append(f"| {contrast} | {key} | {difference} | {_interval_text(rate['history_interval'])} |")
    lines += ["", "## Per-seed correction denominators", "",
              "| Run | Before known | Before absent | Unchanged correct before | Conditional forgetting |", "|---|---|---|---|---|"]
    for name, run in aggregate["runs"].items():
        keys = ("before.known_accuracy", "before.absent_accuracy", "correction.unchanged_known.before_accuracy", "correction.unchanged_known.conditional_forgetting")
        counts = [f"{run['rates'][key]['numerator']}/{run['rates'][key]['denominator']}" for key in keys]
        lines.append("| " + name + " | " + " | ".join(counts) + " |")
    lines += ["", "## Development competence", ""]
    for run, gate in report["development_gates"].items():
        lines.append(f"- {run}: **{gate['interpretation']}**.")
    lines += ["", "## Interpretation and uncertainty", "",
        "- Every JSON rate retains per-seed pooled numerators and denominators. Family means average the seed-specific ratios, never pool seeds as extra histories.",
        "- JSON includes complete CC/CW/WC/WW counts, target/addition/repetition outcomes, untouched versus unchanged cohorts, absent errors and all state accuracies.",
        "- Paired percentile intervals resample whole histories jointly across methods and the observed fixed seeds. Seed SD/range are separate; these intervals do not quantify uncertainty over new training seeds.",
        "- If any bootstrap draw has an empty denominator, its interval is withheld and the undefined-draw count is reported. No undefined draw is silently discarded.",
        "- Contrasts are descriptive, not multiplicity-adjusted significance tests. Positive forgetting differences mean more forgetting, not improvement.",
        "- Conditional subsets can differ between methods. Poor before-state competence precludes a strong preservation claim; consult unconditional outcomes and denominators.",
        "- Addition increases live information; correction and repetition do not. This study measures one additional event, not long-delay robustness.",
        "- Full context is unbounded; no-memory has no history state; fingerprint uses a handcrafted reader. Shared parameters/dictionaries are recorded separately from the 66-byte stream budget.",
        "", "See report.json for all counts, per-seed outcomes, intervals, contrast directions, and artifact identities.", ""]
    return "\n".join(lines)


def report_updates(launch_dir: Path, output: Path, *, split: str, resamples: int = 10000, seed: int = 20260905) -> dict:
    if split not in ("development", "confirmation"):
        raise ValueError("explicit development or confirmation split required")
    if output.exists():
        raise FileExistsError("report output must be fresh")
    launch = read_json(launch_dir / "protocol.json")
    data = load_development_data(Path(launch["data"]))
    verify_launch(launch_dir, data)
    for run in launch["runs"]:
        verify_training(launch_dir, launch, data, run["method"], run["seed"])
    episodes = confirmation_episodes(launch_dir, data, launch) if split == "confirmation" else data.development
    runs, families, artifacts, gates = {}, {}, {}, {}
    for method in (*NEURAL_METHODS, *BASELINES, *CONTROLS):
        members = []
        for optimization_seed in SEEDS if method in NEURAL_METHODS else (None,):
            name = method if optimization_seed is None else f"{method}_seed_{optimization_seed}"
            runs[name], artifacts[name] = verify_evaluation(launch_dir, launch, data, episodes, method, optimization_seed, split)
            members.append(name)
            if optimization_seed is not None:
                gates[name] = read_json(launch_dir / "runs" / name / "gate.json")
        families[method] = members
    contrasts = [(method, reference) for method in NEURAL_METHODS for reference in ("latest_vocabulary", "latest_template")]
    contrasts.insert(0, ("query_pool", "mean_pool"))
    aggregate = aggregate_updates(runs, families=families, contrasts=contrasts, resamples=resamples, seed=seed)
    report = {"schema": "memory_update_report_v1", "evidence_kind": launch["evidence_kind"], "split": split,
              "launch_complete_sha256": file_sha256(launch_dir / "complete.json"),
              "shared_costs": launch["shared_costs"], "evaluations": artifacts, "development_gates": gates,
              "report_source_sha256": {str(Path(__file__).name): file_sha256(Path(__file__)),
                  "update_aggregate.py": file_sha256(Path(__file__).with_name("update_aggregate.py"))},
              "aggregate": aggregate}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "report.json", report)
    with (output / "report.md").open("x") as handle:
        handle.write(render_report(report))
    return report
