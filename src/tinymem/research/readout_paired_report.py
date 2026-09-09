"""Validation for complete paired readout experiments."""

from collections.abc import Mapping, Sequence


def validate_arm_pairs(protocols: Sequence[Mapping[str, object]]) -> list[int]:
    """Return sorted seeds after requiring exactly one run per arm and seed."""
    pairs: dict[int, set[str]] = {}
    for protocol in protocols:
        arm = protocol.get("arm")
        seed = protocol.get("seed")
        if arm not in ("affine", "gelu"):
            raise ValueError("unknown readout arm")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        arms = pairs.setdefault(seed, set())
        if arm in arms:
            raise ValueError("duplicate arm and seed")
        arms.add(arm)
    if not pairs or any(arms != {"affine", "gelu"} for arms in pairs.values()):
        raise ValueError("both arms are required for every seed")
    return sorted(pairs)


def validate_paired_protocols(protocols: Sequence[Mapping[str, object]]) -> list[int]:
    """Require a common experiment contract and a matched schedule per seed."""
    seeds = validate_arm_pairs(protocols)
    common_fields = (
        "kind", "steps", "splits", "checkpoint_selection", "optimizer",
        "max_new_tokens", "persistent_bytes", "input_identity",
        "input_identity_verification", "reader_parameters_sha256", "reader_config",
        "reader_width", "device", "torch_version", "source_sha256", "shared_parameters",
        "cuda_version", "device_name", "deterministic_algorithms", "reader_dtype",
    )
    reference = protocols[0]
    for field in common_fields:
        if field not in reference:
            raise ValueError(f"missing protocol field: {field}")
        for protocol in protocols[1:]:
            if field not in protocol or protocol[field] != reference[field]:
                raise ValueError(f"unmatched protocol field: {field}")
    for seed in seeds:
        pair = [protocol for protocol in protocols if protocol["seed"] == seed]
        if "schedule" not in pair[0] or "schedule" not in pair[1]:
            raise ValueError("missing training schedule")
        if pair[0]["schedule"] != pair[1]["schedule"]:
            raise ValueError("unmatched training schedule")
    return seeds


def write_paired_report(report: dict, destination) -> None:
    """Write a JSON report and a readable summary to a fresh directory."""
    import json
    from pathlib import Path

    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    lines = [
        "# Readout interface paired report", "",
        f"Persistent bytes per history: {report['persistent_bytes']}",
        f"Seeds: {report['seeds']}",
        f"Full-text qualification: {report['full_text_qualification']}",
        f"Training peak memory: {report['training_peak_memory']}", "",
        report['interpretation'], "",
        "The machine-readable report.json contains all metrics, paired intervals,",
        "control contrasts, run seals, and measured costs.", "",
    ]
    lines += ["## Run metrics", "", "| Arm | Seed | Split / phase / condition | Known accuracy | Absent accuracy | Answer CE |", "| --- | --- | --- | --- | --- | --- |"]
    def display(value):
        return "n/a" if value is None else f"{value:.6g}"

    for run in report["runs"]:
        for group, metrics in sorted(run["metrics"].items()):
            lines.append(
                f"| {run['arm']} | {run['seed']} | {group} | "
                f"{display(metrics['known_accuracy'])} | {display(metrics['absent_correct'])} | "
                f"{display(metrics['answer_ce_token_weighted'])} |"
            )
    lines += ["", "## Measured costs", "", "| Arm | Seed | Elapsed seconds | Shared parameter bytes | Temporary vector bytes (max) | Training peak bytes |", "| --- | --- | --- | --- | --- | --- |"]
    for run in report["runs"]:
        costs = run["costs"]
        lines.append(
            f"| {run['arm']} | {run['seed']} | {display(costs['elapsed_seconds'])} | "
            f"{costs['shared_parameter_bytes']} | {costs['temporary_vector_bytes_max']} | "
            f"{display(costs['training_peak_memory_bytes'])} |"
        )
    lines.append("")
    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "report.json").write_text(payload, encoding="utf-8")
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")


def build_paired_report(paths, *, resamples=2000, bootstrap_seed=0):
    """Summarize sealed paired runs without deciding scientific qualification."""
    import json
    from collections import Counter
    from pathlib import Path

    from safetensors.torch import load_file
    import torch

    from tinymem.evaluation.longmemeval import normalized_answer
    from tinymem.research.readout_experiment import verify_run
    from tinymem.research.readout_report import (
        accuracy_metrics, condition_metrics, load_run_groups, paired_history_comparison,
        paired_control_comparison,
    )

    paths = [Path(path) for path in paths]
    loaded = []
    for path in paths:
        seal = verify_run(path)
        protocol = json.loads((path / "protocol.json").read_text())
        encodings = json.loads((path / "encodings.json").read_text())
        groups = load_run_groups(path)
        loaded.append((path, seal, protocol, encodings, groups))
    seeds = validate_paired_protocols([run[2] for run in loaded])
    reference = loaded[0][3]
    if any(run[3] != reference for run in loaded[1:]):
        raise ValueError("unmatched encodings")
    for seed in seeds:
        pair = [run for run in loaded if run[2]["seed"] == seed]
        left = load_file(str(pair[0][0] / "initial.safetensors"))
        right = load_file(str(pair[1][0] / "initial.safetensors"))
        if left.keys() != right.keys() or any(
            not torch.equal(left[key], right[key]) for key in left
        ):
            raise ValueError("unmatched initial tensors")

    comparisons = {}
    control_comparisons = {}
    for split in ("train", "development"):
        rows = []
        reference_full_text = None
        for _, _, protocol, _, groups in loaded:
            full_text = sorted(
                groups[(split, "reference", "full_text")],
                key=lambda row: (row["history_id"], row["case_id"]),
            )
            if reference_full_text is None:
                reference_full_text = full_text
            elif full_text != reference_full_text:
                raise ValueError("unmatched full-text reference")
            for row in groups[(split, "final", "normal")]:
                rows.append({
                    "arm": protocol["arm"], "seed": protocol["seed"],
                    "history_id": row["history_id"], "case_id": row["case_id"],
                    "category": "known" if row["category"] == "update_known" else "missing",
                    "correct": normalized_answer(row["prediction"]) == normalized_answer(row["answer"]),
                })
        comparisons[split] = paired_history_comparison(
            rows, resamples=resamples, bootstrap_seed=bootstrap_seed,
        )

    for split in ("train", "development"):
        control_comparisons[split] = {}
        for arm in ("affine", "gelu"):
            control_comparisons[split][arm] = {}
            for control in ("zero", "no_memory", "shuffled"):
                rows = []
                for _, _, protocol, _, groups in loaded:
                    if protocol["arm"] != arm:
                        continue
                    for condition in ("normal", control):
                        for row in groups[(split, "final", condition)]:
                            rows.append({
                                "arm": arm, "seed": protocol["seed"],
                                "condition": condition,
                                "history_id": row["history_id"], "case_id": row["case_id"],
                                "category": "known" if row["category"] == "update_known" else "missing",
                                "correct": normalized_answer(row["prediction"]) == normalized_answer(row["answer"]),
                            })
                control_comparisons[split][arm][control] = paired_control_comparison(
                    rows, reference=control, resamples=resamples, bootstrap_seed=bootstrap_seed,
                )

    train_answers = Counter(
        query["answer"] for history in reference["train"] for query in history["queries"]
    )
    mode = min(train_answers, key=lambda answer: (-train_answers[answer], answer))
    baseline = {
        split: accuracy_metrics([
            {"history_id": history["history_id"], "case_id": query["case_id"],
             "category": query["category"], "answer": query["answer"], "prediction": mode}
            for history in reference[split] for query in history["queries"]
        ])
        for split in ("train", "development")
    }
    return {
        "kind": "readout_paired_report_v1", "seeds": seeds,
        "persistent_bytes": 66, "full_text_qualification": "rule_not_frozen",
        "training_marginal_answer_mode": mode,
        "training_marginal_answer_metrics": baseline,
        "comparisons": comparisons,
        "control_comparisons": control_comparisons,
        "runs": [{
            "path": str(path), "seal": seal,
            "arm": protocol["arm"], "seed": protocol["seed"],
            "costs": {
                "elapsed_seconds": seal["elapsed_seconds"],
                "shared_parameters": protocol["shared_parameters"],
                "shared_parameter_bytes": 4 * sum(protocol["shared_parameters"].values()),
                "persistent_bytes_per_history": protocol["persistent_bytes"],
                "training_peak_memory_bytes": seal.get("training_peak_memory_bytes"),
                "temporary_vector_bytes_max": max(
                    row["temporary_vector_bytes"]
                    for rows in groups.values() for row in rows
                ),
            },
            "metrics": {"/".join(key): condition_metrics(rows) for key, rows in groups.items()},
        } for path, seal, protocol, _, groups in loaded],
        "interpretation": "Jointly trained arms; not a fixed-trained-state causal contrast.",
        "training_peak_memory": (
            "cuda_allocated_bytes_including_reader" if all(
                seal.get("training_peak_memory_bytes") is not None for _, seal, _, _, _ in loaded
            ) else "unmeasured"
        ),
    }
