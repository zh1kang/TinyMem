"""Pair the six sealed training-subset ranking diagnostics without changing gates."""

import argparse
import json
from pathlib import Path
import statistics

import numpy as np

from scripts.diagnose_readout_ranking import (
    CANDIDATES, CONDITIONS, GOLD_CE_TOLERANCE, HISTORY_COUNT, ranking_metrics, summarize, write_json,
)
from tinymem.research.readout_experiment import verify_run
from tinymem.research.update_protocol import file_sha256, read_json


def build_report(source_root: Path, directories: list[Path]) -> dict:
    loaded = {}
    reference = None
    common = ("runtime", "reader_parameters_sha256", "original_sources", "script_sha256",
              "selection_salt", "selected_training_histories", "state_encoding_policy", "donors", "candidate_ids",
              "conditions", "gold_ce_tolerance", "score", "ties", "checkpoint_policy", "scope")
    for directory in directories:
        seal = read_json(directory / "complete.json")
        if seal["kind"] != "readout_ranking_complete_v1" or set(seal["files"]) != {"protocol.json", "scores.jsonl", "summary.json"}:
            raise ValueError("invalid diagnostic completion seal")
        for name, digest in seal["files"].items():
            if file_sha256(directory / name) != digest:
                raise ValueError("diagnostic artifact changed")
        protocol = read_json(directory / "protocol.json")
        if (len(protocol["selected_training_histories"]) != HISTORY_COUNT
                or len(set(protocol["selected_training_histories"])) != HISTORY_COUNT
                or protocol["gold_ce_tolerance"] != GOLD_CE_TOLERANCE):
            raise ValueError("diagnostic selection count or replay tolerance differs")
        key = protocol["arm"], protocol["seed"]
        if key in loaded:
            raise ValueError("duplicate arm and seed")
        source = source_root / f"{key[0]}_seed_{key[1]}"
        if (file_sha256(source / "complete.json") != protocol["source_complete_sha256"]
                or verify_run(source) != protocol["source_seal"]):
            raise ValueError("original run seal does not match")
        if reference is None:
            reference = protocol
        elif any(protocol[field] != reference[field] for field in common):
            raise ValueError("diagnostic protocols differ")
        records = [json.loads(line) for line in (directory / "scores.jsonl").read_text().splitlines()]
        for row in records:
            if not 0 <= row["gold_ce_replay_error"] <= GOLD_CE_TOLERANCE:
                raise ValueError("gold CE replay tolerance exceeded")
            if set(row["scores"]) != set(CANDIDATES):
                raise ValueError("candidate inventory differs")
            expected = ranking_metrics(row["scores"], row["answer"])
            if any(row[field] != value for field, value in expected.items()):
                raise ValueError("cached ranking metrics differ from scores")
        if not records or max(row["gold_ce_replay_error"] for row in records) != seal["max_gold_ce_replay_error"]:
            raise ValueError("gold CE replay maximum differs from seal")
        if len(records) != seal["records"] or len(records) != 10 * len(protocol["selected_training_histories"]) * len(CONDITIONS):
            raise ValueError("diagnostic coverage differs")
        if summarize(records) != read_json(directory / "summary.json"):
            raise ValueError("cached summary differs")
        indexed = {(r["history_id"], r["case_id"], r["condition"]): r for r in records}
        if len(indexed) != len(records):
            raise ValueError("duplicate query records")
        loaded[key] = (protocol, seal, records, indexed)
    seeds = (1337, 2027, 4099)
    if set(loaded) != {(arm, seed) for arm in ("affine", "gelu") for seed in seeds}:
        raise ValueError("all six declared arm/seed diagnostics are required")
    histories = reference["selected_training_histories"]
    rng = np.random.default_rng(0)
    weights = rng.multinomial(len(histories), np.full(len(histories), 1 / len(histories)), size=2000)
    fields = ("top1_correct", "gold_reciprocal_midrank", "gold_log_probability", "gold_margin", "known_only_top1_correct")
    contrasts = {}
    for arm in ("affine", "gelu"):
        contrasts[arm] = {}
        for control in ("shuffled", "zero"):
            contrasts[arm][control] = {}
            for category, count in (("update_known", 8), ("update_missing", 2)):
                metrics = {}
                for field in fields:
                    if field == "known_only_top1_correct" and category == "update_missing":
                        continue
                    per_seed, bootstrap = {}, []
                    for seed in seeds:
                        indexed = loaded[arm, seed][3]
                        differences = []
                        for history in histories:
                            normal = [r for (h, _, condition), r in indexed.items()
                                      if h == history and condition == "normal" and r["category"] == category]
                            if len(normal) != count:
                                raise ValueError("incomplete query/category coverage")
                            delta = []
                            for row in normal:
                                other = indexed[history, row["case_id"], control]
                                if (other["answer"], other["category"]) != (row["answer"], row["category"]):
                                    raise ValueError("unmatched control query")
                                delta.append(float(row[field]) - float(other[field]))
                            differences.append(statistics.mean(delta))
                        per_seed[str(seed)] = statistics.mean(differences)
                        bootstrap.append(weights @ np.asarray(differences) / len(histories))
                    metrics[field] = {
                        "mean_difference": statistics.mean(per_seed.values()), "seed_differences": per_seed,
                        "seed_sd": statistics.stdev(per_seed.values()),
                        "interval95": np.quantile(np.mean(bootstrap, axis=0), [0.025, 0.975]).tolist(),
                    }
                contrasts[arm][control][category] = metrics
    return {
        "kind": "readout_ranking_report_v1", "histories": len(histories),
        "interpretation": "post_hoc_training_subset; descriptive paired history intervals conditional on three observed seeds",
        "resamples": 2000, "bootstrap_seed": 0, "contrasts": contrasts,
        "runs": [{"arm": arm, "seed": seed, "seal": value[1], "metrics": summarize(value[2])}
                 for (arm, seed), value in sorted(loaded.items())],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = build_report(args.source_root, args.runs)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "report.json", report)
    print(json.dumps({"report": str(args.output / "report.json")}))


if __name__ == "__main__":
    main()
