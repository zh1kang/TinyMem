"""Training-only answer ranking from sealed readout checkpoints; no optimization."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import torch

from tinymem.data.opaque_qa1 import ROOMS
from tinymem.research.prefix_reader import prefix_answer_loss
from tinymem.research.readout_checkpoint import load_checkpoint
from tinymem.research.readout_controls import controlled_state, state_donors
from tinymem.research.readout_experiment import _reader_hash, _source_hashes, verify_run
from tinymem.research.readout_interface import encode_readout_history
from tinymem.research.study_runtime import check_repository, prepare_device
from tinymem.research.update_protocol import file_sha256, load_shared_reader, read_json


CANDIDATES = (*ROOMS, "unknown")
CONDITIONS = ("normal", "shuffled", "zero")
SELECTION_SALT = "tinymem-readout-ranking-v1:"
GOLD_CE_TOLERANCE = 3e-6
HISTORY_COUNT = 32
STATE_ENCODING_POLICY = "all_training_histories_in_original_order_with_loaded_parameter_flags"


def select_histories(rows: list[dict], count: int) -> list[str]:
    """Use a fixed hash order, independent of answers, length, and accuracy."""
    ids = [row["history_id"] for row in rows]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("history identities must be nonempty and unique")
    if type(count) is not int or not 1 <= count <= len(ids):
        raise ValueError("history count is outside the available training set")
    return sorted(ids, key=lambda value: (hashlib.sha256((SELECTION_SALT + value).encode()).hexdigest(), value))[:count]


@torch.inference_mode()
def candidate_scores(reader, before, memory, after, candidates: dict[str, tuple[int, ...]]) -> dict[str, float]:
    """Sum causal log probabilities of each complete answer, including EOS.

    Candidates are fixed across questions. Gold labels do not enter this call.
    Serial execution preserves the original reader's numerical conventions.
    """
    if not candidates or any(not name or len(ids) < 2 for name, ids in candidates.items()):
        raise ValueError("nonempty candidates with answer and stop tokens are required")
    if any(module.training for module in reader.model.modules()) or any(p.requires_grad or p.grad is not None for p in reader.model.parameters()):
        raise ValueError("reader must be frozen and in evaluation mode")
    result = {}
    for name, ids in candidates.items():
        loss = prefix_answer_loss(reader, before, memory, after,
                                  torch.tensor(ids, device=reader.model.device))
        result[name] = -float(loss) * len(ids)
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("candidate scores must be finite")
    return result


def ranking_metrics(scores: dict[str, float], gold: str) -> dict:
    if gold not in scores or "unknown" not in scores or len(scores) < 2:
        raise ValueError("scores must include gold, unknown, and another candidate")
    if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError("candidate scores must be finite")
    gold_score = scores[gold]
    winners = sorted(name for name, value in scores.items() if value == max(scores.values()))
    rank_min = 1 + sum(value > gold_score for value in scores.values())
    rank_max = sum(value >= gold_score for value in scores.values())
    rooms = {name: value for name, value in scores.items() if name != "unknown"}
    room_winners = [name for name, value in rooms.items() if value == max(rooms.values())]
    return {
        "top_candidates": winners, "top1_correct": winners == [gold],
        "gold_rank_min": rank_min, "gold_rank_max": rank_max,
        "gold_reciprocal_midrank": 2 / (rank_min + rank_max),
        "gold_log_probability": gold_score,
        "gold_margin": gold_score - max(value for name, value in scores.items() if name != gold),
        "gold_minus_unknown": gold_score - scores["unknown"] if gold != "unknown" else None,
        "known_only_top1_correct": room_winners == [gold] if gold != "unknown" else None,
    }


def load_inputs(directory: Path, count: int) -> tuple[dict, dict, list[dict], dict, list[str]]:
    if count != HISTORY_COUNT:
        raise ValueError("this diagnostic requires exactly 32 training histories")
    seal = verify_run(directory)
    protocol = read_json(directory / "protocol.json")
    if protocol["source_sha256"] != _source_hashes():
        raise ValueError("original execution sources do not match this checkout")
    rows = read_json(directory / "encodings.json")["train"]
    selected = select_histories(rows, count)
    if len(rows) != 256 or protocol["steps"] != 1000 or protocol["persistent_bytes"] != 66:
        raise ValueError("expected the completed 256-history, 1000-step, 66-byte study")
    saved = {}
    with (directory / "predictions.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] == "train" and row["phase"] == "final" and row["condition"] in CONDITIONS:
                key = row["history_id"], row["case_id"], row["condition"]
                if key in saved:
                    raise ValueError("duplicate saved training prediction")
                saved[key] = row
    donors = state_donors([row["history_id"] for row in rows])
    cases = set()
    for row in rows:
        queries = row["queries"]
        if (len(queries) != 10 or sum(q["category"] == "update_known" for q in queries) != 8
                or sum(q["category"] == "update_missing" for q in queries) != 2):
            raise ValueError("each history requires eight known and two absent queries")
        for query in queries:
            if query["case_id"] in cases or query["answer"] not in CANDIDATES:
                raise ValueError("invalid or duplicate query")
            cases.add(query["case_id"])
            if (query["category"] == "update_missing") != (query["answer"] == "unknown"):
                raise ValueError("query category disagrees with answer")
            for condition in CONDITIONS:
                original = saved[row["history_id"], query["case_id"], condition]
                donor = donors[row["history_id"]] if condition == "shuffled" else row["history_id"]
                if (original["answer"] != query["answer"] or original["category"] != query["category"]
                        or original["donor_history_id"] != donor):
                    raise ValueError("saved prediction disagrees with encoded training query or donor")
    if len(saved) != 256 * 10 * len(CONDITIONS):
        raise ValueError("unexpected saved training prediction coverage")
    return seal, protocol, rows, saved, selected


def summarize(records: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in records:
        groups[row["condition"], row["category"]].append(row)
    result = {}
    for (condition, category), rows in groups.items():
        fields = ("top1_correct", "gold_reciprocal_midrank", "gold_log_probability", "gold_margin",
                  "gold_minus_unknown", "known_only_top1_correct")
        result[f"{condition}/{category}"] = {
            "queries": len(rows), "histories": len({row["history_id"] for row in rows}),
            **{field: statistics.mean(row[field] for row in rows) if rows[0][field] is not None else None
               for field in fields},
            "saved_generation_accuracy": statistics.mean(row["saved_correct"] for row in rows),
            "top1_unknown_rate": statistics.mean(row["top_candidates"] == ["unknown"] for row in rows),
            "top1_tie_rate": statistics.mean(len(row["top_candidates"]) > 1 for row in rows),
        }
    return result


def write_json(path: Path, value: object) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


@torch.inference_mode()
def evaluate_ranking(reader, encoder, bridge, rows, saved, selected, candidates, handle) -> list[dict]:
    """Read only detached states; retain the original full-training donor cycle."""
    by_id = {row["history_id"]: row for row in rows}
    donors = state_donors(list(by_id))
    device = reader.model.device
    # Match original state construction; the alternate replay changed one FP32 value.
    states = {row["history_id"]: controlled_state(encode_readout_history(
        reader, encoder, torch.tensor(row["history_ids"], device=device)), "normal")
        for row in rows}
    records = []
    for index, identity in enumerate(selected, 1):
        row = by_id[identity]
        before = torch.tensor(row["before_ids"], device=device)
        for condition in CONDITIONS:
            donor = donors[identity] if condition == "shuffled" else identity
            state = controlled_state(states[donor], "zero" if condition == "zero" else "normal")
            memory = bridge(state)
            for query in row["queries"]:
                scores = candidate_scores(reader, before, memory,
                    torch.tensor(query["after_ids"], device=device), candidates)
                original = saved[identity, query["case_id"], condition]
                gold_ce = -scores[query["answer"]] / len(candidates[query["answer"]])
                error = abs(gold_ce - original["answer_ce"])
                if error > GOLD_CE_TOLERANCE:
                    raise ValueError(f"gold CE replay differs for {query['case_id']}/{condition}: {error}")
                record = {
                    "history_id": identity, "case_id": query["case_id"], "condition": condition,
                    "donor_history_id": donor, "category": query["category"], "answer": query["answer"],
                    "scores": scores, **ranking_metrics(scores, query["answer"]),
                    "saved_prediction": original["prediction"], "saved_correct": original["correct"],
                    "gold_ce_replay_error": error,
                }
                records.append(record)
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        print(json.dumps({"histories_scored": index, "total_histories": len(selected)}), flush=True)
    return records


def run(directory: Path, output: Path, count: int, device_name: str) -> None:
    if output.resolve().is_relative_to(directory.resolve()):
        raise ValueError("diagnostic output must be outside the sealed input run")
    if output.exists():
        raise FileExistsError(output)
    seal, protocol, rows, saved, selected = load_inputs(directory, count)
    started = time.perf_counter()
    device = prepare_device(device_name)
    reader = load_shared_reader(protocol["input_identity"]["reader"], device)
    reader_hash = _reader_hash(reader)
    runtime = {"device": str(reader.model.device), "torch_version": str(torch.__version__),
               "cuda_version": torch.version.cuda,
               "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
               "reader_dtype": str(reader.model.get_input_embeddings().weight.dtype),
               "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    if reader_hash != protocol["reader_parameters_sha256"] or any(protocol[k] != v for k, v in runtime.items()):
        raise ValueError("reader parameters or numerical runtime differ from the sealed run")
    candidates = {name: (*reader.tokenizer.encode(name, add_special_tokens=False), reader.tokenizer.eos_token_id)
                  for name in CANDIDATES}
    if len(set(candidates.values())) != len(candidates):
        raise ValueError("candidate tokenizations must be distinct")
    for row in rows:
        for query in row["queries"]:
            if candidates[query["answer"]] != tuple(query["answer_ids"]):
                raise ValueError("candidate tokenization differs from the sealed gold answer")
    encoder, bridge = load_checkpoint(directory / "final.safetensors",
        expected_sha256=seal["files"]["final.safetensors"], reader_width=protocol["reader_width"], kind=protocol["arm"])
    encoder.to(device)
    bridge.to(device)
    weights = {name: tensor.detach().clone() for prefix, module in (("encoder", encoder), ("bridge", bridge))
               for name, tensor in ((f"{prefix}.{key}", value) for key, value in module.state_dict().items())}
    script_hash = file_sha256(Path(__file__))
    manifest = {"kind": "readout_ranking_diagnostic_v1", "run": str(directory.resolve()),
                "arm": protocol["arm"], "seed": protocol["seed"], "source_seal": seal,
                "source_complete_sha256": file_sha256(directory / "complete.json"),
                "runtime": runtime, "reader_parameters_sha256": reader_hash,
                "original_sources": protocol["source_sha256"], "script_sha256": script_hash,
                "selection_salt": SELECTION_SALT, "selected_training_histories": selected,
                "state_encoding_policy": STATE_ENCODING_POLICY,
                "donors": state_donors([row["history_id"] for row in rows]),
                "candidate_ids": candidates, "conditions": CONDITIONS, "gold_ce_tolerance": GOLD_CE_TOLERANCE,
                "score": "sum_log_probability_including_eos_no_length_normalization",
                "ties": "strict_top1_and_min_max_gold_ranks", "checkpoint_policy": "existing_final_only",
                "scope": "post_hoc_training_subset_diagnostic_not_a_replacement_generation_metric"}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", manifest)
    with (output / "scores.jsonl").open("x") as handle:
        records = evaluate_ranking(reader, encoder, bridge, rows, saved, selected, candidates, handle)
    for prefix, module in (("encoder", encoder), ("bridge", bridge)):
        if any(not torch.equal(tensor, weights[f"{prefix}.{name}"]) for name, tensor in module.state_dict().items()):
            raise ValueError("diagnostic changed encoder or bridge weights")
    if (_reader_hash(reader) != reader_hash or _source_hashes() != protocol["source_sha256"]
            or file_sha256(Path(__file__)) != script_hash or verify_run(directory) != seal):
        raise ValueError("reader, source, or input artifacts changed during the diagnostic")
    write_json(output / "summary.json", summarize(records))
    write_json(output / "complete.json", {
        "kind": "readout_ranking_complete_v1", "records": len(records),
        "elapsed_seconds": time.perf_counter() - started,
        "max_gold_ce_replay_error": max(row["gold_ce_replay_error"] for row in records),
        "files": {name: file_sha256(output / name) for name in ("protocol.json", "scores.jsonl", "summary.json")},
    })


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--histories", type=int, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    args = parser.parse_args(argv)
    check_repository()
    if args.action == "check":
        _, protocol, _, _, selected = load_inputs(args.run, args.histories)
        print(json.dumps({"arm": protocol["arm"], "seed": protocol["seed"], "selected": selected,
                          "model_loaded": False, "split": "train"}, indent=2))
    else:
        if args.output is None or args.device is None:
            parser.error("run requires --output and --device")
        run(args.run, args.output, args.histories, args.device)


if __name__ == "__main__":
    main()
