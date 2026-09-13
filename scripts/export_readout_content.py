"""Export frozen states and independently replayed first-appearance room targets."""

import argparse
import json
from pathlib import Path
import re

import torch
from safetensors.torch import save_file

from scripts.diagnose_readout_ranking import load_inputs, GOLD_CE_TOLERANCE, STATE_ENCODING_POLICY
from tinymem.data.memory_updates import replay_update_chunks
from tinymem.data.opaque_qa1 import ROOMS
from tinymem.research.prefix_reader import prefix_answer_loss
from tinymem.research.readout_checkpoint import load_checkpoint
from tinymem.research.readout_controls import controlled_state
from tinymem.research.readout_experiment import _reader_hash, _source_hashes, verify_run
from tinymem.research.readout_interface import encode_readout_history
from tinymem.research.study_runtime import check_repository, prepare_device
from tinymem.research.update_protocol import file_sha256, load_shared_reader, read_json


def targets_from_history(tokenizer, row):
    text = tokenizer.decode(row["history_ids"], skip_special_tokens=False)
    chunks = tuple(text.strip().split("\n\n"))
    final = replay_update_chunks(chunks)
    if len(final) != 8:
        raise ValueError("history must contain eight entities")
    queried = set()
    for query in row["queries"]:
        matches = re.findall(r"Where is (person[0-9a-f]{20})\?", tokenizer.decode(query["after_ids"]))
        if len(matches) != 1 or matches[0] in queried:
            raise ValueError("query must identify one distinct opaque entity")
        queried.add(matches[0])
        if final.get(matches[0], "unknown") != query["answer"]:
            raise ValueError("independent replay disagrees with saved label")
    if not set(final).issubset(queried) or len(queried) != 10:
        raise ValueError("queries do not cover the eight known and two absent entities")
    return [ROOMS.index(room) for room in final.values()]


def write_json(path, value):
    with path.open("x") as output:
        json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")


def export(source, output, declaration_path):
    declaration = read_json(declaration_path)
    seal, protocol, _, _, _ = load_inputs(source, 32)
    name = f"{protocol['arm']}_seed_{protocol['seed']}"
    if file_sha256(source / "complete.json") != declaration["source_complete_sha256"][name]:
        raise ValueError("source is outside the declared checkpoints")
    if output.resolve().is_relative_to(source.resolve()) or output.exists():
        raise ValueError("export output must be fresh and outside source")
    script_hash = file_sha256(Path(__file__))
    if script_hash != declaration["source_sha256"]["export_readout_content.py"]:
        raise ValueError("export code differs from declaration")
    splits = read_json(source / "encodings.json")
    if set(splits) != {"train", "development"} or len(splits["train"]) != 256 or len(splits["development"]) != 32:
        raise ValueError("unexpected source split counts")
    saved = {}
    for line in (source / "predictions.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record["phase"] == "final" and record["condition"] == "normal":
            key = record["split"], record["history_id"], record["case_id"]
            if key in saved:
                raise ValueError("duplicate original prediction")
            saved[key] = record
    if len(saved) != 2880:
        raise ValueError("missing original normal predictions")
    device = prepare_device("cuda")
    reader = load_shared_reader(protocol["input_identity"]["reader"], device)
    reader_hash = _reader_hash(reader)
    runtime = {"device": str(reader.model.device), "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda, "device_name": torch.cuda.get_device_name(device),
        "reader_dtype": str(reader.model.get_input_embeddings().weight.dtype),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    if reader_hash != protocol["reader_parameters_sha256"] or any(protocol[k] != v for k, v in runtime.items()):
        raise ValueError("reader or runtime differs from original")
    encoder, bridge = load_checkpoint(source / "final.safetensors", expected_sha256=seal["files"]["final.safetensors"],
                                     reader_width=protocol["reader_width"], kind=protocol["arm"])
    encoder.to(device)
    bridge.to(device)
    weights = {f"{prefix}.{name}": value.detach().clone() for prefix, module in (("encoder", encoder), ("bridge", bridge))
               for name, value in module.state_dict().items()}
    buffers = {name: value.cpu().clone() for name, value in reader.model.named_buffers()}
    output.mkdir(parents=True, exist_ok=False)
    with torch.inference_mode():
        states, metadata_rows, errors = {}, {}, []
        with (output / "replay.jsonl").open("x") as log:
            for split in ("train", "development"):
                rows = splits[split]
                table = {row["history_id"]: controlled_state(encode_readout_history(
                    reader, encoder, torch.tensor(row["history_ids"], device=device)), "normal") for row in rows}
                values, metadata_rows[split] = [], []
                for row in rows:
                    state = controlled_state(table[row["history_id"]], "normal")
                    if not state.valid.all() or state.nbytes != 66:
                        raise ValueError("invalid original state boundary")
                    memory = bridge(state)
                    before = torch.tensor(row["before_ids"], device=device)
                    for query in row["queries"]:
                        original = saved[split, row["history_id"], query["case_id"]]
                        ce = float(prefix_answer_loss(reader, before, memory,
                            torch.tensor(query["after_ids"], device=device), torch.tensor(query["answer_ids"], device=device)))
                        error = abs(ce - original["answer_ce"])
                        log.write(json.dumps({"split": split, "history_id": row["history_id"], "case_id": query["case_id"],
                            "answer": query["answer"], "answer_ce": ce, "saved_answer_ce": original["answer_ce"],
                            "error": error}, allow_nan=False) + "\n")
                        log.flush()
                        if error > GOLD_CE_TOLERANCE:
                            raise ValueError(f"gold replay failed: {query['case_id']} error={error}")
                        errors.append(error)
                    values.append(state.values.reshape(16).cpu().clone())
                    metadata_rows[split].append({"history_id": row["history_id"],
                        "source_group_ids": row["source_group_ids"], "targets": targets_from_history(reader.tokenizer, row)})
                states[split] = torch.stack(values)
                print(json.dumps({"split": split, "states": len(values), "max_replay_error": max(errors)}), flush=True)
    for prefix, module in (("encoder", encoder), ("bridge", bridge)):
        if any(not torch.equal(value, weights[f"{prefix}.{name}"]) for name, value in module.state_dict().items()):
            raise ValueError("export changed learned weights")
    current_buffers = dict(reader.model.named_buffers())
    if buffers.keys() != current_buffers.keys() or any(not torch.equal(value, current_buffers[name].cpu()) for name, value in buffers.items()):
        raise ValueError("reader buffer changed")
    if (_reader_hash(reader) != reader_hash or verify_run(source) != seal
            or _source_hashes() != protocol["source_sha256"] or file_sha256(Path(__file__)) != script_hash):
        raise ValueError("reader, source, or exporter changed")
    save_file(states, str(output / "states.safetensors"))
    write_json(output / "metadata.json", {"arm": protocol["arm"], "seed": protocol["seed"], "runtime": runtime,
        "state_encoding_policy": STATE_ENCODING_POLICY, "reader_parameters_sha256": reader_hash,
        "source_complete_sha256": file_sha256(source / "complete.json"), "source_seal": seal,
        "export_script_sha256": script_hash, "declaration_sha256": file_sha256(declaration_path), "rows": metadata_rows})
    write_json(output / "complete.json", {"kind": "readout_content_export_v1", "records": len(errors),
        "max_replay_error": max(errors), "files": {name: file_sha256(output / name)
            for name in ("metadata.json", "states.safetensors", "replay.jsonl")}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--declaration", type=Path, required=True)
    args = parser.parse_args()
    check_repository()
    export(args.source, args.output, args.declaration)


if __name__ == "__main__":
    main()
