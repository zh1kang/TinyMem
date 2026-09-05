#!/usr/bin/env python3
"""Evaluate query-independent raw storage and labeled reference conditions."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from peft import PeftModel

from tinymem.data.reader_gate import ReaderCase
from tinymem.research.study_runtime import (
    attach_execution, check_repository, prepare_device, repository_path, sha256, synchronize,
)

from tinymem.evaluation.reader_gate import reader_exact_match, summarize_reader_predictions
from tinymem.memory.fingerprint_facts import FingerprintFactRetention
from tinymem.memory.latest_fact_tokens import append_latest_fact_sentence
from tinymem.memory.packed_tokens import PackedTokenRetention
from tinymem.memory.template_facts import TemplateFactRetention
from tinymem.memory.vocabulary_tokens import VocabularyTokenRetention
from tinymem.research.memory_prompt import encode_history_chunks, encode_memory_example
from tinymem.research.prefix_reader import generate_prefix_answer, prefix_answer_loss
from tinymem.research.pretrained import load_qwen_reader, verify_qwen_snapshot
from tinymem.utils.experiment import current_git_source_state
from tinymem.utils.seed import seed_everything


def digest(path):
    return sha256(path)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=repository_path, required=True)
parser.add_argument("--reader-gate", type=repository_path, required=True)
parser.add_argument("--vocabulary", type=repository_path, required=True)
parser.add_argument("--split", choices=("development", "confirmation"), required=True)
parser.add_argument("--output", type=repository_path, required=True)
parser.add_argument("--study-protocol", type=repository_path)
parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
args = parser.parse_args()
check_repository()
device = prepare_device(args.device)
gate = json.loads((args.reader_gate / "protocol.json").read_text())
gate_result = json.loads((args.reader_gate / "results.json").read_text())
assert gate_result["reader_accepted"] is True
data_protocol = json.loads((args.data / "protocol.json").read_text())
assert digest(args.data / "protocol.json") == gate["data_protocol_sha256"]
if args.split == "confirmation":
    assert args.study_protocol is not None
    frozen = json.loads(args.study_protocol.read_text())
    assert all(digest(Path(path)) == expected for path, expected in frozen["source_sha256"].items())
    assert frozen["data_protocol_sha256"] == digest(args.data / "protocol.json")
    assert frozen["reader_gate_results_sha256"] == digest(args.reader_gate / "results.json")
    assert frozen["reader_gate_protocol_sha256"] == digest(args.reader_gate / "protocol.json")
    assert frozen["vocabulary_sha256"] == digest(args.vocabulary)
    assert len(frozen["runs"]) == 6
    assert len({Path(run).resolve() for run in frozen["runs"]}) == 6
    completed_pairs = set()
    for run in frozen["runs"]:
        run = Path(run)
        result = json.loads((run / "results.json").read_text())
        assert result["profile"] is False
        trained = json.loads((run / "protocol.json").read_text())
        assert trained["study_protocol_sha256"] == digest(args.study_protocol)
        assert trained["steps"] == frozen["steps"]
        assert trained["data_protocol_sha256"] == frozen["data_protocol_sha256"]
        assert trained["reader_gate_results_sha256"] == frozen["reader_gate_results_sha256"]
        assert trained["reader_gate_protocol_sha256"] == frozen["reader_gate_protocol_sha256"]
        assert digest(run / f"step_{frozen['steps']:06d}.safetensors") == result["checkpoint_sha256"]
        assert digest(run / "metrics.jsonl") == result["metrics_sha256"]
        metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        assert [row["step"] for row in metrics] == list(range(1, frozen["steps"] + 1))
        assert result["forward_tokens"] == sum(row["forward_tokens"] for row in metrics)
        assert result["supervised_tokens"] == sum(row["supervised_tokens"] for row in metrics)
        completed_pairs.add((trained["writer_kind"], trained["seed"]))
    assert len(completed_pairs) == 6
    assert completed_pairs == {(writer, seed) for writer in frozen["writers"] for seed in frozen["seeds"]}
if gate["protocol"] == "opaque_qa1_reader_qualification_v1":
    adapter, adapter_hashes = repository_path(gate["adapter"]), gate["adapter_sha256"]
elif gate["protocol"] == "opaque_qa1_reader_continuation_v1":
    adapter, adapter_hashes = repository_path(gate_result["final_adapter"]), gate_result["adapter_sha256"]
else:
    raise ValueError("unsupported reader qualification protocol")
assert all(digest(adapter / name) == expected for name, expected in adapter_hashes.items())
snapshot = verify_qwen_snapshot(Path("data/raw/pretrained/qwen3-1.7b"))
assert snapshot == gate["snapshot"]
reader = load_qwen_reader(Path("data/raw/pretrained/qwen3-1.7b"), device=device, dtype=torch.bfloat16)
reader.model = PeftModel.from_pretrained(reader.model, adapter, is_trainable=False, local_files_only=True, use_safetensors=True)
reader.model.eval().requires_grad_(False)
seed_everything(1337)
vocabulary = json.loads(args.vocabulary.read_text())
train_path = args.data / "train.json"
assert digest(train_path) == data_protocol["data_sha256"]["train.json"]
train = json.loads(train_path.read_text())
expected = sorted({token for row in train for token in reader.tokenizer.encode(row["opaque"]["queries"][0]["context"] + "\n\n", add_special_tokens=False)})
assert vocabulary == expected
native = PackedTokenRetention(29, reader.model.config.vocab_size)
coded = VocabularyTokenRetention(66, reader.model.config.vocab_size, vocabulary)
template, fingerprint = TemplateFactRetention(66), FingerprintFactRetention(66)
assert native.payload_bytes == coded.payload_bytes == template.payload_bytes == fingerprint.payload_bytes == 66
assert digest(args.data / f"{args.split}.json") == data_protocol["data_sha256"][f"{args.split}.json"]
worlds = json.loads((args.data / f"{args.split}.json").read_text())
args.output.mkdir(parents=True, exist_ok=False)
methods = ("drop", "recent_native", "recent_vocabulary", "latest_vocabulary", "latest_template", "fingerprint", "full_history")
sources = [Path(__file__), *(Path("src/tinymem") / name for name in (
    "memory/packed_tokens.py", "memory/vocabulary_tokens.py", "memory/latest_fact_tokens.py", "memory/template_facts.py",
    "memory/fingerprint_facts.py", "research/prefix_reader.py", "research/memory_prompt.py", "research/pretrained.py",
    "data/opaque_qa1.py", "data/symbolic_world.py", "evaluation/reader_gate.py", "evaluation/longmemeval.py", "memory/storage.py"))]
protocol = {
    "protocol": "opaque_qa1_bounded_baselines_v1", "source": current_git_source_state(Path.cwd()).to_dict(),
    "snapshot": snapshot, "data": str(args.data), "data_protocol_sha256": digest(args.data / "protocol.json"),
    "split": args.split, "split_sha256": digest(args.data / f"{args.split}.json"),
    "reader_gate_results_sha256": digest(args.reader_gate / "results.json"), "adapter": str(adapter), "adapter_sha256": adapter_hashes,
    "vocabulary_sha256": digest(args.vocabulary), "vocabulary_training_only": True,
    "study_protocol_sha256": None if args.study_protocol is None else digest(args.study_protocol),
    "methods": methods, "variants": ["opaque", "short"], "bytes_each_bounded_method": 66,
    "unbounded_reference": "full_history", "zero_byte_reference": "drop",
    "handcrafted_non_Qwen_reference": "fingerprint", "source_sha256": {str(path): digest(path) for path in sources},
    "generation": {"batch_size": 1, "max_new_tokens": 8, "greedy": True, "cache": False, "padding": False},
    "vocabulary_dictionary_serialized_bytes": coded.dictionary_serialized_bytes,
    "template_grammar_serialized_bytes": template.dictionary_serialized_bytes,
    "write_boundary": "same_four_chunks_with_complete_visible_sentence_updates_inside_raw_policy",
    "query_aware_write": False, "measurements": "CPU_policy_write_and_selected_device_generation_wall_time_not_peak_memory",
    "read_seconds": "generation_only_excludes_decoding_materialization_embedding_expansion_and_answer_CE",
}
attach_execution(protocol, sources, device)
(args.output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
for path in sources:
    (args.output / path.name).write_bytes(path.read_bytes())
predictions, states_log = [], []
started = time.perf_counter()
with torch.inference_mode(), (args.output / "predictions.jsonl").open("x") as handle:
    for variant in ("opaque", "short"):
        for world_row in worlds:
            world = world_row[variant]
            cases = [ReaderCase(**item) for item in world["queries"]]
            examples = [encode_memory_example(reader, case) for case in cases]
            chunks = encode_history_chunks(reader, cases[0], world["chunks"])
            states = {"recent_native": native.empty(1, device="cpu"), "recent_vocabulary": coded.empty(1, device="cpu"),
                      "latest_vocabulary": coded.empty(1, device="cpu"), "latest_template": template.empty(), "fingerprint": fingerprint.empty()}
            write_seconds = {method: 0.0 for method in states}
            for text_chunk, token_chunk in zip(world["chunks"], chunks, strict=True):
                ids = torch.tensor([token_chunk], dtype=torch.long)
                valid = torch.ones_like(ids, dtype=torch.bool)
                for method, policy in (("recent_native", native), ("recent_vocabulary", coded)):
                    tick = time.perf_counter()
                    states[method] = policy.append(states[method], ids, valid)
                    write_seconds[method] += time.perf_counter() - tick
                for sentence in text_chunk.splitlines():
                    tick = time.perf_counter()
                    states["latest_vocabulary"] = append_latest_fact_sentence(coded, reader.tokenizer, states["latest_vocabulary"], sentence)
                    write_seconds["latest_vocabulary"] += time.perf_counter() - tick
                    for method, policy in (("latest_template", template), ("fingerprint", fingerprint)):
                        tick = time.perf_counter()
                        states[method] = policy.append_sentence(states[method], sentence)
                        write_seconds[method] += time.perf_counter() - tick
                assert all(state.nbytes == 66 for state in states.values())
            materialized = {"drop": [], "full_history": list(examples[0].history_ids),
                            "latest_template": reader.tokenizer.encode(template.text(states["latest_template"]), add_special_tokens=False)}
            for method, policy in (("recent_native", native), ("recent_vocabulary", coded), ("latest_vocabulary", coded)):
                ids, valid = policy.materialize(states[method], pad_id=0)
                materialized[method] = ids[0, valid[0]].tolist()
            states_log.append({"variant": variant, "world_id": world["world_id"], "payloads": {name: state.payload[0].tolist() for name, state in states.items()},
                               "write_seconds": write_seconds, "materialized_native_ids": materialized})
            for method in methods:
                memory = None if method == "fingerprint" else reader.model.get_input_embeddings()(torch.tensor(materialized[method], device=device, dtype=torch.long))
                for case, example in zip(cases, examples, strict=True):
                    synchronize(device)
                    tick = time.perf_counter()
                    if method == "fingerprint":
                        name = case.question.removeprefix("Where is ").removesuffix("?")
                        generated = {"prediction": fingerprint.lookup(states[method], name), "decoder": "handcrafted_lookup"}
                    else:
                        generated = generate_prefix_answer(reader, torch.tensor(example.before_ids, device=device), memory,
                                                           torch.tensor(example.after_ids, device=device), max_new_tokens=8)
                    synchronize(device)
                    read_seconds = time.perf_counter() - tick
                    answer_ce = None if method == "fingerprint" else float(prefix_answer_loss(
                        reader, torch.tensor(example.before_ids, device=device), memory,
                        torch.tensor(example.after_ids, device=device), torch.tensor(example.answer_ids, device=device)))
                    row = {"condition": f"{variant}:{method}", "world_id": world["world_id"], **asdict(case), **generated,
                           "read_seconds": read_seconds, "answer_ce": answer_ce,
                           "exact_match": reader_exact_match(generated["prediction"], case.answer, case.category)}
                    predictions.append(row)
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
            print(json.dumps({"variant": variant, "world_id": world["world_id"], "predictions": len(predictions)}), flush=True)
(args.output / "states.json").write_text(json.dumps(states_log, indent=2, sort_keys=True) + "\n")
summary = summarize_reader_predictions(predictions)
(args.output / "results.json").write_text(json.dumps({"by_condition": summary["by_condition"], "seconds": time.perf_counter() - started,
    "predictions_sha256": digest(args.output / "predictions.jsonl"), "states_sha256": digest(args.output / "states.json")}, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary["by_condition"], indent=2), flush=True)
