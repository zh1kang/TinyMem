"""Sealed-test native answers, explicit storage, and memory controls."""

from dataclasses import asdict
import json
from pathlib import Path

import torch

from tinymem.data.cases import ReaderCase
from tinymem.reader.prompt import reader_exact_match, reader_messages
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.studies.delta.data import ENTITIES, ROOM_PAIRS, parse_statement, replay
from tinymem.studies.delta.encoding import build_feature_cache, encode_episode
from tinymem.studies.delta.evaluation import collect_states, delta_geometry, predict_probe
from tinymem.studies.delta.fit import execution_record, load_state_records, load_trained, save_state_records, selected_episodes
from tinymem.studies.artifacts import _checkpoint_tensors, file_hash, frozen_base_hash, write_json
from tinymem.studies.delta.protocol import cell_identity, require_training_seal, seal_directory, verify_completion
from tinymem.studies.delta.readout import read_answer


def row_key(episode_id: str, position: int, entity: int) -> str:
    return f"{episode_id}/write{position}/entity{entity}"


def packed_facts(statements) -> torch.Tensor:
    """Four value bits and four known bits, updated from independently parsed text."""
    state = torch.zeros(1, dtype=torch.uint8)
    for statement in statements:
        parsed = parse_statement(statement.text)
        mask = 1 << parsed.entity
        state[0] = (int(state[0]) & ~mask) | (parsed.value << parsed.entity) | (1 << (parsed.entity + 4))
    return state


def reference_identity(study: Path) -> dict:
    return {"stage": "reference", "protocol_sha256": file_hash(study / "protocol.json"),
            "training_seal_sha256": file_hash(study / "training_sealed.json")}


def score_reference(reader, study: Path, protocol: dict, dataset) -> dict:
    require_training_seal(study, protocol)
    if reader.model.device.type != protocol["settings"]["device"]:
        raise ValueError("reference device differs from declaration")
    if (hasattr(reader.model, "peft_config") or any(p.requires_grad or p.grad is not None for p in reader.model.parameters())
            or any(module.training for module in reader.model.modules())):
        raise ValueError("reference requires the frozen unadapted reader")
    base_before = frozen_base_hash(reader)
    trained = json.loads((study / "training/0/report.json").read_text())
    if base_before != trained["unadapted_base_sha256"] or execution_record(reader) != trained["runtime"]:
        raise ValueError("reference base model or execution environment differs from training")
    directory = study / "reference"
    directory.mkdir(parents=True, exist_ok=False)
    episodes = selected_episodes(dataset.test, protocol["settings"])
    predictions = {}
    total = correct = 0
    with (directory / "predictions.jsonl").open("x") as handle:
        for episode in episodes:
            statements = (*episode.prefix, *episode.tail)
            for position in (8, 9, 16) if episode.tail else (8,):
                history = statements[:position]
                truth = replay(history)
                packed = packed_facts(history)
                if packed.untyped_storage().nbytes() != 1:
                    raise ValueError("explicit baseline storage differs")
                for entity in range(4):
                    value = (int(packed[0]) >> entity) & 1
                    if not (int(packed[0]) & (1 << (entity + 4))) or value != truth[entity]:
                        raise ValueError("explicit storage disagrees with replay")
                    answer = ROOM_PAIRS[entity][value]
                    key = row_key(episode.id, position, entity)
                    cache_key = (episode.prefix_id, episode.wording, position, entity,
                                 episode.condition if position != 8 else "prefix",
                                 episode.target if position != 8 else None)
                    if cache_key not in predictions:
                        case = ReaderCase(key, "update_known", episode.prefix_id,
                                          "\n".join(s.text for s in history), f"Where is {ENTITIES[entity]}?", answer)
                        prompt = reader.tokenizer.apply_chat_template(reader_messages(case, condition="full_context"),
                                  tokenize=False, add_generation_prompt=True, enable_thinking=False)
                        predictions[cache_key] = reader.generate([prompt], max_new_tokens=protocol["settings"]["max_new_tokens"])[0]
                    generated = predictions[cache_key]
                    row = {"key": key, "answer": answer, "explicit_bytes": 1, "explicit_correct": True,
                           "base_full_text": generated, "base_full_text_correct": reader_exact_match(generated["prediction"], answer, "update_known")}
                    correct += row["base_full_text_correct"]
                    total += 1
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
    if frozen_base_hash(reader) != base_before:
        raise ValueError("reference evaluation mutated the reader")
    report = {"cases_including_shared_prefix_duplicates": total, "base_full_text_correct": correct,
              "unique_generated_reads": len(predictions), "runtime": execution_record(reader),
              "base_sha256_unchanged": base_before}
    write_json(directory / "report.json", report)
    seal_directory(directory, reference_identity(study))
    return report


def score_cell(reader, study: Path, protocol: dict, dataset, index: int) -> dict:
    require_training_seal(study, protocol)
    verify_completion(study / "reference", reference_identity(study))
    spec, cell = protocol["settings"], protocol["cells"][index]
    directory = study / "evaluation" / str(index)
    directory.mkdir(parents=True, exist_ok=False)
    episodes = selected_episodes(dataset.test, spec)
    # Donors remain another prefix, even in the one-prefix implementation smoke.
    donor_episodes = tuple(e for e in dataset.test if e.condition == "no_write")
    features = build_feature_cache(reader, (*episodes, *donor_episodes))
    writer, bridge = load_trained(reader, protocol, cell, study / "training" / str(index) / "checkpoint.safetensors")
    base_before = frozen_base_hash(reader)
    trained = json.loads((study / "training" / str(index) / "report.json").read_text())
    if base_before != trained["base_after_sha256"] or execution_record(reader) != trained["runtime"]:
        raise ValueError("evaluation base model or execution environment differs from training")
    adapter_names = {name for name, _ in reader.model.named_parameters() if ".lora_A." in name or ".lora_B." in name}
    parameters_before = _checkpoint_tensors(reader, writer, bridge, adapter_names)
    records = collect_states(writer, episodes, features)
    save_state_records(records, directory / "states.safetensors")
    records = load_state_records(directory / "states.safetensors")
    donor_records = collect_states(writer, donor_episodes, features)
    by_prefix = {(r.prefix_id, r.wording): r for r in donor_records}
    prefix_order = sorted({r.prefix_id for r in donor_records})
    donors = {prefix: prefix_order[(i + 1) % len(prefix_order)] for i, prefix in enumerate(prefix_order)}
    if len(prefix_order) < 2:
        raise ValueError("memory donor control requires another prefix")
    probe = json.loads((study / "training" / str(index) / "probe.json").read_text())
    reference_rows = [json.loads(line) for line in (study / "reference" / "predictions.jsonl").read_text().splitlines()]
    references = {row["key"]: row for row in reference_rows}
    if len(references) != len(reference_rows):
        raise ValueError("reference rows contain duplicate identities")
    by_episode = {e.id: e for e in episodes}
    encodings = {}
    predictions = {}
    seen = set()
    with (directory / "predictions.jsonl").open("x") as handle:
        for record in records:
            episode = by_episode[record.episode_id]
            encoding_key = (record.prefix_id, record.wording)
            if encoding_key not in encodings:
                encodings[encoding_key] = encode_episode(reader, episode, features)
            example = encodings[encoding_key]
            state = LatentSlotState(record.values.to(reader.model.device), record.valid.to(reader.model.device))
            donor = by_prefix[(donors[record.prefix_id], record.wording)]
            donor_state = LatentSlotState(donor.values.to(reader.model.device), donor.valid.to(reader.model.device))
            zero = LatentSlotState(torch.zeros_like(state.values), torch.ones_like(state.valid))
            probe_bits = (predict_probe(probe, record.values.reshape(1, -1).numpy())[0] >= 0.5).tolist()
            for entity, query in enumerate(example.endpoints[0].queries):
                answer = ROOM_PAIRS[entity][record.truth[entity]]
                key = row_key(record.episode_id, record.after_write, entity)
                if key not in references or references[key]["answer"] != answer or key in seen:
                    raise ValueError("native reference and state replay cases disagree")
                seen.add(key)
                outputs = {}
                for mode, memory in (("memory", state), ("zero", zero), ("donor", donor_state)):
                    cache_key = ((record.prefix_id, record.wording,
                                  record.after_write, record.condition, record.target, entity)
                                 if mode == "memory" and record.after_write != 8
                                 else (record.prefix_id, record.wording, "prefix", entity, mode))
                    if cache_key not in predictions:
                        predictions[cache_key] = read_answer(reader, bridge, memory,
                            torch.tensor(example.before_ids, device=reader.model.device),
                            torch.tensor(query.after_ids, device=reader.model.device), max_new_tokens=spec["max_new_tokens"])
                    generated = predictions[cache_key]
                    outputs[mode] = {**generated, "correct": reader_exact_match(generated["prediction"], answer, query.category)}
                metadata = {k: v for k, v in asdict(record).items() if k not in ("values", "valid", "truth")}
                row = {**metadata, "key": key, "entity": entity, "answer": answer,
                       "scope": "all" if record.target is None else "target" if record.target == entity else "unspoken",
                       "truth_bit": record.truth[entity], "probe_bit": int(probe_bits[entity]),
                       "probe_correct": probe_bits[entity] == record.truth[entity], "reads": outputs,
                       "donor_prefix": donor.prefix_id, "donor_truth_agrees": donor.truth[entity] == record.truth[entity],
                       "base_full_text_correct": references[key]["base_full_text_correct"], "explicit_correct": True}
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
    if seen != set(references):
        raise ValueError("evaluation omitted reference cases")
    if frozen_base_hash(reader) != base_before:
        raise ValueError("evaluation mutated the reader")
    parameters_after = _checkpoint_tensors(reader, writer, bridge, adapter_names)
    if any(not torch.equal(value, parameters_after[name]) for name, value in parameters_before.items()):
        raise ValueError("evaluation mutated the trained writer, bridge, or adapter")
    geometry = None
    if cell["writer"] == "delta":
        geometry = {"all_wordings": delta_geometry(writer, features)}
        for wording in ("familiar", "heldout"):
            texts = {s.text for e in (*episodes, *donor_episodes) if e.wording == wording for s in (*e.prefix, *e.tail)}
            geometry[wording] = delta_geometry(writer, {text: features[text] for text in sorted(texts)})
    report = {"cases": len(seen), "unique_generated_reads": len(predictions), "runtime": execution_record(reader),
              "persistent_bytes": cell["persistent_bytes"], "base_sha256_unchanged": base_before,
              "key_geometry_test": geometry}
    write_json(directory / "report.json", report)
    identity = {**cell_identity(study, protocol, index, "evaluation"),
                "training_seal_sha256": file_hash(study / "training_sealed.json")}
    seal_directory(directory, identity)
    return report
