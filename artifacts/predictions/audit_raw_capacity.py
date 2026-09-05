#!/usr/bin/env python3
"""Training-only lossless coding and retained-evidence audit, without a model."""

import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from tinymem.data.symbolic_world import MOVEMENT_SEPARATORS, parse_qa1_movement
from tinymem.memory.fingerprint_facts import FingerprintFactRetention, entity_fingerprint
from tinymem.memory.latest_fact_tokens import append_latest_fact_sentence
from tinymem.memory.template_facts import TemplateFactRetention
from tinymem.memory.vocabulary_tokens import VocabularyTokenRetention


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


root = Path("artifacts/predictions")
output = root / "raw_capacity_audit_20260905"
old_path = root / "native_memory_pilot_20260904/data_manifest.json"
old = json.loads(old_path.read_text())["train"]
model_dir = Path("data/raw/pretrained/qwen3-1.7b")
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
vocab_size = 151936
vocabulary = sorted({token for row in old for token in tokenizer.encode(row["case"]["context"] + "\n\n", add_special_tokens=False)})
assert len(vocabulary) == 21
packing = VocabularyTokenRetention(19, vocab_size, vocabulary)
updates, sentences, maximum_tokens = 0, set(), 0
for row in old:
    retained = {}
    state = packing.empty(1, device="cpu")
    for sentence in row["case"]["context"].splitlines():
        person = parse_qa1_movement(sentence, 1)[0]
        retained.pop(person, None)
        retained[person] = sentence
        text = "\n".join(retained.values()) + "\n\n"
        expected = tokenizer.encode(text, add_special_tokens=False)
        maximum_tokens = max(maximum_tokens, len(expected))
        assert set(expected) <= set(vocabulary)
        state = append_latest_fact_sentence(packing, tokenizer, state, sentence)
        ids, valid = packing.materialize(state, pad_id=0)
        assert ids[0, valid[0]].tolist() == expected
        assert state.nbytes == 19
        sentences.add(sentence)
        updates += 1
assert (updates, maximum_tokens, len(sentences)) == (2354, 29, 120)
assert sum(len(sentence.encode()) for sentence in sentences) == 3484
data = root / "opaque_qa1_data_20260905"
data_protocol = json.loads((data / "protocol.json").read_text())
assert digest(data / "train.json") == data_protocol["data_sha256"]["train.json"]
worlds = json.loads((data / "train.json").read_text())
new_vocabulary = sorted({token for row in worlds for token in tokenizer.encode(row["opaque"]["queries"][0]["context"] + "\n\n", add_special_tokens=False)})
new_packing = VocabularyTokenRetention(66, vocab_size, new_vocabulary)
template = TemplateFactRetention(66)
fingerprint = FingerprintFactRetention(66)
records = []
for row in worlds:
    world = row["opaque"]
    raw_state, template_state, fingerprint_state = new_packing.empty(1, device="cpu"), template.empty(), fingerprint.empty()
    latest = {}
    for sentence in world["queries"][0]["context"].splitlines():
        person = parse_qa1_movement(sentence, 1)[0]
        latest[person] = sentence
        raw_state = append_latest_fact_sentence(new_packing, tokenizer, raw_state, sentence)
        template_state = template.append_sentence(template_state, sentence)
        fingerprint_state = fingerprint.append_sentence(fingerprint_state, sentence)
    ids, valid = new_packing.materialize(raw_state, pad_id=0)
    raw_text = tokenizer.decode(ids[0, valid[0]].tolist(), skip_special_tokens=False)
    raw_lines = raw_text.splitlines()
    template_lines = template.sentences(template_state)
    assert all(sentence in latest.values() for sentence in template_lines)
    assert all(sentence in latest.values() for sentence in raw_lines if sentence)
    known = [query["question"].removeprefix("Where is ").removesuffix("?") for query in world["queries"][:8]]
    missing = world["entities"][-1]
    keys = [entity_fingerprint(name) for name in known]
    records.append({"world_id": world["world_id"], "vocabulary_retained_facts": len([line for line in raw_lines if line]),
                    "template_retained_facts": len(template_lines), "fingerprint_known_collisions": len(keys) - len(set(keys)),
                    "fingerprint_absent_collision": entity_fingerprint(missing) in keys,
                    "fingerprint_known_correct": sum(fingerprint.lookup(fingerprint_state, name) == query["answer"]
                                                     for name, query in zip(known, world["queries"][:8], strict=True)),
                    "fingerprint_absent_correct": fingerprint.lookup(fingerprint_state, missing) == "unknown",
                    "native_history_tokens": len(tokenizer.encode(world["queries"][0]["context"] + "\n\n", add_special_tokens=False))})
result = {
    "claim": "training-only_coding_and_symbolic_coverage_not_Qwen_answer_accuracy", "source_sha256": {
        str(path): digest(path) for path in (Path(__file__), old_path, data / "protocol.json", data / "train.json",
            model_dir / "tokenizer.json", model_dir / "tokenizer_config.json", Path("src/tinymem/memory/vocabulary_tokens.py"),
            Path("src/tinymem/memory/latest_fact_tokens.py"), Path("src/tinymem/memory/template_facts.py"), Path("src/tinymem/memory/fingerprint_facts.py"))},
    "original_qa1": {"histories": len(old), "updates": updates, "vocabulary_ids": vocabulary,
                     "shared_dictionary_serialized_bytes": packing.dictionary_serialized_bytes,
                     "payload_bytes": 19, "largest_token_count": maximum_tokens,
                     "exact_sentence_types": len(sentences), "sentence_table_utf8_bytes_without_offsets": 3484,
                     "analytic_four_sentence_payload_bits": 31, "sentence_dictionary_codec_implemented": False},
    "opaque_train": {"worlds": len(worlds), "vocabulary_size": len(new_vocabulary),
                     "vocabulary_shared_serialized_bytes": new_packing.dictionary_serialized_bytes,
                     "template_shared_grammar_json_bytes": template.dictionary_serialized_bytes,
                     "payload_bytes_each": 66, "template_opaque_entry_bits": 87, "template_count_bits": template.length_bits,
                     "mean_vocabulary_retained_facts": sum(row["vocabulary_retained_facts"] for row in records) / len(records),
                     "mean_template_retained_facts": sum(row["template_retained_facts"] for row in records) / len(records),
                     "fingerprint_known_collisions": sum(row["fingerprint_known_collisions"] for row in records),
                     "fingerprint_absent_collisions": sum(row["fingerprint_absent_collision"] for row in records),
                     "fingerprint_known_correct": sum(row["fingerprint_known_correct"] for row in records),
                     "fingerprint_absent_correct": sum(row["fingerprint_absent_correct"] for row in records)},
    "development_or_confirmation_used": False, "model_loaded": False,
}
output.mkdir(parents=True, exist_ok=False)
(output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
(output / "per_world.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
(output / "opaque_train_vocabulary.json").write_text(json.dumps(new_vocabulary) + "\n")
(output / "source_sentences.json").write_text(json.dumps(sorted(sentences), indent=2) + "\n")
print(json.dumps(result, indent=2))
