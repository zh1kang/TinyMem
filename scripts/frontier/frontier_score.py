"""Fixed final test scoring, with serialized state and complete predictions."""
from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from frontier_prepare import sha, write
from frontier_run import features

from tinymem.reader.adapter import frozen_history_features
from tinymem.studies.frontier.baselines import SUPPORTED_CODECS, TextStore
from tinymem.studies.frontier.data import load_questions, write_records
from tinymem.studies.frontier.eval import (
    base_encoder,
    generate,
    normalized_exact_match,
)
from tinymem.studies.frontier.training import (
    encode_answer,
    rollout,
    text_vectors,
)


def batches(values, size):
    for start in range(0, len(values), size):
        yield start, values[start:start + size]


def summarize(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[(row['mode'], row['budget'], row['level'], row['task'])].append(row)
    return {f'{mode}:{budget}:{level}:{task}': {
        'questions': len(group), 'correct': sum(r['correct'] for r in group),
        'accuracy': sum(r['correct'] for r in group) / len(group),
        'normalized_accuracy': sum(r['normalized_correct'] for r in group) / len(group),
        'mean_payload_bytes': sum(r['payload_bytes'] for r in group) / len(group),
        'max_payload_bytes': max(r['payload_bytes'] for r in group),
    } for (mode, budget, level, task), group in groups.items()}


@torch.no_grad()
def evaluate(reader, memory, study: Path, protocol: dict, data: dict, cell: int) -> None:
    directory = study / 'evaluation' / str(cell)
    directory.mkdir(parents=True, exist_ok=False)
    cases = load_questions(study / 'inputs', 'test')
    if len(cases) != 5000 or any(sum(q.task == f'qa{i}' for q in cases) != 1000 for i in range(1, 6)):
        raise ValueError('the complete five-task official test inventory is required')
    native = [encode_answer(reader, q) for q in cases]
    declaration = protocol['cells'][cell]
    spec = protocol['evaluation']
    noise = tuple(data['noise']['test'])
    cache = features(study, protocol, data) if memory is not None else {}
    if memory is not None:
        missing = sorted(({r for q in cases for r in q.records} | set(noise)) - cache.keys())
        # Adapters are frozen and disabled. These are independent current records,
        # never a story-level encoder pass or a feature fit on test answers.
        # base_encoder restores the frozen adapted reader on exit.
        with base_encoder(reader):
            for record in missing:
                ids = reader.tokenizer.encode(record + '\n', add_special_tokens=False)
                if not 0 < len(ids) <= 64:
                    raise ValueError('a test record exceeds the fixed feature window')
                cache[record] = frozen_history_features(reader, ids)
    probe_indices = [i for task in ('qa1', 'qa2', 'qa3', 'qa4', 'qa5')
                     for i in [i for i, q in enumerate(cases) if q.task == task][:100]]
    rows = []
    started = time.perf_counter()
    with (directory / 'predictions.jsonl').open('x') as handle:
        def score(mode, budget, level, indices, vectors, payloads, donor_ids=None):
            predictions = generate(reader, [native[i] for i in indices], vectors,
                                   max_new_tokens=spec['max_new_tokens'])
            for position, (index, prediction, payload) in enumerate(zip(indices, predictions, payloads, strict=True)):
                q = cases[index]
                row = {'id': q.id, 'story': q.story, 'task': q.task, 'answer': q.answer,
                       'mode': mode, 'budget': budget, 'level': level, 'cell': cell,
                       'seed': declaration['seed'], 'fact_count': len(q.records),
                       'payload_bytes': len(payload), 'payload_sha256': hashlib.sha256(payload).hexdigest(),
                       'correct': prediction['prediction'].strip().lower() == q.answer.strip().lower(),
                       'normalized_correct': normalized_exact_match(prediction['prediction'], q.answer),
                       **prediction}
                if donor_ids is not None:
                    row['donor_id'] = donor_ids[position]
                if budget is not None and len(payload) > budget:
                    raise ValueError('retained memory exceeds the declared byte budget')
                rows.append(row)
                handle.write(json.dumps(row, allow_nan=False) + '\n')
            handle.flush()

        for level in spec['levels']:
            histories = [write_records(q, noise, level=level, seed=protocol['settings']['noise_seed']) for q in cases]
            if memory is not None:
                payloads = []
                for start, batch in batches(histories, spec['batch_size']):
                    state = rollout(memory, batch, cache)
                    payloads.extend(memory.pack(s[None]) for s in state)
                    if start % 400 == 0:
                        print(json.dumps({'stage': 'test-write', 'cell': cell, 'level': level, 'questions': start}), flush=True)
                (directory / f'states-{level}.bin').write_bytes(b''.join(payloads))
                # Reads use only restored bytes. The source text/cache is not a reader input.
                for start, batch in batches(payloads, spec['batch_size']):
                    restored = torch.cat([memory.unpack(payload) for payload in batch])
                    score('learned', memory.persistent_bytes, level, list(range(start, start + len(batch))),
                          list(memory.memory_vectors(restored)), batch)
                for mode in ('zero', 'donor'):
                    for _, indices in batches(probe_indices, spec['batch_size']):
                        donor_indices = [next(j for j, q in enumerate(cases)
                                              if q.task == cases[i].task and q.story != cases[i].story)
                                         for i in indices]
                        current = ([bytes(memory.persistent_bytes)] * len(indices) if mode == 'zero'
                                   else [payloads[j] for j in donor_indices])
                        restored = torch.cat([memory.unpack(payload) for payload in current])
                        score(mode, memory.persistent_bytes, level, indices,
                              list(memory.memory_vectors(restored)), current,
                              [cases[j].id for j in donor_indices] if mode == 'donor' else None)
            else:
                for budget in protocol['settings']['budgets']:
                    for codec in SUPPORTED_CODECS:
                        store = TextStore(budget, codec, tuple(data['dictionary']) if codec == 'dictionary_recent' else ())
                        for start, batch in batches(histories, spec['batch_size']):
                            payloads = []
                            for history in batch:
                                payload = store.empty()
                                for record in history:
                                    payload = store.update(payload, record)
                                payloads.append(payload)
                            score(codec, budget, level, list(range(start, start + len(batch))),
                                  text_vectors(reader, [store.decode(p) for p in payloads]), payloads)
                            if start % 400 == 0:
                                print(json.dumps({'stage': 'test-text', 'cell': cell, 'level': level,
                                                  'budget': budget, 'codec': codec, 'questions': start}), flush=True)
                if level == 0:
                    for start, batch in batches(histories, spec['batch_size']):
                        texts = ['\n'.join(history) for history in batch]
                        score('full_text', None, level, list(range(start, start + len(batch))),
                              text_vectors(reader, texts), [t.encode() for t in texts])
    report = {'results': summarize(rows), 'seconds': time.perf_counter() - started,
              'cell': cell, 'seed': declaration['seed'], 'complete_test_questions': len(cases),
              'normalization': 'primary strip+lower exact; normalized exact secondary',
              'training_checkpoint_sha256': sha(study / 'training' / str(cell) / 'checkpoint.safetensors'),
              'shared_dictionary_bytes': TextStore(64, 'dictionary_recent', tuple(data['dictionary'])).shared_dictionary_bytes(),
              'official_test_scored': True, 'checkpoint_selection': 'fixed_final'}
    write(directory / 'report.json', report)
    write(directory / 'complete.json', {'protocol_sha256': sha(study / 'protocol.json'), 'cell': cell,
                                       'files': {p.name: sha(p) for p in directory.iterdir() if p.is_file()}})
