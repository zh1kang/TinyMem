"""Full-text BABILong transfer without evidence filtering or retained features."""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from frontier_prepare import sha, write

from tinymem.research.adapted_readout import frozen_history_features
from tinymem.research.storage_baselines import SUPPORTED_CODECS, TextStore
from tinymem.research.storage_frontier_data import StorageQuestion
from tinymem.research.storage_frontier_eval import (
    base_encoder,
    generate,
    normalized_exact_match,
)
from tinymem.research.storage_frontier_training import encode_answer, text_vectors


def chunks(text: str, tokenizer, limit: int = 64) -> tuple[str, ...]:
    """Split by whitespace and token count, preserving every source character."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError('transfer input must be nonempty text')
    units = re.findall(r'\s*\S+\s*', text)
    if ''.join(units) != text:
        raise ValueError('text segmentation lost source characters')
    result, current = [], ''
    for unit in units:
        if len(tokenizer.encode(current + unit + '\n', add_special_tokens=False)) <= limit:
            current += unit
            continue
        if current:
            result.append(current)
            current = ''
        while len(tokenizer.encode(unit + '\n', add_special_tokens=False)) > limit:
            # Very long words remain in the input; split them without truncation.
            end = min(len(unit) - 1, limit)
            while end and len(tokenizer.encode(unit[:end] + '\n', add_special_tokens=False)) > limit:
                end -= 1
            if end == 0:
                raise ValueError('one source character exceeds the token window')
            result.append(unit[:end])
            unit = unit[end:]
        current = unit
    if current:
        result.append(current)
    if ''.join(result) != text or any(not c for c in result):
        raise ValueError('transfer must preserve the complete source stream')
    return tuple(result)


@torch.no_grad()
def evaluate_transfer(reader, memory, study: Path, protocol: dict, data: dict, cell: int) -> None:
    directory = study / 'transfer' / str(cell)
    directory.mkdir(parents=True, exist_ok=False)
    spec = protocol['cells'][cell]
    stores = {(codec, budget): TextStore(budget, codec, tuple(data['dictionary'])
                                       if codec == 'dictionary_recent' else ())
              for budget in protocol['settings']['budgets'] for codec in SUPPORTED_CODECS}
    groups = defaultdict(list)
    started = time.perf_counter()
    with (directory / 'predictions.jsonl').open('x') as handle:
        for task in ('qa1', 'qa2', 'qa3', 'qa4', 'qa5'):
            path = study / 'inputs' / f'babilong-{task}-1k.json'
            examples = json.loads(path.read_text())
            if not isinstance(examples, list) or len(examples) != 100:
                raise ValueError('expected all 100 installed BABILong 1k examples per task')
            for index, example in enumerate(examples):
                if any(not isinstance(example.get(k), str) or not example[k].strip()
                       for k in ('input', 'question', 'target')):
                    raise ValueError('invalid raw BABILong record')
                records = chunks(example['input'], reader.tokenizer)
                identity = f'babilong:{task}:1k:{index}'
                question = StorageQuestion(identity, identity, task, 'test', records,
                                           example['question'], example['target'])
                native = encode_answer(reader, question)
                inputs = []
                if memory is not None:
                    payload = memory.pack(memory.empty(1))
                    with base_encoder(reader):
                        for record in records:
                            hidden = frozen_history_features(reader, reader.tokenizer.encode(
                                record + '\n', add_special_tokens=False)).to(reader.model.device)[None]
                            valid = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
                            state = memory(memory.unpack(payload), hidden, valid)
                            payload = memory.pack(state)
                    # Reading starts after the encoder features are no longer needed.
                    del hidden, state
                    inputs.append(('learned', memory.persistent_bytes, payload,
                                   memory.memory_vectors(memory.unpack(payload))[0]))
                else:
                    for (codec, budget), store in stores.items():
                        payload = store.empty()
                        for record in records:
                            payload = store.update(payload, record)
                        inputs.append((codec, budget, payload, text_vectors(reader, [store.decode(payload)])[0]))
                predictions = generate(reader, [native] * len(inputs), [r[3] for r in inputs],
                                       max_new_tokens=protocol['evaluation']['max_new_tokens'])
                for (mode, budget, payload, _), prediction in zip(inputs, predictions, strict=True):
                    if len(payload) > budget:
                        raise ValueError('transfer payload exceeds declared storage')
                    row = {'id': identity, 'task': task, 'cell': cell, 'seed': spec['seed'],
                           'mode': mode, 'budget': budget, 'payload_bytes': len(payload),
                           'payload_sha256': hashlib.sha256(payload).hexdigest(),
                           'input_sha256': hashlib.sha256(example['input'].encode()).hexdigest(),
                           'input_chars': len(example['input']), 'writes': len(records),
                           'answer': example['target'],
                           'correct': prediction['prediction'].strip().lower() == example['target'].strip().lower(),
                           'normalized_correct': normalized_exact_match(prediction['prediction'], example['target']),
                           **prediction}
                    handle.write(json.dumps(row, allow_nan=False) + '\n')
                    groups[f'{mode}:{budget}:{task}'].append(row['correct'])
                handle.flush()
                if index % 20 == 0:
                    print(json.dumps({'stage': 'transfer', 'cell': cell, 'task': task, 'examples': index}), flush=True)
    write(directory / 'report.json', {'results': {k: {'correct': sum(v), 'questions': len(v),
                                                       'accuracy': sum(v) / len(v)} for k, v in groups.items()},
                                      'seconds': time.perf_counter() - started,
                                      'claim': 'full raw text transfer, previously exposed public benchmark, no filtering'})
    write(directory / 'complete.json', {'protocol_sha256': sha(study / 'protocol.json'), 'cell': cell,
                                       'files': {p.name: sha(p) for p in directory.iterdir() if p.is_file()}})
