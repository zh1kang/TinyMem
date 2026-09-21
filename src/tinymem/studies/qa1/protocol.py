"""Pinned development protocol for reading correct states from official QA1."""

from dataclasses import asdict
import json
from pathlib import Path
import random
import shutil

from tinymem.studies.qa1.data import load_training, story_id
from tinymem.studies.artifacts import file_hash, write_json
from tinymem.studies.delta.protocol import source_files

TRAIN_SHA256 = '749ea9f7c99070feb2d88c975a254417a0dcc8274add4435ae5ae24c7afc7e9d'
TRAIN_FILENAME = 'qa1_single-supporting-fact_train.txt'


def settings() -> dict:
    return {'purpose': 'official_qa1_training_split_readout_development',
            'source_revision': 'tasks_1-20_v1-2/en-10k/qa1',
            'validation_stories': 200, 'split_seed': 2026091601,
            'schedule_seed': 2026091602, 'epochs': 4, 'batch_size': 16,
            'reader_seeds': [5101, 5102, 5103], 'lora_rank': 8,
            'learning_rate': 0.001, 'weight_decay': 0.01, 'clip_norm': 1.0,
            'persistent_bytes': 258, 'device': 'cuda', 'max_new_tokens': 8,
            'state_code': {'matrix': [8, 8], 'beta': 0.75, 'value_scale': 0.5,
                           'key': 'training-vocabulary entity one-hot',
                           'value': 'training-vocabulary room one-hot',
                           'unknown': 'zero entity row', 'serialization': 'row-major two FP32 slots, both valid'},
            'checkpoint_selection': 'fixed_final', 'accuracy_exclusion': 'none',
            'evaluation': 'all development questions; official test not loaded',
            'controls': ['oracle', 'zero', 'donor', 'full_text', 'packed_explicit'],
            'full_text_reader': 'unadapted base; LoRA disabled',
            'scoring': 'greedy exact match after whitespace stripping and lowercasing',
            'zero_control': 'two valid slots with all values zero; same bridge and adapter',
            'donor': 'next different source story with same fact count in sorted development source IDs; no answer-based selection',
            'failure_policy': 'preserve failures; no automatic retry or seed replacement',
            'interpretation': 'new reader competence on privileged states, not learned-writer or official test performance'}


def schedule(training, spec: dict) -> list[list[list[str]]]:
    rng = random.Random(spec['schedule_seed'])
    result = []
    for _ in range(spec['epochs']):
        ids = [row.source_example_id for row in training]
        rng.shuffle(ids)
        result.append([ids[i:i + spec['batch_size']] for i in range(0, len(ids), spec['batch_size'])])
    return result


def prepare(root: Path, output: Path, source: Path, snapshot: dict) -> dict:
    spec = settings()
    vocabulary, training, validation = load_training(source, expected_sha256=TRAIN_SHA256)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source, output / TRAIN_FILENAME)
    hashes = {}
    for path in source_files(root):
        relative = str(path.relative_to(root))
        target = output / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[relative] = file_hash(target)
        if hashes[relative] != file_hash(path):
            raise ValueError('source changed during copy')
    protocol = {'kind': 'babi_qa1_oracle_readout_v1', 'settings': spec,
                'training_source_sha256': TRAIN_SHA256, 'source_sha256': hashes,
                'snapshot': snapshot, 'vocabulary': asdict(vocabulary),
                'training_ids': [e.source_example_id for e in training],
                'validation_ids': [e.source_example_id for e in validation],
                'training_stories': sorted({story_id(e) for e in training}),
                'validation_stories': sorted({story_id(e) for e in validation}),
                'schedule': schedule(training, spec),
                'cells': [{'index': i, 'seed': seed, 'bridge_seed': seed + 1,
                           'adapter_seed': seed + 10000, 'persistent_bytes': 258}
                          for i, seed in enumerate(spec['reader_seeds'])]}
    write_json(output / 'protocol.json', protocol)
    return protocol


def verify(study: Path, root: Path):
    protocol = json.loads((study / 'protocol.json').read_text())
    if protocol['kind'] != 'babi_qa1_oracle_readout_v1' or protocol['settings'] != settings():
        raise ValueError('QA1 development protocol differs')
    vocabulary, training, validation = load_training(study / TRAIN_FILENAME, expected_sha256=TRAIN_SHA256)
    if (protocol['training_source_sha256'] != TRAIN_SHA256
            or protocol['vocabulary'] != json.loads(json.dumps(asdict(vocabulary)))
            or protocol['training_ids'] != [e.source_example_id for e in training]
            or protocol['validation_ids'] != [e.source_example_id for e in validation]
            or protocol['training_stories'] != sorted({story_id(e) for e in training})
            or protocol['validation_stories'] != sorted({story_id(e) for e in validation})
            or protocol['schedule'] != schedule(training, settings())):
        raise ValueError('QA1 training source, vocabulary, split, or schedule differs')
    expected_cells = [{'index': i, 'seed': seed, 'bridge_seed': seed + 1,
                       'adapter_seed': seed + 10000, 'persistent_bytes': 258}
                      for i, seed in enumerate(settings()['reader_seeds'])]
    if protocol['cells'] != expected_cells:
        raise ValueError('reader seeds or initialization differ')
    for source_root in (root, study / 'source'):
        actual = {str(p.relative_to(source_root)): file_hash(p) for p in source_files(source_root)}
        if actual != protocol['source_sha256']:
            raise ValueError('execution source inventory differs')
    return protocol, vocabulary, training, validation
