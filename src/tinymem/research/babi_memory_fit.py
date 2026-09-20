"""Fixed-final bridge and LoRA fits for the official QA1 development split."""

import json
from pathlib import Path
import time

import torch
from safetensors.torch import save_file

from tinymem.research.babi_memory_training import encode_question, train_batch
from tinymem.research.delta_fact_fit import execution_record
from tinymem.research.delta_fact_profile import frozen_base_hash, write_json
from tinymem.research.delta_fact_protocol import cell_identity, seal_directory
from tinymem.research.oracle_fact_fit import checkpoint_tensors, freeze, new_readout
from tinymem.research.study_runtime import allocation_metrics, synchronize


def train_cell(reader, study: Path, protocol: dict, vocabulary, training, index: int) -> dict:
    spec, cell = protocol['settings'], protocol['cells'][index]
    if reader.model.device.type != spec['device']:
        raise ValueError('reader device differs from protocol')
    directory = study / 'training' / str(index)
    directory.mkdir(parents=True, exist_ok=False)
    unadapted_base = frozen_base_hash(reader)
    bridge, adapters = new_readout(reader, protocol, cell)
    base_before = frozen_base_hash(reader)
    runtime = execution_record(reader)
    optimizer = torch.optim.AdamW([*bridge.parameters(), *adapters],
                                 lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
    encoded = {e.source_example_id: encode_question(reader, e, vocabulary) for e in training}
    curves, step = [], 0
    with (directory / 'metrics.jsonl').open('x') as handle:
        for epoch, batches in enumerate(protocol['schedule'], 1):
            total, count = 0.0, 0
            for ids in batches:
                synchronize(reader.model.device)
                started = time.perf_counter()
                metric = train_batch(reader, bridge, tuple(encoded[key] for key in ids), optimizer,
                                     adapters=adapters, vocabulary=vocabulary, clip_norm=spec['clip_norm'])
                synchronize(reader.model.device)
                step += 1
                metric.update(step=step, epoch=epoch, seconds=time.perf_counter() - started,
                              **allocation_metrics(reader.model.device))
                handle.write(json.dumps(metric, allow_nan=False) + '\n')
                handle.flush()
                total += metric['answer_ce'] * metric['questions']
                count += metric['questions']
                if step == 1 or step % 25 == 0:
                    print(json.dumps({'cell': index, **metric}), flush=True)
            curves.append({'epoch': epoch, 'training_question_mean_ce': total / count, 'questions': count})
            write_json(directory / 'curves.json', curves)
    freeze(reader, bridge)
    base_after = frozen_base_hash(reader)
    if base_before != base_after:
        raise ValueError('training changed frozen base weights')
    save_file(checkpoint_tensors(reader, bridge), str(directory / 'checkpoint.safetensors'))
    report = {'optimizer_steps': step, 'epochs': len(curves), 'runtime': runtime,
              'unadapted_base_sha256': unadapted_base, 'base_before_sha256': base_before,
              'base_after_sha256': base_after, 'official_test_scored': False,
              'checkpoint_selection': 'fixed_final', 'persistent_bytes': 258,
              'parameters': {'writer': 0, 'bridge': sum(p.numel() for p in bridge.parameters()),
                             'adapter': sum(p.numel() for p in adapters)}}
    write_json(directory / 'report.json', report)
    seal_directory(directory, cell_identity(study, protocol, index, 'training'))
    return report
