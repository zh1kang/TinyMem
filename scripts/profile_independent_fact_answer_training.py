"""Training-only compute profile for fresh recurrent answer-trained writers."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file, save_file

from scripts.fit_independent_fact_recurrent_training import validate_manifest
from scripts.profile_adapted_readout import write_json
from tinymem.memory.readout_interface import ReadoutBridge
from tinymem.research.adapted_readout import configure_read_adapter
from tinymem.research.independent_fact_data import build_worlds, encode_worlds
from tinymem.research.independent_fact_recurrent_training import pack_recurrent_batch
from tinymem.research.independent_fact_updates import new_update_writer
from tinymem.research.readout_experiment import _reader_hash
from tinymem.research.study_runtime import (
    REPOSITORY, allocation_metrics, execution_record, prepare_device, synchronize,
)
from tinymem.research.update_protocol import file_sha256, load_shared_reader, shared_reader_identity


SETTINGS = {
    'purpose': 'training_execution_and_compute_only',
    'seed': 1337, 'schedule_seed': 1337, 'batch_size': 16,
    'steps_per_arm': 5, 'warmup_steps': 2, 'query_microbatch': 6,
    'coordinate_weights': [0.0, 1.0], 'reader_fold': 'even',
    'answer_weight': 1.0, 'initial_loss_weight': 0.5, 'update_loss_weight': 0.5,
    'questions_per_state': 6, 'horizon': 4, 'persistent_bytes': 66,
    'optimizer': {'lr': 0.001, 'weight_decay': 0.01, 'betas': [0.9, 0.999], 'eps': 1e-8},
    'clip_norm': 1.0, 'evaluation_scored': False, 'checkpoint_reuse': False,
}


def verify_profile_inputs(declaration):
    if declaration['settings'] != SETTINGS or declaration['reader'] != shared_reader_identity():
        raise ValueError('profile settings or reader identity differ')
    for name, digest in declaration['file_sha256'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('profile inputs must be repository-relative paths')
        if file_sha256(REPOSITORY / path) != digest:
            raise ValueError('profile input or source changed: ' + name)
    paths = declaration['paths']
    required = {
        paths['inputs'], paths['manifest'], paths['placement_report'],
        paths['initial_report'], paths['initial_declaration'], paths['initial_complete'],
        paths['initial_proof'], paths['initial_protocol'],
        paths['history_features'], paths['event_features'], paths['reader_adapter'], paths['bridge'],
        'scripts/profile_independent_fact_answer_training.py',
        'src/tinymem/research/independent_fact_answer_training.py',
        'src/tinymem/research/independent_fact_recurrent_training.py',
        'scripts/fit_independent_fact_recurrent_training.py',
        'src/tinymem/research/prefix_reader.py',
        'src/tinymem/memory/readout_interface.py',
        'src/tinymem/memory/query_pool_slots.py',
    }
    if not required.issubset(declaration['file_sha256']):
        raise ValueError('profile declaration omits a consumed file')


def verify_feature_provenance(paths, reader_identity, initial_reader_hash):
    report = json.loads(paths['initial_report'].read_text())
    declaration = json.loads(paths['initial_declaration'].read_text())
    seal = json.loads(paths['initial_complete'].read_text())
    proof = json.loads(paths['initial_proof'].read_text())
    protocol = json.loads(paths['initial_protocol'].read_text())
    if (report['status'] != 'complete' or proof['verified'] is not True
            or proof['report_sha256'] != file_sha256(paths['initial_report'])
            or proof['declaration_sha256'] != file_sha256(paths['initial_declaration'])
            or protocol['declaration_sha256'] != proof['declaration_sha256']
            or declaration['reader'] != reader_identity
            or declaration['input_sha256'] != file_sha256(paths['inputs'])
            or declaration['placement_report_sha256'] != file_sha256(paths['placement_report'])
            or report['feature_reader_sha256'] != initial_reader_hash):
        raise ValueError('feature provenance does not match the original frozen feature reader')
    for field, filename in (
        ('history_features', 'history_features.safetensors'),
        ('event_features', 'event_features.safetensors'),
        ('initial_report', 'report.json'), ('initial_protocol', 'protocol.json'),
    ):
        if file_sha256(paths[field]) != seal['files'][filename]:
            raise ValueError('feature files differ from their verified completion seal')


def main():
    from tinymem.research.independent_fact_answer_training import train_answer_recurrent_batch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declaration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    declaration_hash = file_sha256(args.declaration)
    d = json.loads(args.declaration.read_text())
    verify_profile_inputs(d)
    paths = {name: REPOSITORY / value for name, value in d['paths'].items()}
    manifest = json.loads(paths['manifest'].read_text())
    validate_manifest(manifest)
    order = list(range(256))
    random.Random(SETTINGS['schedule_seed']).shuffle(order)
    histories = {int(k): v for k, v in load_file(str(paths['history_features'])).items()}
    events = {tuple(map(int, k.split('.'))): v for k, v in load_file(str(paths['event_features'])).items()}
    ordered = [manifest['training_streams'][i] for i in order]
    batches = [pack_recurrent_batch(ordered[i:i + 16], histories, events) for i in range(0, 256, 16)]
    device = prepare_device('cuda')
    torch.set_num_threads(4)
    reader = load_shared_reader(d['reader'], device)
    prior = json.loads(paths['placement_report'].read_text())['fits']['separate_fact0']['even']
    if (file_sha256(paths['bridge']) != prior['final_checkpoint_sha256']['bridge']
            or file_sha256(paths['reader_adapter']) != prior['final_checkpoint_sha256']['adapter']):
        raise ValueError('selected files do not match the even placement checkpoint')
    verify_feature_provenance(paths, d['reader'], prior['reader_initial_sha256'])
    if _reader_hash(reader) != prior['reader_initial_sha256']:
        raise ValueError('original reader differs')
    set_peft_model_state_dict(reader.model, load_file(str(paths['reader_adapter']), device=str(device)), adapter_name='default')
    configure_read_adapter(reader, trainable=False)
    reader_hash = _reader_hash(reader)
    if reader_hash != prior['reader_final_sha256']:
        raise ValueError('frozen primary reader differs')
    bridge = ReadoutBridge(2048, 'affine').to(device)
    bridge.load_state_dict({k.removeprefix('bridge.'): v for k, v in load_file(str(paths['bridge']), device=str(device)).items()})
    bridge.requires_grad_(False).eval()
    bridge_before = {k: v.detach().clone() for k, v in bridge.state_dict().items()}
    rows = encode_worlds(reader, build_worlds())
    if json.loads(json.dumps([asdict(row) for row in rows])) != json.loads(paths['inputs'].read_text())['encodings']:
        raise ValueError('native prompt encodings differ')
    args.output.mkdir(parents=True)
    write_json(args.output / 'protocol.json', {
        'kind': 'independent_fact_answer_profile_v1', 'settings': SETTINGS,
        'declaration_sha256': declaration_hash, 'reader_sha256': reader_hash,
        'training_order': order, 'execution': execution_record(device),
    })
    summaries = {}
    initial_values = None
    for coordinate_weight in SETTINGS['coordinate_weights']:
        arm = 'answer_only' if coordinate_weight == 0 else 'answer_and_coordinates'
        directory = args.output / arm
        directory.mkdir()
        writer = new_update_writer(2048, SETTINGS['seed'])
        if initial_values is None:
            initial_values = {k: v.detach().clone() for k, v in writer.state_dict().items()}
        elif any(not torch.equal(v, initial_values[k]) for k, v in writer.state_dict().items()):
            raise ValueError('fresh paired writer initialization differs')
        save_file(writer.state_dict(), str(directory / 'initial.safetensors'))
        optimizer = torch.optim.AdamW(writer.parameters(), **SETTINGS['optimizer'])
        records = []
        with (directory / 'metrics.jsonl').open('x') as handle:
            for step in range(SETTINGS['steps_per_arm']):
                synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                metric = train_answer_recurrent_batch(
                    writer, batches[step], optimizer, reader=reader, bridge=bridge,
                    worlds=rows, coordinate_weight=coordinate_weight,
                    query_microbatch=SETTINGS['query_microbatch'],
                )
                if any(metric.get(key) != expected for key, expected in {
                    'history_examples': 16, 'event_examples': 64,
                    'writer_examples': 80, 'answer_sequences': 480,
                    'coordinate_weight': coordinate_weight,
                }.items()) or metric['answer_state_gradient_norm'] <= 0:
                    raise ValueError('profile did not execute the declared answer training workload')
                synchronize(device)
                metric.update(step=step + 1, warmup=step < SETTINGS['warmup_steps'],
                              seconds=time.perf_counter() - started, **allocation_metrics(device))
                handle.write(json.dumps(metric, sort_keys=True, allow_nan=False) + '\n')
                handle.flush()
                records.append(metric)
                print(json.dumps({'arm': arm, **metric}, allow_nan=False), flush=True)
        if all(torch.equal(v, initial_values[k]) for k, v in writer.state_dict().items()):
            raise ValueError('profile writer parameters did not change')
        save_file(writer.state_dict(), str(directory / 'final_disposable.safetensors'))
        measured = records[SETTINGS['warmup_steps']:]
        summaries[arm] = {
            'measured_steps': len(measured),
            'mean_step_seconds': sum(r['seconds'] for r in measured) / len(measured),
            'min_step_seconds': min(r['seconds'] for r in measured),
            'max_step_seconds': max(r['seconds'] for r in measured),
            'peak_allocated_bytes': max(r['cuda_peak_allocated_bytes'] for r in records),
            'initial_sha256': file_sha256(directory / 'initial.safetensors'),
            'final_disposable_sha256': file_sha256(directory / 'final_disposable.safetensors'),
            'optimizer_steps': len(records),
        }
        del optimizer, writer
    if _reader_hash(reader) != reader_hash or any(not torch.equal(v, bridge_before[k]) for k, v in bridge.state_dict().items()):
        raise ValueError('profile mutated fixed reader or bridge')
    if any(p.grad is not None for p in (*reader.model.parameters(), *bridge.parameters())):
        raise ValueError('frozen read path acquired gradients')
    verify_profile_inputs(d)
    if file_sha256(args.declaration) != declaration_hash:
        raise ValueError('profile declaration changed during execution')
    write_json(args.output / 'report.json', {
        'status': 'complete', 'purpose': SETTINGS['purpose'], 'arms': summaries,
        'reader_unchanged': True, 'bridge_unchanged': True, 'evaluation_scored': False,
        'total_optimizer_steps': 10, 'fresh_paired_initialization_verified': True,
        'full_training_seconds_at_400_steps_each': sum(r['mean_step_seconds'] * 400 for r in summaries.values()),
        'checkpoint_reuse': False,
    })
    files = {str(p.relative_to(args.output)): file_sha256(p) for p in args.output.rglob('*') if p.is_file()}
    write_json(args.output / 'complete.json', {'kind': 'independent_fact_answer_profile_complete_v1',
               'declaration_sha256': declaration_hash, 'files': files})


if __name__ == '__main__':
    main()
