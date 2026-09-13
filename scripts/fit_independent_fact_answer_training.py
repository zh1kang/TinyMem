"""Fresh paired answer-trained writers with sealed recurrent evaluation."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import random
import shutil
import time

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file, save_file

from scripts.fit_independent_fact_recurrent_training import collect_states
from scripts.profile_adapted_readout import write_json
from scripts.profile_independent_fact_answer_training import verify_feature_provenance, verify_profile_inputs
from tinymem.memory.readout_interface import ReadoutBridge
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.adapted_readout import configure_read_adapter
from tinymem.research.independent_fact_answer_training import train_answer_recurrent_batch
from tinymem.research.independent_fact_data import build_worlds, encode_worlds
from tinymem.research.independent_fact_placement import placement_state
from tinymem.research.independent_fact_recurrent_training import pack_recurrent_batch
from tinymem.research.independent_fact_updates import new_update_writer
from tinymem.research.readout_experiment import _reader_hash
from tinymem.research.readout_read import read_state_answer
from tinymem.research.study_runtime import REPOSITORY, execution_record, prepare_device, synchronize, validate_execution
from tinymem.research.update_protocol import file_sha256, shared_reader_identity, load_shared_reader

ARMS = {'answer_only': 0, 'answer_and_coordinates': 1}
SETTINGS = {
    'arms': ARMS, 'seed': 1337, 'schedule_seed': 1337, 'steps_per_arm': 400,
    'batch_streams': 16, 'training_horizon': 4, 'questions_per_state': 6,
    'query_microbatch': 6, 'initial_loss_weight': 0.5, 'updated_loss_weight': 0.5,
    'optimizer': {'lr': .001, 'weight_decay': .01, 'betas': [.9, .999], 'eps': 1e-8},
    'clip_norm': 1.0, 'checkpoint': 'final_only', 'cpu_threads': 4,
    'training_reader': 'even', 'evaluation_readers': ['even', 'odd'],
    'evaluation_horizon': 16, 'new_evaluation_streams': 32, 'continuity_streams': 32,
    'states_per_model': 1168, 'normal_answers': 28032, 'swap_answers': 768,
    'constant_control_answers': 24, 'reference_replays': 192,
    'persistent_bytes': 66, 'max_new_tokens': 8,
    'hard_scientific_cutoff': False, 'automatic_followup': False,
}


def verify_inputs(d):
    if d['settings'] != SETTINGS or d['reader'] != shared_reader_identity():
        raise ValueError('full training settings or reader identity differ')
    for name, digest in d['file_sha256'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or file_sha256(REPOSITORY/path) != digest:
            raise ValueError('declared full-run source or input differs: ' + name)
    paths = {k: REPOSITORY/v for k, v in d['paths'].items()}
    required_sources = {
        'scripts/fit_independent_fact_answer_training.py',
        'src/tinymem/research/independent_fact_answer_protocol.py',
        'docs/independent_fact_answer_training_run.md',
        'artifacts/diagnostics/independent_fact_answer_training_20260911_v1/verify_training.py',
        'artifacts/diagnostics/independent_fact_answer_training_20260911_v1/fit.slurm',
    }
    if not (set(d['paths'].values()) | required_sources).issubset(d['file_sha256']):
        raise ValueError('consumed file missing from full-run declaration')
    profile = json.loads(paths['profile_declaration'].read_text())
    verify_profile_inputs(profile)
    if any(d['file_sha256'].get(k) != v for k, v in profile['file_sha256'].items()):
        raise ValueError('completed profile source/input identity changed')
    run_dir = REPOSITORY/'artifacts/diagnostics/independent_fact_answer_training_20260911_v1'
    if (d['parent_declaration_sha256'] != file_sha256(paths['profile_declaration'])
            or d['study_plan_sha256'] != file_sha256(REPOSITORY/'docs/independent_fact_answer_training_run.md')
            or d['launch_sha256'] != file_sha256(run_dir/'fit.slurm')
            or d['verifier_sha256'] != file_sha256(run_dir/'verify_training.py')):
        raise ValueError('full run plan, parent, verifier, or launch identity differs')
    complete = json.loads(paths['profile_complete'].read_text())
    if complete['declaration_sha256'] != file_sha256(paths['profile_declaration']):
        raise ValueError('profile completion and declaration differ')
    for field, filename in (
        ('profile_report', 'report.json'), ('profile_protocol', 'protocol.json'),
        ('profile_initial', 'answer_only/initial.safetensors'),
        ('profile_answer_only_metrics', 'answer_only/metrics.jsonl'),
        ('profile_answer_only_final', 'answer_only/final_disposable.safetensors'),
        ('profile_answer_and_coordinates_metrics', 'answer_and_coordinates/metrics.jsonl'),
        ('profile_answer_and_coordinates_final', 'answer_and_coordinates/final_disposable.safetensors'),
    ):
        if file_sha256(paths[field]) != complete['files'][filename]:
            raise ValueError('completed profile files differ')
    if json.loads(paths['profile_report'].read_text())['status'] != 'complete':
        raise ValueError('completed compute profile required')
    return paths, profile


def load_read_path(d, paths, fold, device):
    prior = json.loads(paths['placement_report'].read_text())['fits']['separate_fact0'][fold]
    reader = load_shared_reader(d['reader'], device)
    if _reader_hash(reader) != prior['reader_initial_sha256']:
        raise ValueError('original frozen reader differs')
    verify_feature_provenance(paths, d['reader'], prior['reader_initial_sha256'])
    adapter, checkpoint = paths[fold+'_adapter'], paths[fold+'_bridge']
    if (file_sha256(adapter) != prior['final_checkpoint_sha256']['adapter']
            or file_sha256(checkpoint) != prior['final_checkpoint_sha256']['bridge']):
        raise ValueError('selected reader and bridge do not match the declared fold')
    set_peft_model_state_dict(reader.model, load_file(str(adapter), device=str(device)), adapter_name='default')
    configure_read_adapter(reader, trainable=False)
    if _reader_hash(reader) != prior['reader_final_sha256']:
        raise ValueError('fixed adapted reader differs')
    bridge = ReadoutBridge(2048, 'affine').to(device)
    bridge.load_state_dict({k.removeprefix('bridge.'): v for k, v in load_file(str(checkpoint), device=str(device)).items()})
    bridge.requires_grad_(False).eval()
    rows = encode_worlds(reader, build_worlds())
    if json.loads(json.dumps([asdict(r) for r in rows])) != json.loads(paths['inputs'].read_text())['encodings']:
        raise ValueError('native token encodings differ')
    return reader, bridge, rows, prior['reader_final_sha256']


def check_fixed_read_path(reader, bridge, reader_hash, bridge_snapshot):
    if (_reader_hash(reader) != reader_hash
            or any(not torch.equal(v, bridge_snapshot[k]) for k, v in bridge.state_dict().items())
            or any(p.requires_grad or p.grad is not None for p in (*reader.model.parameters(), *bridge.parameters()))):
        raise ValueError('fixed read path changed')


def check_profile_prefix(metric, expected):
    fields = [k for k in expected if k not in ('step', 'warmup', 'seconds') and not k.startswith('cuda_')]
    if any(metric.get(k) != expected[k] for k in fields):
        raise ValueError('training prefix does not exactly replay the completed profile')


def main():
    from tinymem.research.independent_fact_answer_protocol import build_answer_manifest, state_catalog

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declaration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    declaration_hash = file_sha256(args.declaration)
    d = json.loads(args.declaration.read_text())
    paths, profile = verify_inputs(d)
    previous = json.loads(paths['previous_manifest'].read_text())
    manifest = json.loads(paths['manifest'].read_text())
    if manifest != build_answer_manifest(previous):
        raise ValueError('declared training or evaluation manifest differs')
    catalog = state_catalog(manifest)
    if len(catalog) != SETTINGS['states_per_model']:
        raise ValueError('state catalog coverage differs')
    device = prepare_device('cuda')
    torch.set_num_threads(4)
    execution = execution_record(device)
    validate_execution(execution, expected=json.loads(paths['profile_protocol'].read_text())['execution'])
    histories = {int(k): v for k, v in load_file(str(paths['history_features'])).items()}
    features = {tuple(map(int, k.split('.'))): v for k, v in load_file(str(paths['event_features'])).items()}
    order = list(range(256)); random.Random(SETTINGS['schedule_seed']).shuffle(order)
    ordered = [manifest['training_streams'][i] for i in order]
    batches = [pack_recurrent_batch(ordered[i:i+16], histories, features) for i in range(0, 256, 16)]
    args.output.mkdir(parents=True)
    write_json(args.output/'protocol.json', {'declaration_sha256': declaration_hash, 'settings': SETTINGS,
        'execution': execution, 'training_order': order, 'state_catalog': catalog})
    shutil.copyfile(paths['manifest'], args.output/'data_manifest.json')
    reader, bridge, rows, reader_hash = load_read_path(d, paths, 'even', device)
    bridge_snapshot = {k: v.clone() for k, v in bridge.state_dict().items()}
    initial_reference = load_file(str(paths['profile_initial']))
    combined = {'evaluation_streams': manifest['old_continuity_streams'] + manifest['evaluation_streams']}
    states = {}
    with (args.output/'metrics.jsonl').open('x') as handle:
        for arm, coordinate_weight in ARMS.items():
            writer = new_update_writer(2048, SETTINGS['seed'])
            if any(not torch.equal(v, initial_reference[k]) for k, v in writer.state_dict().items()):
                raise ValueError('fresh writer initialization differs from profiled initialization')
            save_file(writer.state_dict(), str(args.output/(arm+'_initial.safetensors')))
            optimizer = torch.optim.AdamW(writer.parameters(), **SETTINGS['optimizer'])
            prefix = [json.loads(line) for line in paths['profile_'+arm+'_metrics'].read_text().splitlines()]
            for step in range(SETTINGS['steps_per_arm']):
                synchronize(device); start = time.perf_counter()
                metric = train_answer_recurrent_batch(writer, batches[step % 16], optimizer,
                    reader=reader, bridge=bridge, worlds=rows, coordinate_weight=coordinate_weight, query_microbatch=6)
                if any(metric[k] != v for k, v in {'history_examples':16, 'event_examples':64,
                    'writer_examples':80, 'answer_sequences':480, 'coordinate_weight':coordinate_weight}.items()):
                    raise ValueError('training workload differs')
                if step < 5:
                    check_profile_prefix(metric, prefix[step])
                if step == 4:
                    profiled = load_file(str(paths['profile_'+arm+'_final']))
                    if any(not torch.equal(v, profiled[k]) for k, v in writer.state_dict().items()):
                        raise ValueError('profile prefix checkpoint replay differs')
                synchronize(device)
                metric.update(arm=arm, step=step+1, batch=step % 16, seconds=time.perf_counter()-start)
                handle.write(json.dumps(metric, allow_nan=False)+'\n'); handle.flush()
                if (step+1) % 10 == 0:
                    print(json.dumps(metric, allow_nan=False), flush=True)
            writer.zero_grad(set_to_none=True); writer.requires_grad_(False).eval()
            checkpoint = args.output/(arm+'_final.safetensors')
            save_file(writer.state_dict(), str(checkpoint))
            restored = new_update_writer(2048, SETTINGS['seed'])
            restored.load_state_dict(load_file(str(checkpoint)))
            restored.requires_grad_(False).eval()
            if any(not torch.equal(v, restored.state_dict()[k]) for k, v in writer.state_dict().items()):
                raise ValueError('final writer reload differs')
            model_states = collect_states(restored, histories, features, combined)
            if set(model_states) != set(catalog):
                raise ValueError('written state IDs differ')
            states.update({arm+'/'+k:v for k,v in model_states.items()})
            check_fixed_read_path(reader, bridge, reader_hash, bridge_snapshot)
            del writer, restored, optimizer, model_states
    save_file({f'{key}.{part}': getattr(state, part) for key, state in states.items()
               for part in ('values', 'valid')}, str(args.output/'written_states.safetensors'))
    frozen = {p.name: file_sha256(p) for p in args.output.iterdir()}
    del reader, bridge, rows, bridge_snapshot, batches, histories, features
    gc.collect(); torch.cuda.empty_cache()
    counts = {'normal': 0, 'swap': 0, 'constant': 0, 'reference': 0}
    with (args.output/'predictions.jsonl').open('x') as predictions, (args.output/'controls.jsonl').open('x') as controls, (args.output/'reference_replays.jsonl').open('x') as replays:
        for fold in SETTINGS['evaluation_readers']:
            reader, bridge, rows, reader_hash = load_read_path(d, paths, fold, device)
            bridge_snapshot = {k: v.clone() for k,v in bridge.state_dict().items()}
            def read(state, query, *, reader=reader, bridge=bridge, rows=rows):
                return read_state_answer(reader, bridge, state, torch.tensor(rows[0].before_ids, device=device),
                    torch.tensor(rows[0].queries[query].after_ids, device=device), max_new_tokens=8)
            for old in map(json.loads, paths[fold+'_predictions'].read_text().splitlines()):
                if old['condition'] != 'normal':
                    continue
                actual = read(placement_state(old['code'], device, 'separate_fact0'), old['query_index'])
                if any(actual[k] != old[k] for k in actual):
                    raise ValueError('exact-state reference replay differs')
                replays.write(json.dumps({'fold':fold, 'code':old['code'], 'query_index':old['query_index'], **actual})+'\n')
                counts['reference'] += 1
            payload = load_file(str(args.output/'written_states.safetensors'), device=str(device))
            def stored(key, *, payload=payload):
                return LatentSlotState(payload[key+'.values'], payload[key+'.valid'])
            for arm in ARMS:
                for name in catalog:
                    for query in range(6):
                        actual = read(stored(arm+'/'+name), query)
                        predictions.write(json.dumps({'model':arm, 'fold':fold, 'state_id':name, 'query_index':query, **actual})+'\n')
                        counts['normal'] += 1
                predictions.flush()
                for stream in manifest['evaluation_streams']:
                    name = stream['id']+':step-16'
                    donor = f'fresh-{stream["initial_code"] ^ 15:02d}-{stream["id"].rsplit("-",1)[1]}:step-16'
                    if catalog[name]['code'] ^ catalog[donor]['code'] != 15:
                        raise ValueError('swap donor does not change all four answers')
                    for query in range(6):
                        actual = read(stored(arm+'/'+donor), query)
                        controls.write(json.dumps({'model':arm, 'fold':fold, 'condition':'swap', 'state_id':name,
                            'donor_state_id':donor, 'query_index':query, **actual})+'\n')
                        counts['swap'] += 1
                controls.flush()
                print(json.dumps({'phase':'evaluation', 'fold':fold, 'arm':arm, **counts}), flush=True)
            for condition in ('zero', 'no_memory'):
                state = LatentSlotState(torch.zeros(1,2,8,device=device), torch.full((1,2), condition=='zero', dtype=torch.bool, device=device))
                for query in range(6):
                    controls.write(json.dumps({'fold':fold, 'condition':condition, 'query_index':query, **read(state,query)})+'\n')
                    counts['constant'] += 1
            check_fixed_read_path(reader, bridge, reader_hash, bridge_snapshot)
            if any(not torch.equal(value.cpu(), getattr(states[key.rsplit('.',1)[0]], key.rsplit('.',1)[1])) for key,value in payload.items()):
                raise ValueError('serialized state changed during reads')
            del read, stored, reader, bridge, rows, payload, state, bridge_snapshot
            gc.collect(); torch.cuda.empty_cache()
    if counts != {'normal':28032, 'swap':768, 'constant':24, 'reference':192}:
        raise ValueError('final evaluation coverage differs')
    verify_inputs(d)
    validate_execution(execution_record(device), expected=execution)
    if file_sha256(args.declaration) != declaration_hash or any(file_sha256(args.output/name) != digest for name,digest in frozen.items()):
        raise ValueError('declared inputs or frozen outputs changed')
    write_json(args.output/'report.json', {'status':'complete', 'training_steps':800, 'teacher_forced_sequences':384000,
        'states':2336, 'counts':counts, 'reader_unchanged':True, 'bridge_unchanged':True,
        'profile_prefix_steps_replayed':10, 'profile_checkpoints_replayed':2,
        'fresh_initialization_verified':True, 'checkpoint_reload_verified':True, 'hard_scientific_cutoff':False})
    write_json(args.output/'complete.json', {'kind':'independent_fact_answer_training_complete_v1',
        'declaration_sha256':declaration_hash, 'files':{p.name:file_sha256(p) for p in args.output.iterdir()}})


if __name__ == '__main__':
    main()
