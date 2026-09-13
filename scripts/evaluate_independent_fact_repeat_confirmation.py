"""Evaluate the sealed correction checkpoints on fresh repetition programs."""

import argparse
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file, save_file

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
if str(REPOSITORY / 'src') not in sys.path:
    sys.path.insert(0, str(REPOSITORY / 'src'))

from scripts.evaluate_independent_fact_training_fit import READ_FIELDS
from scripts.fit_independent_fact_answer_training import check_fixed_read_path, load_read_path
from scripts.fit_independent_fact_correction_weight import require_declaration as require_correction
from scripts.profile_adapted_readout import write_json
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.independent_fact_placement import placement_state
from tinymem.research.independent_fact_repeat_confirmation import (
    build_manifest, collect_states, state_catalog,
)
from tinymem.research.independent_fact_updates import new_update_writer
from tinymem.research.readout_read import read_state_answer
from tinymem.research.study_runtime import execution_record, prepare_device, validate_execution
from tinymem.research.update_protocol import file_sha256


ARMS = ('uniform', 'correction_weighted')
SETTINGS = {
    'kind': 'independent_fact_repeat_confirmation_v1',
    'arms': list(ARMS),
    'seed': 20260912,
    'training_steps': 0,
    'streams': 64,
    'prefix_horizon': 8,
    'tail_horizon': 8,
    'cpu_threads': 4,
    'reader': 'even',
    'states_per_model': 3088,
    'states': 6176,
    'normal_answers': 37056,
    'no_write_answers': 768,
    'swap_answers': 768,
    'reference_answers': 96,
    'constant_answers': 12,
    'parent_initial_replays': 192,
    'total_reads': 38700,
    'persistent_bytes': 66,
    'max_new_tokens': 8,
    'step0_optimization': False,
    'hard_scientific_cutoff': False,
    'reference_mode': 'coordinate',
    'checkpoint_source': 'independent_correction_weight_results',
}


def _required_sources(d):
    return {
        d['correction_declaration'], d['correction_proof'], d['manifest'],
        str(Path(d['correction_results']) / 'complete.json'),
        'scripts/evaluate_independent_fact_repeat_confirmation.py',
        'src/tinymem/research/independent_fact_repeat_confirmation.py',
        'docs/independent_fact_repeat_confirmation.md',
        'artifacts/diagnostics/independent_fact_repeat_confirmation_20260912_v1/verify_repeat.py',
    }


def require_declaration(path):
    newd = json.loads(path.read_text())
    if (newd.get('settings') != SETTINGS or newd.get('walltime_minutes') != 120
            or newd.get('correction_declaration') != 'independent_correction_weight_declaration.json'
            or newd.get('correction_results') != 'independent_correction_weight_results'
            or newd.get('correction_proof') != 'independent_correction_weight_verification.json'
            or newd.get('manifest') != 'independent_repeat_confirmation_data.json'):
        raise ValueError('repeat declaration settings differ')
    hashes = newd.get('file_sha256', {})
    if not _required_sources(newd).issubset(hashes):
        raise ValueError('repeat provenance coverage differs')
    for name, digest in hashes.items():
        relative = Path(name)
        if (relative.is_absolute() or '..' in relative.parts
                or not (REPOSITORY / relative).is_file()
                or file_sha256(REPOSITORY / relative) != digest):
            raise ValueError('repeat source or input differs: ' + name)

    correction_path = REPOSITORY / newd['correction_declaration']
    parent_d, parent, paths, original_results, fit_results = require_correction(correction_path)
    correction_results = REPOSITORY / newd['correction_results']
    correction_seal_path = correction_results / 'complete.json'
    correction_proof_path = REPOSITORY / newd['correction_proof']
    if not correction_results.is_dir() or not correction_seal_path.is_file():
        raise ValueError('correction results are missing')
    correction_seal = json.loads(correction_seal_path.read_text())
    if correction_seal.get('declaration_sha256') != file_sha256(correction_path):
        raise ValueError('correction completion identity differs')
    if (not correction_seal.get('files')
            or any(Path(name).name != name
                   or file_sha256(correction_results / name) != digest
                   for name, digest in correction_seal['files'].items())):
        raise ValueError('correction result seal differs')
    if any(hashes.get(str((correction_results / name).relative_to(REPOSITORY))) != digest
           for name, digest in correction_seal['files'].items()):
        raise ValueError('correction result seal is not declared')
    proof = json.loads(correction_proof_path.read_text())
    if (proof.get('verified') is not True
            or proof.get('declaration_sha256') != file_sha256(correction_path)
            or proof.get('completion_sha256') != file_sha256(correction_seal_path)
            or proof.get('report_sha256') != file_sha256(correction_results / 'report.json')):
        raise ValueError('correction proof differs')
    return newd, parent, paths, original_results, fit_results


def _copy_state(state):
    return LatentSlotState(state.values.detach().clone().contiguous(),
                           state.valid.detach().clone().contiguous())


def _stored(payload, key):
    return LatentSlotState(payload[key + '.values'], payload[key + '.valid'])


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _baseline_records(results):
    result = {}
    seen = {}
    for row in _records(results / 'predictions.jsonl'):
        if row.get('model') not in ARMS or not row['state_id'].startswith('initial:'):
            continue
        key = row['state_id'], row['query_index']
        actual = {field: row[field] for field in READ_FIELDS}
        seen.setdefault(key, set()).add(row['model'])
        if key in result:
            if result[key] != actual:
                raise ValueError('correction initial replay overlap differs')
        else:
            result[key] = actual
    expected = {(f'initial:{code:02d}', query) for code in range(16) for query in range(6)}
    if set(result) != expected:
        raise ValueError('correction baseline answer coverage differs')
    if any(models != set(ARMS) for models in seen.values()):
        raise ValueError('correction baseline arm coverage differs')
    return result


def _write_rows(handle, model, state_id, query, actual, **extra):
    handle.write(json.dumps({'model': model, 'state_id': state_id,
                             'query_index': query, **extra, **actual}) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declaration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    declaration_hash = file_sha256(args.declaration)
    declaration, parent, paths, original_results, fit_results = require_declaration(args.declaration)
    correction_results = REPOSITORY / declaration['correction_results']
    correction_protocol = json.loads((correction_results / 'protocol.json').read_text())
    previous_manifest = json.loads(paths['manifest'].read_text())
    manifest = build_manifest(previous_manifest)
    manifest_path = REPOSITORY / declaration['manifest']
    if json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('repeat manifest differs from source construction')
    catalog = state_catalog(manifest)
    if len(manifest['streams']) != 64 or len(catalog) != SETTINGS['states_per_model']:
        raise ValueError('repeat state coverage differs')

    device = prepare_device('cuda')
    torch.set_num_threads(SETTINGS['cpu_threads'])
    execution = execution_record(device)
    validate_execution(execution, expected=correction_protocol['execution'])
    histories = {int(key): value for key, value in
                 load_file(str(paths['history_features'])).items()}
    features = {tuple(map(int, key.split('.'))): value for key, value in
                load_file(str(paths['event_features'])).items()}

    initializer_checkpoint = load_file(str(original_results / 'answer_only_final.safetensors'))
    initializer = new_update_writer(2048, 1337)
    initializer.load_state_dict(initializer_checkpoint)
    initializer.requires_grad_(False).eval().to('cpu')
    teacher = load_file(str(correction_results / 'teacher_states.safetensors'))
    teacher_values = teacher['values'].cpu()
    teacher_valid = teacher['valid'].cpu()
    if not teacher_values.dtype == torch.float32 or teacher_valid.dtype != torch.bool:
        raise ValueError('correction teacher dtype differs')
    parent_checkpoints = {}
    states = {}
    for arm in ARMS:
        checkpoint = correction_results / f'{arm}_final.safetensors'
        if not checkpoint.is_file():
            raise ValueError('correction checkpoint is missing: ' + arm)
        parent_checkpoints[arm] = file_sha256(checkpoint)
        updater = new_update_writer(2048, 1337)
        updater.load_state_dict(load_file(str(checkpoint)))
        updater.requires_grad_(False).eval().to('cpu')
        arm_states = collect_states(initializer, updater, histories, features, manifest)
        states.update({f'{arm}/{name}': _copy_state(value) for name, value in arm_states.items()})
        del updater
    if not torch.equal(teacher_values,
                       torch.cat([states[f'uniform/initial:{code:02d}'].values
                                  for code in range(16)])):
        raise ValueError('correction teacher differs from frozen initializer')
    if not torch.equal(teacher_valid,
                       torch.cat([states[f'uniform/initial:{code:02d}'].valid
                                  for code in range(16)])):
        raise ValueError('correction teacher validity differs')
    initializer_snapshot = {key: value.clone() for key, value in initializer.state_dict().items()}

    baseline = _baseline_records(correction_results)
    reader, bridge, rows, reader_hash = load_read_path(parent, paths, SETTINGS['reader'], device)
    bridge_snapshot = {key: value.clone() for key, value in bridge.state_dict().items()}
    args.output.mkdir(parents=True)
    save_file({f'{key}.{field}': getattr(state, field).cpu().contiguous()
               for key, state in states.items() for field in ('values', 'valid')},
              str(args.output / 'written_states.safetensors'))
    write_json(args.output / 'protocol.json', {
        'declaration_sha256': declaration_hash,
        'settings': SETTINGS,
        'execution': execution,
        'manifest_sha256': file_sha256(manifest_path),
        'state_catalog': catalog,
        'parent_updater_sha256': parent_checkpoints,
        'parent_initializer_sha256': file_sha256(original_results / 'answer_only_final.safetensors'),
        'correction_declaration_sha256': file_sha256(REPOSITORY / declaration['correction_declaration']),
        'inherited_state_weight': 0.25030934554044326,
        'optimizer_steps': 0,
        'read_path_loaded_before_evaluation': True,
    })

    payload = load_file(str(args.output / 'written_states.safetensors'), device=str(device))
    payload_snapshot = {key: value.clone() for key, value in payload.items()}
    before = torch.tensor(rows[0].before_ids, device=device)
    questions = [torch.tensor(query.after_ids, device=device) for query in rows[0].queries]

    def read(state, query):
        return read_state_answer(reader, bridge, state, before, questions[query],
                                 max_new_tokens=SETTINGS['max_new_tokens'])

    normal = {}
    counts = {'normal': 0, 'no_write': 0, 'swap': 0, 'constant': 0, 'reference': 0}
    with ((args.output / 'predictions.jsonl').open('x') as predictions,
          (args.output / 'controls.jsonl').open('x') as controls,
          (args.output / 'reference_replays.jsonl').open('x') as references):
        for arm in ARMS:
            for name in catalog:
                for query in range(6):
                    actual = read(_stored(payload, f'{arm}/{name}'), query)
                    normal[arm, name, query] = actual
                    _write_rows(predictions, arm, name, query, actual)
                    counts['normal'] += 1
                    if counts['normal'] % 512 == 0:
                        print(json.dumps({'normal_reads': counts['normal'],
                                          'normal_reads_expected': 37056}), flush=True)
                    if name.startswith('initial:') and any(
                            actual[field] != baseline[name, query][field] for field in READ_FIELDS):
                        raise ValueError('initial correction replay differs')
            for stream in manifest['streams']:
                endpoint = f"{stream['id']}/prefix/08"
                for query in range(6):
                    actual = read(_stored(payload, f'{arm}/{endpoint}'), query)
                    if any(actual[field] != normal[arm, endpoint, query][field]
                           for field in READ_FIELDS):
                        raise ValueError('no-write replay differs')
                    _write_rows(controls, arm, endpoint, query, actual, condition='no_write')
                    counts['no_write'] += 1
                endpoint = f"{stream['id']}/balanced/08"
                donor = f"repeat-{stream['initial_code'] ^ 15:02d}-{stream['id'].rsplit('-', 1)[1]}/balanced/08"
                for query in range(6):
                    actual = read(_stored(payload, f'{arm}/{donor}'), query)
                    _write_rows(controls, arm, endpoint, query, actual,
                                condition='swap', donor_state_id=donor)
                    counts['swap'] += 1
        for condition in ('zero', 'no_memory'):
            state = LatentSlotState(torch.zeros(1, 2, 8, device=device),
                                    torch.full((1, 2), condition == 'zero',
                                               dtype=torch.bool, device=device))
            for query in range(6):
                actual = read(state, query)
                _write_rows(controls, '', '', query, actual, condition=condition)
                counts['constant'] += 1
        for row in _records(original_results / 'reference_replays.jsonl'):
            if row.get('fold') != 'even':
                continue
            actual = read(placement_state(row['code'], device, 'separate_fact0'), row['query_index'])
            if any(actual[field] != row[field] for field in READ_FIELDS):
                raise ValueError('coordinate reference replay differs')
            references.write(json.dumps({'code': row['code'], 'query_index': row['query_index'], **actual}) + '\n')
            counts['reference'] += 1

    check_fixed_read_path(reader, bridge, reader_hash, bridge_snapshot)
    if any(not torch.equal(value, payload_snapshot[key]) for key, value in payload.items()):
        raise ValueError('state payload changed during reads')
    if any(not torch.equal(value, snapshot) for key, snapshot in initializer_snapshot.items()
           for value in [initializer.state_dict()[key]]):
        raise ValueError('initializer changed')
    expected_counts = {'normal': 37056, 'no_write': 768, 'swap': 768,
                       'constant': 12, 'reference': 96}
    if counts != expected_counts:
        raise ValueError('repeat answer coverage differs')
    require_declaration(args.declaration)
    if file_sha256(args.declaration) != declaration_hash:
        raise ValueError('repeat declaration changed')
    report = {
        'status': 'complete', 'training_steps': 0, 'states': 6176,
        'streams': 64, 'counts': counts, 'total_reads': 38700,
        'parent_initial_replays': 192, 'optimizer_steps': 0,
        'initializer_unchanged': True, 'reader_unchanged': True,
        'bridge_unchanged': True, 'state_unchanged_during_reads': True,
        'full_new_gpu_inference_independently_replayed': False,
    }
    write_json(args.output / 'report.json', report)
    write_json(args.output / 'complete.json', {
        'declaration_sha256': declaration_hash,
        'files': {path.name: file_sha256(path) for path in args.output.iterdir()},
    })


if __name__ == '__main__':
    main()
