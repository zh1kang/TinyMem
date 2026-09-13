"""Continue the sealed joint updater with uniform and correction-weighted CE."""

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

import torch
from safetensors.torch import load_file, save_file

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
if str(REPOSITORY / 'src') not in sys.path:
    sys.path.insert(0, str(REPOSITORY / 'src'))

from scripts.evaluate_independent_fact_training_fit import READ_FIELDS
from scripts.fit_independent_fact_answer_training import check_fixed_read_path, load_read_path
from scripts.fit_independent_fact_joint_update import require_declaration as require_joint
from scripts.fit_independent_fact_learned_state import SplitWriter
from scripts.fit_independent_fact_recurrent_training import collect_states, state_catalog
from scripts.profile_adapted_readout import write_json
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.independent_fact_correction_weight import train_correction_weight_batch
from tinymem.research.independent_fact_placement import placement_state
from tinymem.research.independent_fact_recurrent_training import pack_recurrent_batch
from tinymem.research.independent_fact_updates import build_update_cases, new_update_writer
from tinymem.research.readout_read import read_state_answer
from tinymem.research.study_runtime import execution_record, prepare_device, validate_execution
from tinymem.research.update_protocol import file_sha256


ARMS = ('uniform', 'correction_weighted')
CORRECTION_MULTIPLIERS = {'uniform': 1.0, 'correction_weighted': 2.0}
INHERITED_STATE_WEIGHT = 0.25030934554044326
SETTINGS = {
    'kind': 'independent_fact_correction_weight_v1',
    'arms': list(ARMS),
    'correction_multipliers': CORRECTION_MULTIPLIERS,
    'seed': 1337,
    'steps_per_arm': 400,
    'batch_streams': 16,
    'training_horizon': 4,
    'cpu_threads': 4,
    'optimizer': {'lr': .001, 'weight_decay': .01, 'betas': [.9, .999], 'eps': 1e-8},
    'reader': 'even',
    'states_per_model': 1680,
    'new_normal_answers': 20160,
    'trajectory_steps': [100, 200],
    'trajectory_states': 512,
    'trajectory_answers': 3072,
    'swap_answers': 384,
    'constant_answers': 12,
    'exact_replays': 96,
    'parent_initial_replays': 192,
    'reused_baseline_answers': 10080,
    'persistent_bytes': 66,
    'max_new_tokens': 8,
    'warm_start': 'answer_and_state_final.safetensors',
    'inherited_state_weight': INHERITED_STATE_WEIGHT,
    'hard_scientific_cutoff': False,
    'full_answer_ce_continuation_control': True,
}


def require_declaration(path):
    newd = json.loads(path.read_text())
    if (newd.get('settings') != SETTINGS or newd.get('walltime_minutes') != 600
            or newd.get('joint_declaration') != 'independent_joint_update_declaration.json'
            or newd.get('joint_results') != 'independent_joint_update_results'
            or newd.get('joint_proof') != 'independent_joint_update_verification.json'):
        raise ValueError('correction-weight declaration settings differ')
    for name, digest in newd['file_sha256'].items():
        relative = Path(name)
        if (relative.is_absolute() or '..' in relative.parts
                or file_sha256(REPOSITORY / relative) != digest):
            raise ValueError('correction-weight source or input differs: ' + name)

    joint_path = REPOSITORY / newd['joint_declaration']
    joint_d, parent, paths, parent_results, fit_results = require_joint(joint_path)
    required = {
        newd['joint_declaration'], newd['joint_proof'],
        str((REPOSITORY / newd['joint_results'] / 'complete.json').relative_to(REPOSITORY)),
        'scripts/fit_independent_fact_joint_update.py',
        'scripts/fit_independent_fact_correction_weight.py',
        'src/tinymem/research/independent_fact_joint_update.py',
        'src/tinymem/research/independent_fact_correction_weight.py',
        'docs/independent_fact_correction_weight.md',
        'artifacts/diagnostics/independent_fact_correction_weight_20260912_v1/verify_correction.py',
    }
    if not required.issubset(newd['file_sha256']):
        raise ValueError('correction-weight provenance coverage differs')
    joint_results = REPOSITORY / newd['joint_results']
    if not joint_results.is_dir():
        raise ValueError('joint parent results are missing')
    seal = json.loads((joint_results / 'complete.json').read_text())
    if seal.get('declaration_sha256') != file_sha256(joint_path):
        raise ValueError('joint parent completion identity differs')
    if (not seal.get('files')
            or any(Path(name).name != name or file_sha256(joint_results / name) != digest
                   for name, digest in seal['files'].items())):
        raise ValueError('joint parent result seal differs')
    if any(newd['file_sha256'].get(str((joint_results / name).relative_to(REPOSITORY))) != digest
           for name, digest in seal['files'].items()):
        raise ValueError('joint parent seal is not declared')
    proof = json.loads((REPOSITORY / newd['joint_proof']).read_text())
    if (proof.get('verified') is not True
            or proof.get('completion_sha256') != file_sha256(joint_results / 'complete.json')
            or proof.get('report_sha256') != file_sha256(joint_results / 'report.json')
            or proof.get('declaration_sha256') != file_sha256(joint_path)):
        raise ValueError('joint parent proof differs')
    calibration = json.loads((joint_results / 'calibration.json').read_text())
    if (calibration.get('optimizer_steps') != 0
            or calibration.get('batch_count') != 16
            or calibration.get('gradient_fraction') != .25
            or not math.isclose(calibration.get('state_weight', -1), INHERITED_STATE_WEIGHT,
                                rel_tol=0, abs_tol=1e-15)):
        raise ValueError('joint calibration weight differs')
    return newd, parent, paths, parent_results, fit_results


def baseline_records(results, catalog):
    result = {}
    for row in map(json.loads, (results / 'predictions.jsonl').read_text().splitlines()):
        if row.get('model') == 'answer_and_state' and row['state_id'] in catalog:
            key = row['state_id'], row['query_index']
            actual = {k: row[k] for k in READ_FIELDS}
            if key in result and result[key] != actual:
                raise ValueError('joint baseline overlap differs')
            result[key] = actual
    if set(result) != {(name, query) for name in catalog for query in range(6)}:
        raise ValueError('joint baseline answer coverage differs')
    return result


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _copy_state(state):
    return LatentSlotState(state.values.detach().clone().contiguous(),
                           state.valid.detach().clone().contiguous())


def primitive_states(writer, teacher, features):
    result = {}
    for case in build_update_cases():
        before = LatentSlotState(teacher.values[case.before_code:case.before_code + 1],
                                 teacher.valid[case.before_code:case.before_code + 1])
        hidden = features[case.target_fact, case.new_bit].unsqueeze(0)
        with torch.inference_mode():
            state = writer(before, hidden, torch.ones(1, hidden.shape[1], dtype=torch.bool))
        result['one:' + case.case_id] = _copy_state(state)
    if len(result) != 128:
        raise ValueError('trajectory primitive coverage differs')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declaration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    declaration_hash = file_sha256(args.declaration)
    d, parent, paths, parent_results, fit_results = require_declaration(args.declaration)
    joint_results = REPOSITORY / d['joint_results']
    joint_protocol = json.loads((joint_results / 'protocol.json').read_text())
    manifest = json.loads(paths['manifest'].read_text())
    evaluation = {'evaluation_streams': manifest['training_streams'] + manifest['evaluation_streams']}
    catalog = state_catalog(evaluation)
    if len(catalog) != SETTINGS['states_per_model']:
        raise ValueError('correction-weight state coverage differs')
    device = prepare_device('cuda')
    torch.set_num_threads(SETTINGS['cpu_threads'])
    execution = execution_record(device)
    validate_execution(execution, expected=joint_protocol['execution'])
    histories = {int(k): v for k, v in load_file(str(paths['history_features'])).items()}
    features = {tuple(map(int, k.split('.'))): v for k, v in load_file(str(paths['event_features'])).items()}
    initializer_checkpoint = load_file(str(parent_results / 'answer_only_final.safetensors'))
    joint_checkpoint = load_file(str(joint_results / SETTINGS['warm_start']))
    initializer = new_update_writer(2048, 1337)
    initializer.load_state_dict(initializer_checkpoint)
    initializer.requires_grad_(False).eval()
    joint_updater = new_update_writer(2048, 1337)
    joint_updater.load_state_dict(joint_checkpoint)
    joint_updater.requires_grad_(False).eval()
    joint_states = load_file(str(joint_results / 'written_states.safetensors'))
    baseline = collect_states(SplitWriter(initializer, joint_updater), histories, features, evaluation)
    for name, state in baseline.items():
        key = f'answer_and_state/{name}'
        if any(not torch.equal(getattr(state, field), joint_states[key + '.' + field])
               for field in ('values', 'valid')):
            raise ValueError('joint baseline state replay differs: ' + name)
    teacher = load_file(str(joint_results / 'teacher_states.safetensors'))
    teacher_state = LatentSlotState(teacher['values'], teacher['valid'])
    if (not torch.equal(teacher_state.values,
                        torch.cat([baseline[f'initial:{code:02d}'].values for code in range(16)]))
            or not torch.equal(teacher_state.valid,
                               torch.cat([baseline[f'initial:{code:02d}'].valid for code in range(16)]))):
        raise ValueError('joint teacher differs from original initializer')
    baseline_answers = baseline_records(joint_results, catalog)
    order = joint_protocol['training_order']
    expected_order = list(range(256))
    random.Random(SETTINGS['seed']).shuffle(expected_order)
    if order != expected_order:
        raise ValueError('joint training order differs')
    ordered = [manifest['training_streams'][index] for index in order]
    batches = [pack_recurrent_batch(ordered[i:i + 16], histories, features)
               for i in range(0, 256, 16)]
    reader, bridge, rows, reader_hash = load_read_path(parent, paths, SETTINGS['reader'], device)
    bridge_snapshot = {key: value.clone() for key, value in bridge.state_dict().items()}
    args.output.mkdir(parents=True)
    save_file({'values': teacher_state.values, 'valid': teacher_state.valid},
              str(args.output / 'teacher_states.safetensors'))
    write_json(args.output / 'protocol.json', {
        'declaration_sha256': declaration_hash, 'settings': SETTINGS,
        'execution': execution, 'training_order': order, 'state_catalog': catalog,
        'parent_updater_sha256': file_sha256(joint_results / SETTINGS['warm_start']),
        'parent_initializer_sha256': file_sha256(parent_results / 'answer_only_final.safetensors'),
        'joint_declaration_sha256': file_sha256(REPOSITORY / d['joint_declaration']),
        'inherited_state_weight': INHERITED_STATE_WEIGHT,
        'baseline_cpu_states_replayed': 1680, 'read_path_loaded_before_training': True,
    })
    with (args.output / 'baseline_predictions.jsonl').open('x') as handle:
        for (name, query), record in baseline_answers.items():
            handle.write(json.dumps({'state_id': name, 'query_index': query, **record}) + '\n')

    states = {}
    checkpoints = {}
    worlds = rows
    with (args.output / 'metrics.jsonl').open('x') as metrics_handle:
        for arm in ARMS:
            updater = new_update_writer(2048, 1337)
            updater.load_state_dict(joint_checkpoint)
            optimizer = torch.optim.AdamW(updater.parameters(), **SETTINGS['optimizer'])
            multiplier = CORRECTION_MULTIPLIERS[arm]
            for step in range(SETTINGS['steps_per_arm']):
                _sync(device)
                started = time.perf_counter()
                metric = train_correction_weight_batch(
                    updater, batches[step % len(batches)], optimizer,
                    reader=reader, bridge=bridge, worlds=worlds, targets=teacher_state,
                    state_weight=INHERITED_STATE_WEIGHT, correction_multiplier=multiplier,
                )
                _sync(device)
                metric.update(arm=arm, step=step + 1, batch=step % len(batches),
                              correction_multiplier=multiplier, seconds=time.perf_counter() - started)
                metrics_handle.write(json.dumps(metric, allow_nan=False) + '\n')
                metrics_handle.flush()
                if (step + 1) % 10 == 0:
                    print(json.dumps({'arm': arm, 'step': step + 1,
                                      'answer_ce': metric['answer_ce']}), flush=True)
                if step + 1 in SETTINGS['trajectory_steps']:
                    checkpoint = args.output / f'{arm}_step{step + 1}.safetensors'
                    save_file(updater.state_dict(), str(checkpoint))
                    checkpoints[arm, step + 1] = checkpoint
            updater.zero_grad(set_to_none=True)
            updater.requires_grad_(False).eval()
            final = args.output / f'{arm}_final.safetensors'
            save_file(updater.state_dict(), str(final))
            checkpoints[arm, 'final'] = final
            restored = new_update_writer(2048, 1337)
            restored.load_state_dict(load_file(str(final)))
            restored.requires_grad_(False).eval()
            states.update({f'{arm}/{name}': state for name, state in
                           collect_states(SplitWriter(initializer, restored), histories, features, evaluation).items()})
            if any(not torch.equal(value, initializer_checkpoint[key])
                   for key, value in initializer.state_dict().items()):
                raise ValueError('initializer changed')
            del optimizer, updater, restored
    trajectory_states = {}
    for arm in ARMS:
        for step in SETTINGS['trajectory_steps']:
            writer = new_update_writer(2048, 1337)
            writer.load_state_dict(load_file(str(checkpoints[arm, step])))
            writer.requires_grad_(False).eval()
            trajectory_states.update({f'{arm}/step{step}/{name}': state
                                      for name, state in primitive_states(writer, teacher_state, features).items()})
    save_file({f'{key}.{field}': getattr(state, field) for key, state in states.items()
               for field in ('values', 'valid')}, str(args.output / 'written_states.safetensors'))
    save_file({f'{key}.{field}': getattr(state, field) for key, state in trajectory_states.items()
               for field in ('values', 'valid')}, str(args.output / 'trajectory_states.safetensors'))

    payload = load_file(str(args.output / 'written_states.safetensors'), device=str(device))
    trajectory_payload = load_file(str(args.output / 'trajectory_states.safetensors'), device=str(device))
    payload_snapshot = {key: value.clone() for key, value in payload.items()}
    trajectory_snapshot = {key: value.clone() for key, value in trajectory_payload.items()}
    before = torch.tensor(rows[0].before_ids, device=device)
    questions = [torch.tensor(query.after_ids, device=device) for query in rows[0].queries]

    def read(state, query):
        return read_state_answer(reader, bridge, state, before, questions[query], max_new_tokens=8)

    def stored(key, source=payload):
        return LatentSlotState(source[key + '.values'], source[key + '.valid'])

    references = [row for row in map(json.loads, (parent_results / 'reference_replays.jsonl').read_text().splitlines())
                  if row['fold'] == 'even']
    with (args.output / 'reference_replays.jsonl').open('x') as handle:
        for old in references:
            actual = read(placement_state(old['code'], device, 'separate_fact0'), old['query_index'])
            if any(actual[field] != old[field] for field in READ_FIELDS):
                raise ValueError('exact reference replay differs')
            handle.write(json.dumps({'code': old['code'], 'query_index': old['query_index'], **actual}) + '\n')
    counts = {'normal': 0, 'trajectory': 0, 'swap': 0, 'constant': 0, 'reference': len(references)}
    with ((args.output / 'predictions.jsonl').open('x') as predictions,
          (args.output / 'trajectory_predictions.jsonl').open('x') as trajectories,
          (args.output / 'controls.jsonl').open('x') as controls):
        for arm in ARMS:
            for name in catalog:
                for query in range(6):
                    actual = read(stored(f'{arm}/{name}'), query)
                    if name.startswith('initial:') and any(actual[field] != baseline_answers[name, query][field]
                                                          for field in READ_FIELDS):
                        raise ValueError('frozen initial answer differs')
                    predictions.write(json.dumps({'model': arm, 'state_id': name,
                                                  'query_index': query, **actual}) + '\n')
                    counts['normal'] += 1
            for step in SETTINGS['trajectory_steps']:
                for key in trajectory_states:
                    if not key.startswith(f'{arm}/step{step}/'):
                        continue
                    name = key.split('/', 2)[2]
                    for query in range(6):
                        actual = read(stored(key, trajectory_payload), query)
                        trajectories.write(json.dumps({'model': arm, 'step': step, 'state_id': name,
                                                       'query_index': query, **actual}) + '\n')
                        counts['trajectory'] += 1
            for stream in manifest['evaluation_streams']:
                name = stream['id'] + ':step-16'
                donor = f'fresh-{stream["initial_code"] ^ 15:02d}-{stream["program"]}:step-16'
                for query in range(6):
                    actual = read(stored(f'{arm}/{donor}'), query)
                    controls.write(json.dumps({'model': arm, 'condition': 'swap', 'state_id': name,
                                               'donor_state_id': donor, 'query_index': query, **actual}) + '\n')
                    counts['swap'] += 1
        for condition in ('zero', 'no_memory'):
            state = LatentSlotState(torch.zeros(1, 2, 8, device=device),
                                    torch.full((1, 2), condition == 'zero', dtype=torch.bool, device=device))
            for query in range(6):
                controls.write(json.dumps({'condition': condition, 'query_index': query, **read(state, query)}) + '\n')
                counts['constant'] += 1
    check_fixed_read_path(reader, bridge, reader_hash, bridge_snapshot)
    if any(not torch.equal(value, payload_snapshot[key]) for key, value in payload.items()):
        raise ValueError('state mutated during read')
    if any(not torch.equal(value, trajectory_snapshot[key]) for key, value in trajectory_payload.items()):
        raise ValueError('trajectory state mutated during read')
    if counts != {'normal': 20160, 'trajectory': 3072, 'swap': 384, 'constant': 12, 'reference': 96}:
        raise ValueError('correction-weight answer coverage differs')
    require_declaration(args.declaration)
    if file_sha256(args.declaration) != declaration_hash:
        raise ValueError('correction-weight declaration changed')
    write_json(args.output / 'report.json', {
        'status': 'complete', 'training_steps': 800, 'states': 3360,
        'trajectory_states': 512, 'counts': counts,
        'training_answer_sequences': 307200,
        'warm_start': SETTINGS['warm_start'], 'inherited_state_weight': INHERITED_STATE_WEIGHT,
        'baseline_answers_reused': 10080, 'baseline_cpu_states_replayed': 1680,
        'parent_initial_replays': 192, 'initializer_unchanged': True,
        'reader_unchanged': True, 'bridge_unchanged': True,
        'state_unchanged_during_reads': True,
    })
    write_json(args.output / 'complete.json', {
        'declaration_sha256': declaration_hash,
        'files': {p.name: file_sha256(p) for p in args.output.iterdir()},
    })


if __name__ == '__main__':
    main()
