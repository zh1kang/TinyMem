"""Train updater-only answer and answer-plus-state objectives from a sealed parent."""

import argparse
import json
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
from scripts.fit_independent_fact_learned_state import (
    SplitWriter, baseline_records, require_declaration as require_prior,
)
from scripts.fit_independent_fact_recurrent_training import collect_states, state_catalog
from scripts.profile_adapted_readout import write_json
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.independent_fact_joint_update import calibrate_state_weight, train_joint_update_batch
from tinymem.research.independent_fact_updates import build_update_cases, new_update_writer
from tinymem.research.independent_fact_placement import placement_state
from tinymem.research.independent_fact_recurrent_training import pack_recurrent_batch
from tinymem.research.readout_read import read_state_answer
from tinymem.research.study_runtime import execution_record, prepare_device, validate_execution
from tinymem.research.update_protocol import file_sha256


ARMS = ('answer_only', 'answer_and_state')
SETTINGS = {
    'kind': 'independent_fact_joint_update_v1',
    'arms': list(ARMS),
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
    'calibration_batches': 16,
    'calibration_rule': 'gradient_fraction_rms_norm_ratio',
    'calibration_gradient_fraction': .25,
    'hard_scientific_cutoff': False,
    'full_answer_ce_continuation_control': True,
}


def require_declaration(path):
    """Validate this run and the sealed learned-state parent before execution."""
    newd = json.loads(path.read_text())
    if (newd.get('settings') != SETTINGS or newd.get('walltime_minutes') != 600
            or newd.get('repair_declaration') != 'independent_learned_state_declaration.json'
            or newd.get('repair_results') != 'independent_learned_state_results'
            or newd.get('repair_proof') != 'independent_learned_state_verification.json'):
        raise ValueError('joint-update declaration settings differ')
    for name, digest in newd['file_sha256'].items():
        relative = Path(name)
        if (relative.is_absolute() or '..' in relative.parts
                or file_sha256(REPOSITORY / relative) != digest):
            raise ValueError('joint-update source or input differs: ' + name)

    prior_path = REPOSITORY / newd['repair_declaration']
    prior_d, parent, paths, parent_results, fit_results = require_prior(prior_path)
    required = {
        newd['repair_declaration'], newd['repair_proof'],
        str((REPOSITORY / newd['repair_results'] / 'complete.json').relative_to(REPOSITORY)),
        'scripts/fit_independent_fact_learned_state.py',
        'scripts/fit_independent_fact_joint_update.py',
        'src/tinymem/research/independent_fact_learned_state.py',
        'src/tinymem/research/independent_fact_joint_update.py',
        'docs/independent_fact_joint_update.md',
        'artifacts/diagnostics/independent_fact_joint_update_20260911_v1/verify_joint.py',
    }
    if not required.issubset(newd['file_sha256']):
        raise ValueError('joint-update provenance coverage differs')
    prior_results = REPOSITORY / newd['repair_results']
    if not prior_results.is_dir():
        raise ValueError('learned-state parent results are missing')
    seal = json.loads((prior_results / 'complete.json').read_text())
    if seal.get('declaration_sha256') != file_sha256(prior_path):
        raise ValueError('learned-state parent completion identity differs')
    if (not seal.get('files')
            or any(Path(name).name != name or file_sha256(prior_results / name) != digest
                   for name, digest in seal['files'].items())):
        raise ValueError('learned-state parent result seal differs')
    if any(newd['file_sha256'].get(str((prior_results / name).relative_to(REPOSITORY))) != digest
           for name, digest in seal['files'].items()):
        raise ValueError('learned-state parent seal is not declared')
    proof = json.loads((REPOSITORY / newd['repair_proof']).read_text())
    if (proof.get('verified') is not True
            or proof.get('completion_sha256') != file_sha256(prior_results / 'complete.json')
            or proof.get('report_sha256') != file_sha256(prior_results / 'report.json')
            or proof.get('declaration_sha256') != file_sha256(prior_path)):
        raise ValueError('learned-state parent proof differs')
    return newd, parent, paths, parent_results, fit_results
def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _state_copy(state):
    return LatentSlotState(state.values.detach().clone().contiguous(),
                           state.valid.detach().clone().contiguous())


def primitive_states(writer, teacher, features):
    """Replay only the 128 one-event states needed by trajectory diagnostics."""
    result = {}
    for case in build_update_cases():
        before = LatentSlotState(teacher.values[case.before_code:case.before_code + 1],
                                 teacher.valid[case.before_code:case.before_code + 1])
        hidden = features[case.target_fact, case.new_bit].unsqueeze(0)
        valid = torch.ones(1, hidden.shape[1], dtype=torch.bool)
        with torch.inference_mode():
            state = writer(before, hidden, valid)
        result['one:' + case.case_id] = _state_copy(state)
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
    manifest = json.loads(paths['manifest'].read_text())
    evaluation = {'evaluation_streams': manifest['training_streams'] + manifest['evaluation_streams']}
    catalog = state_catalog(evaluation)
    if len(catalog) != 1680:
        raise ValueError('joint-update state coverage differs')

    device = prepare_device('cuda')
    torch.set_num_threads(SETTINGS['cpu_threads'])
    execution = execution_record(device)
    validate_execution(execution, expected=json.loads((parent_results / 'protocol.json').read_text())['execution'])
    histories = {int(k): v for k, v in load_file(str(paths['history_features'])).items()}
    features = {tuple(map(int, k.split('.'))): v for k, v in load_file(str(paths['event_features'])).items()}
    original = load_file(str(parent_results / 'answer_only_final.safetensors'))
    initializer = new_update_writer(2048, 1337)
    initializer.load_state_dict(original)
    initializer.requires_grad_(False).eval()
    original_states = load_file(str(parent_results / 'written_states.safetensors'))
    fit_states = load_file(str(fit_results / 'written_states.safetensors'))
    baseline = collect_states(initializer, histories, features, evaluation)
    for name, state in baseline.items():
        source = fit_states if name.startswith('train-') else original_states
        key = f'answer_only/serial_batch1/{name}' if name.startswith('train-') else f'answer_only/{name}'
        if any(not torch.equal(getattr(state, p), source[key + '.' + p]) for p in ('values', 'valid')):
            raise ValueError('baseline state replay differs: ' + name)
    teacher = LatentSlotState(
        torch.cat([baseline[f'initial:{code:02d}'].values for code in range(16)]),
        torch.cat([baseline[f'initial:{code:02d}'].valid for code in range(16)]),
    )
    baseline_answers = baseline_records(parent_results, fit_results, catalog)
    order = list(range(256))
    random.Random(SETTINGS['seed']).shuffle(order)
    ordered = [manifest['training_streams'][i] for i in order]
    batches = [pack_recurrent_batch(ordered[i:i + SETTINGS['batch_streams']], histories, features)
               for i in range(0, 256, SETTINGS['batch_streams'])]
    reader, bridge, rows, reader_hash = load_read_path(parent, paths, SETTINGS['reader'], device)
    bridge_snapshot = {k: v.clone() for k, v in bridge.state_dict().items()}
    worlds = rows

    args.output.mkdir(parents=True)
    save_file({'values': teacher.values, 'valid': teacher.valid}, str(args.output / 'teacher_states.safetensors'))
    write_json(args.output / 'protocol.json', {
        'declaration_sha256': declaration_hash, 'settings': SETTINGS, 'execution': execution,
        'training_order': order, 'state_catalog': catalog,
        'initializer_parameters': sum(v.numel() for v in original.values()),
        'updater_parameters': sum(v.numel() for v in original.values()),
        'baseline_cpu_states_replayed': 1680,
        'read_path_loaded_before_training': True,
    })
    with (args.output / 'baseline_predictions.jsonl').open('x') as handle:
        for (name, query), record in baseline_answers.items():
            handle.write(json.dumps({'state_id': name, 'query_index': query, **record}) + '\n')

    calibration = []
    calibration_writer = new_update_writer(2048, 1337)
    calibration_writer.load_state_dict(original)
    calibration_optimizer = torch.optim.AdamW(calibration_writer.parameters(), **SETTINGS['optimizer'])
    calibration_snapshot = {k: v.clone() for k, v in calibration_writer.state_dict().items()}
    with (args.output / 'calibration.jsonl').open('x') as handle:
        for step in range(SETTINGS['calibration_batches']):
            _sync(device)
            start = time.perf_counter()
            metric = train_joint_update_batch(
                calibration_writer, batches[step], calibration_optimizer,
                reader=reader, bridge=bridge, worlds=worlds, targets=teacher,
                state_weight=0.0, measure_only=True,
            )
            _sync(device)
            metric.update(step=step + 1, batch=step, seconds=time.perf_counter() - start)
            calibration.append(metric)
            handle.write(json.dumps(metric, allow_nan=False) + '\n')
            handle.flush()
            if (step + 1) % 4 == 0:
                print(json.dumps({'calibration_step': step + 1, 'state_weight_metric': metric['state_parameter_gradient_norm']}),
                      flush=True)
    if calibration_optimizer.state:
        raise ValueError('calibration optimizer received state')
    if any(not torch.equal(v, calibration_writer.state_dict()[k]) for k, v in calibration_snapshot.items()):
        raise ValueError('calibration changed original updater')
    calibration_writer.zero_grad(set_to_none=True)
    state_weight = calibrate_state_weight(calibration)
    write_json(args.output / 'calibration.json', {
        'rule': 'gradient_fraction', 'gradient_fraction': .25,
        'state_weight': state_weight, 'batch_count': 16, 'optimizer_steps': 0,
    })
    del calibration_optimizer, calibration_writer

    states = {}
    final_checkpoints = {}
    trajectory_checkpoints = {}
    with (args.output / 'metrics.jsonl').open('x') as metrics_handle:
        for arm in ARMS:
            updater = new_update_writer(2048, 1337)
            updater.load_state_dict(original)
            optimizer = torch.optim.AdamW(updater.parameters(), **SETTINGS['optimizer'])
            weight = 0.0 if arm == 'answer_only' else state_weight
            for step in range(SETTINGS['steps_per_arm']):
                _sync(device)
                start = time.perf_counter()
                metric = train_joint_update_batch(
                    updater, batches[step % len(batches)], optimizer,
                    reader=reader, bridge=bridge, worlds=worlds, targets=teacher,
                    state_weight=weight,
                )
                _sync(device)
                metric.update(arm=arm, step=step + 1, batch=step % len(batches),
                              seconds=time.perf_counter() - start)
                metrics_handle.write(json.dumps(metric, allow_nan=False) + '\n')
                metrics_handle.flush()
                if (step + 1) % 10 == 0:
                    print(json.dumps({'arm': arm, 'step': step + 1, 'answer_ce': metric['answer_ce'],
                                      'state_weight': weight}), flush=True)
                if step + 1 in SETTINGS['trajectory_steps']:
                    checkpoint_path = args.output / f'{arm}_step{step + 1}.safetensors'
                    save_file(updater.state_dict(), str(checkpoint_path))
                    trajectory_checkpoints[arm, step + 1] = checkpoint_path
            updater.zero_grad(set_to_none=True)
            updater.requires_grad_(False).eval()
            final_path = args.output / f'{arm}_final.safetensors'
            save_file(updater.state_dict(), str(final_path))
            final_checkpoints[arm] = final_path
            restored = new_update_writer(2048, 1337)
            restored.load_state_dict(load_file(str(final_path)))
            restored.requires_grad_(False).eval()
            model_states = collect_states(SplitWriter(initializer, restored), histories, features, evaluation)
            states.update({f'{arm}/{name}': state for name, state in model_states.items()})
            if any(not torch.equal(v, original[k]) for k, v in initializer.state_dict().items()):
                raise ValueError('initializer changed')
            del optimizer, updater, restored

    trajectory_states = {}
    for arm in ARMS:
        for step in SETTINGS['trajectory_steps']:
            checkpoint = load_file(str(trajectory_checkpoints[arm, step]))
            writer = new_update_writer(2048, 1337)
            writer.load_state_dict(checkpoint)
            writer.requires_grad_(False).eval()
            for name, state in primitive_states(writer, teacher, features).items():
                trajectory_states[f'{arm}/step{step}/{name}'] = state
    save_file({f'{key}.{part}': getattr(state, part) for key, state in states.items()
               for part in ('values', 'valid')}, str(args.output / 'written_states.safetensors'))
    save_file({f'{key}.{part}': getattr(state, part) for key, state in trajectory_states.items()
               for part in ('values', 'valid')}, str(args.output / 'trajectory_states.safetensors'))

    payload = load_file(str(args.output / 'written_states.safetensors'), device=str(device))
    trajectory_payload = load_file(str(args.output / 'trajectory_states.safetensors'), device=str(device))
    trajectory_snapshot = {key: value.clone() for key, value in trajectory_payload.items()}
    before = torch.tensor(rows[0].before_ids, device=device)
    questions = [torch.tensor(q.after_ids, device=device) for q in rows[0].queries]

    def read(state, query):
        return read_state_answer(reader, bridge, state, before, questions[query], max_new_tokens=8)

    def stored(key, source=payload):
        return LatentSlotState(source[key + '.values'], source[key + '.valid'])

    references = [r for r in map(json.loads, (parent_results / 'reference_replays.jsonl').read_text().splitlines())
                  if r['fold'] == 'even']
    with (args.output / 'reference_replays.jsonl').open('x') as handle:
        for old in references:
            actual = read(placement_state(old['code'], device, 'separate_fact0'), old['query_index'])
            if any(actual[k] != old[k] for k in READ_FIELDS):
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
                    if name.startswith('initial:') and any(actual[k] != baseline_answers[name, query][k] for k in READ_FIELDS):
                        raise ValueError('frozen initial answer differs')
                    predictions.write(json.dumps({'model': arm, 'state_id': name,
                                                  'query_index': query, **actual}) + '\n')
                    counts['normal'] += 1
            for step in SETTINGS['trajectory_steps']:
                for name in (key.split('/', 2)[2] for key in trajectory_states if key.startswith(f'{arm}/step{step}/')):
                    for query in range(6):
                        actual = read(stored(f'{arm}/step{step}/{name}', trajectory_payload), query)
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
    if any(not torch.equal(v.cpu(), getattr(states[key.rsplit('.', 1)[0]], key.rsplit('.', 1)[1]))
           for key, v in payload.items()):
        raise ValueError('state mutated during read')
    if any(not torch.equal(value, trajectory_snapshot[key]) for key, value in trajectory_payload.items()):
        raise ValueError('trajectory state mutated during read')
    if counts != {'normal': 20160, 'trajectory': 3072, 'swap': 384, 'constant': 12, 'reference': 96}:
        raise ValueError('joint-update answer coverage differs')
    require_declaration(args.declaration)
    if file_sha256(args.declaration) != declaration_hash:
        raise ValueError('joint-update declaration changed')
    write_json(args.output / 'report.json', {
        'status': 'complete', 'training_steps': 800, 'states': 3360,
        'trajectory_states': 512, 'counts': counts,
        'calibration_optimizer_steps': 0, 'calibration_answer_sequences': 6144,
        'training_answer_sequences': 307200,
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
