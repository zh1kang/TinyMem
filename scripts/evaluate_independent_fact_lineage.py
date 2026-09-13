"""Evaluate fresh repetition states and the fixed CPU lineage probe."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from safetensors.torch import load_file, save_file

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
if str(REPOSITORY / 'src') not in sys.path:
    sys.path.insert(0, str(REPOSITORY / 'src'))

from scripts.fit_independent_fact_learned_state import SplitWriter
from scripts.fit_independent_fact_recurrent_training import collect_states as collect_old_states
from scripts.evaluate_independent_fact_content_probe import basic, endpoint, views
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.independent_fact_content_probe import (
    permute_family_labels, predict_ridge, select_ridge,
)
from tinymem.research.independent_fact_data import ENTITIES, ROOM_PAIRS
from tinymem.research.independent_fact_repeat_confirmation import (
    collect_states as collect_fresh_states,
    state_catalog as fresh_state_catalog,
)
from scripts.fit_independent_fact_recurrent_training import state_catalog as old_state_catalog
from tinymem.research.independent_fact_placement import placement_state


ARMS = ('uniform', 'correction_weighted')
READ_FIELDS = ('prediction', 'generated_ids', 'memory_positions',
               'native_envelope_tokens', 'input_positions')
ALPHAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
SHUFFLE_SEEDS = list(range(20260912, 20261011))
BOOTSTRAP_SEED = 20260912
BOOTSTRAP_SAMPLES = 10000


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=_json_default) + '\n')


def _copy_state(state):
    return LatentSlotState(state.values.detach().clone().contiguous(),
                           state.valid.detach().clone().contiguous())


def _bits(code):
    return [(code >> fact) & 1 for fact in range(4)]


def _truth_for_events(streams, *, fresh):
    """Validate fixed sentence transitions and return labels for every event state."""
    labels = {}
    for stream in streams:
        code = stream['initial_code']
        events = stream['prefix'] if fresh else stream['events']
        for index, event in enumerate(events, 1):
            expected_text = (f'{ENTITIES[event["target_fact"]]} moved to the '
                             f'{ROOM_PAIRS[event["target_fact"]][event["new_bit"]]}.')
            if (event['step'] != index or event['before_code'] != code
                    or event['text'] != expected_text):
                raise ValueError('manifest sentence transition differs')
            expected_after = code ^ (1 << event['target_fact']) if event['action'] == 'C' else code
            if (event['after_code'] != expected_after
                    or event['new_bit'] != ((expected_after >> event['target_fact']) & 1)):
                raise ValueError('manifest four-bit transition differs')
            name = (f"{stream['id']}/prefix/{index:02d}" if fresh
                    else f"{stream['id']}:step-{index:02d}")
            labels[name] = _bits(event['after_code'])
            code = event['after_code']
        if fresh:
            if code != stream['initial_code'] ^ 15:
                raise ValueError('fresh prefix endpoint differs')
            for condition, tail in stream['tails'].items():
                for index, event in enumerate(tail, 1):
                    expected_text = (f'{ENTITIES[event["target_fact"]]} moved to the '
                                     f'{ROOM_PAIRS[event["target_fact"]][event["new_bit"]]}.')
                    if (event['step'] != index or event['before_code'] != code
                            or event['after_code'] != code or event['action'] != 'R'
                            or event['text'] != expected_text
                            or event['new_bit'] != ((code >> event['target_fact']) & 1)):
                        raise ValueError('fresh repetition transition differs')
                    labels[f"{stream['id']}/{condition}/{index:02d}"] = _bits(code)
    return labels


def _validate_training_design(streams):
    orders = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))
    actions = (('C', 'C', 'C', 'C'), ('R', 'R', 'R', 'R'),
               ('C', 'R', 'C', 'R'), ('R', 'C', 'R', 'C'))
    identities = set()
    for stream in streams:
        parts = stream['id'].split('-')
        if len(parts) != 4 or parts[0] != 'train':
            raise ValueError('training stream identity differs')
        code, order, action = map(int, parts[1:])
        if (code, order, action) in identities or not (0 <= code < 16 and 0 <= order < 4
                                                       and 0 <= action < 4):
            raise ValueError('training identity grid differs')
        identities.add((code, order, action))
        if [event['target_fact'] for event in stream['events']] != list(orders[order]):
            raise ValueError('training order differs')
        _truth_for_events([stream], fresh=False)
        if [event['action'] for event in stream['events']] != list(actions[action]):
            raise ValueError('training action grid differs')
    if identities != {(code, order, action) for code in range(16)
                      for order in range(4) for action in range(4)}:
        raise ValueError('training identity grid is incomplete')


def _state_payload(states, prefix):
    payload = {}
    for name, state in states.items():
        if state.values.device.type != 'cpu' or state.values.dtype != torch.float32:
            raise ValueError('lineage states must be CPU FP32')
        if state.valid.device.type != 'cpu' or state.valid.dtype != torch.bool:
            raise ValueError('lineage validity must be CPU bool')
        if (state.values.shape != (1, 2, 8) or state.valid.shape != (1, 2)
                or not state.valid.all() or state.nbytes != 66
                or not torch.isfinite(state.values).all()
                or (state.values.abs() > 1).any()):
            raise ValueError('invalid lineage state')
        payload[f'{prefix}/{name}.values'] = state.values.detach().clone().contiguous()
        payload[f'{prefix}/{name}.valid'] = state.valid.detach().clone().contiguous()
    return payload


def _catalog_payload(catalog, names):
    return {name: catalog[name] for name in names}


def _state_from_payload(payload, prefix, name):
    return LatentSlotState(payload[f'{prefix}/{name}.values'],
                           payload[f'{prefix}/{name}.valid'])


def _read_row(read, state, query):
    row = read(state, query)
    if set(row) != set(READ_FIELDS):
        raise ValueError('reader record fields differ')
    return row


def _same_fields(left, right):
    return all(left[field] == right[field] for field in READ_FIELDS)


def _hash_rows(values):
    return [hashlib.sha256(np.asarray(row, dtype=np.float32).tobytes()).hexdigest()
            for row in values]


def _collision_measurements(training_values, training_labels, folds, evaluation_values, evaluation_labels,
                            train_names, evaluation_names):
    train_hashes = _hash_rows(training_values)
    eval_hashes = _hash_rows(evaluation_values)
    by_hash = defaultdict(list)
    for index, digest in enumerate(train_hashes):
        by_hash[digest].append(index)
    cross_fold = sum(1 for rows in by_hash.values()
                     if len({int(folds[index]) for index in rows}) > 1)
    train_conflicts = sum(1 for rows in by_hash.values()
                          if len({tuple(training_labels[index]) for index in rows}) > 1)
    eval_by_hash = defaultdict(list)
    for index, digest in enumerate(eval_hashes):
        eval_by_hash[digest].append(index)
    eval_conflicts = sum(1 for rows in eval_by_hash.values()
                         if len({tuple(evaluation_labels[index]) for index in rows}) > 1)
    train_lookup = defaultdict(list)
    for name, digest in zip(train_names, train_hashes, strict=True):
        train_lookup[digest].append(name)
    overlap_names = [name for name, digest in zip(evaluation_names, eval_hashes, strict=True)
                     if digest in train_lookup]
    conflicting_overlap = sum(
        1 for index, digest in enumerate(eval_hashes)
        if digest in train_lookup and any(
            tuple(evaluation_labels[index]) != tuple(training_labels[j])
            for j, train_digest in enumerate(train_hashes) if train_digest == digest))
    return {
        'crossfold_tensor_collisions': int(cross_fold),
        'training_tensor_conflicting_truths': int(train_conflicts),
        'evaluation_tensor_conflicting_truths': int(eval_conflicts),
        'fit_eval_overlap_including_branches': int(len(overlap_names)),
        'fit_eval_overlap_conflicting_truths': int(conflicting_overlap),
        'fit_eval_overlap_by_condition': {
            'prefix': sum('/prefix/' in name for name in overlap_names),
            'balanced': sum('/balanced/' in name for name in overlap_names),
            'single': sum('/single' in name for name in overlap_names),
        },
        'fit_eval_overlap_names': overlap_names,
    }


def _probe(training_values, training_labels, folds, families, evaluation_values,
           evaluation_labels, lm_ok, names, manifest, *, training_names):
    model_data = {}
    data = {'names': names, 'manifest': {'streams': manifest['streams']}}
    all_views = {step: views(data, step) for step in range(1, 9)}
    initial = np.array([index for index, name in enumerate(names) if name.startswith('initial:')])
    draws = (np.random.default_rng(BOOTSTRAP_SEED).integers(0, 4, (BOOTSTRAP_SAMPLES, 16, 4))
             + np.arange(16)[None, :, None] * 4).reshape(-1, 64)
    arrays = {'truth': evaluation_labels, 'training_truth': training_labels, 'folds': folds}
    for arm in ARMS:
        overlap = _collision_measurements(training_values[arm], training_labels, folds,
                                          evaluation_values[arm], evaluation_labels,
                                          training_names, names)
        model, cv = select_ridge(training_values[arm], training_labels, folds, alphas=ALPHAS)
        scores = predict_ridge(model, evaluation_values[arm])
        arrays[f'{arm}_scores'] = scores
        arrays[f'{arm}_lm_correct'] = lm_ok[arm]
        symbolic_training = np.zeros_like(training_values[arm]); symbolic_training[:, :4] = training_labels
        symbolic_evaluation = np.zeros_like(evaluation_values[arm]); symbolic_evaluation[:, :4] = evaluation_labels
        symbolic_model, symbolic_cv = select_ridge(symbolic_training, training_labels, folds, alphas=ALPHAS)
        symbolic_predictions = predict_ridge(symbolic_model, symbolic_evaluation) >= .5
        symbolic_ok = np.array_equal(symbolic_predictions, evaluation_labels)
        if not symbolic_ok:
            raise ValueError('symbolic probe control failed')
        mean_scores = np.broadcast_to(training_labels.mean(0), evaluation_labels.shape)
        reference = (evaluation_values[arm][initial] - model['mean']) / model['scale']
        distances = np.square(((evaluation_values[arm] - model['mean']) / model['scale'])[:, None, :]
                              - reference[None, :, :]).sum(2)
        nearest_scores = evaluation_labels[initial][distances.argmin(1)]
        arm_report = {
            'model': model,
            'cv': cv,
            'cv_margins': _cv_margins(cv),
            'train': basic(predict_ridge(model, training_values[arm]), training_labels,
                           np.arange(len(training_values[arm]))),
            'initial': basic(scores, evaluation_labels, initial),
            'symbolic_control_correct': int((symbolic_predictions == evaluation_labels).sum()),
            'symbolic_control_total': int(evaluation_labels.size),
            'symbolic_cv': symbolic_cv,
            'coefficients': {'weights': model['weights'], 'bias': model['bias'],
                             'mean': model['mean'], 'scale': model['scale']},
            'endpoints': {}, 'tail_curves': {}, 'prefix_curve': [],
            'controls': [], 'distinct_training_states': len(set(_hash_rows(training_values[arm]))),
            'primary_overlap_count': sum(not name.startswith('initial:') and
                                         ('/prefix/' not in name or name.endswith('/prefix/08'))
                                         for name in overlap['fit_eval_overlap_names']),
            'overlap_by_prefix_step': {
                str(step): sum(name.endswith(f'/prefix/{step:02d}')
                               for name in overlap['fit_eval_overlap_names'])
                for step in range(1, 9)
            },
            **overlap,
        }
        for step in range(1, 9):
            indices = np.array([index for index, name in enumerate(names)
                                if name.endswith(f'/prefix/{step:02d}')])
            arm_report['prefix_curve'].append(basic(scores, evaluation_labels, indices))
        for condition, view in all_views[8].items():
            options = {'draws': draws, 'whole_world': condition in ('balanced', 'single_mean')}
            arm_report['endpoints'][condition] = endpoint(scores, evaluation_labels, lm_ok[arm], view, **options)
            arm_report['endpoints'][condition]['intercept_control'] = endpoint(mean_scores, evaluation_labels, lm_ok[arm], view)
            arm_report['endpoints'][condition]['nearest_teacher'] = endpoint(nearest_scores, evaluation_labels, lm_ok[arm], view)
            arm_report['tail_curves'][condition] = [endpoint(scores, evaluation_labels, lm_ok[arm], all_views[step][condition])
                                                      for step in range(1, 9)]
        raw_weights = model['weights'] / model['scale'][:, None]
        arm_report['displacement'] = []
        for condition, view in all_views[8].items():
            root, end, query, _ = view
            change = np.einsum('nd,nd->n', evaluation_values[arm][end].astype(np.float64)
                               - evaluation_values[arm][root], raw_weights[:, query].T)
            if not np.allclose(change, scores[end, query] - scores[root, query], rtol=1e-10, atol=1e-12):
                raise ValueError('linear displacement identity failed')
            arm_report['displacement'].append({
                'condition': condition,
                'mean_signed_score_change': float(((2 * evaluation_labels[end, query] - 1) * change).mean()),
                'mean_state_displacement_l2': float(np.linalg.norm(
                    evaluation_values[arm][end].astype(np.float64) - evaluation_values[arm][root], axis=1).mean()),
            })
        for seed in SHUFFLE_SEEDS:
            shuffled, donor = permute_family_labels(training_labels.reshape(256, 4, 4), families, seed=seed)
            null_model, null_cv = select_ridge(training_values[arm], shuffled.reshape(1024, 4), folds, alphas=ALPHAS)
            null_scores = predict_ridge(null_model, evaluation_values[arm])
            family_identity = bool(np.array_equal(families[donor], families))
            if not family_identity:
                raise ValueError('family shuffle changed fixed family membership')
            arm_report['controls'].append({
                'seed': seed, 'alpha': null_model['alpha'],
                'selected_cv_mse': next(score['mean_mse'] for score in null_cv
                                         if score['alpha'] == null_model['alpha']),
                'family_donor_identity': family_identity,
                'endpoints': {condition: {'accuracy': float(((null_scores[end, query] >= .5)
                                                              == evaluation_labels[end, query]).mean())
                                           }
                                           for condition, (_, end, query, _) in all_views[8].items()},
            })
        arm_report['null_ranges'] = {
            condition: {
                'median': float(np.median(values := [row['endpoints'][condition]['accuracy']
                                                     for row in arm_report['controls']])),
                'interval95': np.quantile(values, [.025, .975]).tolist(),
            } for condition in all_views[8]
        }
        model_data[arm] = arm_report
    return {'models': model_data, 'arrays': arrays}


def _cv_margins(scores):
    ordered = sorted((float(row['mean_mse']), float(row['alpha'])) for row in scores)
    return {
        'best_mean_mse': ordered[0][0],
        'second_mean_mse': ordered[1][0] if len(ordered) > 1 else ordered[0][0],
        'second_minus_best_mse': (ordered[1][0] - ordered[0][0]) if len(ordered) > 1 else 0.0,
        'selected_alpha': max(row['alpha'] for row in scores
                              if np.isclose(row['mean_mse'], ordered[0][0], rtol=1e-12, atol=1e-15)),
    }


def evaluate_lineage(*, output: Path, initializer, updaters: dict, histories: dict,
                     features: dict, training_manifest: dict, refresh_manifest: dict,
                     read, references: list[dict]) -> dict:
    """Create the fixed lineage evaluation and CPU probe artifacts."""
    if output.exists():
        raise FileExistsError(output)
    if set(updaters) != set(ARMS):
        raise ValueError('updaters must contain uniform and correction_weighted')
    if len(training_manifest.get('training_streams', ())) != 256:
        raise ValueError('training manifest must contain 256 streams')
    if len(refresh_manifest.get('streams', ())) != 64:
        raise ValueError('refresh manifest must contain 64 streams')
    for module in (initializer, *updaters.values()):
        if module.training or any(parameter.requires_grad or parameter.grad is not None
                                  for parameter in module.parameters()):
            raise ValueError('lineage writers must be frozen')
        if any(parameter.device.type != 'cpu' or parameter.dtype != torch.float32
               for parameter in module.parameters()):
            raise ValueError('lineage writers must be CPU FP32')

    fresh_labels = _truth_for_events(refresh_manifest['streams'], fresh=True)
    train_labels_by_name = _truth_for_events(training_manifest['training_streams'], fresh=False)
    fresh_catalog = fresh_state_catalog(refresh_manifest)
    old_manifest = {'evaluation_streams': training_manifest['training_streams']}
    _validate_training_design(training_manifest['training_streams'])
    if [stream['initial_code'] for stream in refresh_manifest['streams']] != list(np.repeat(np.arange(16), 4)):
        raise ValueError('fresh bootstrap world strata differ')
    train_catalog = old_state_catalog(old_manifest)
    train_names = [name for name in train_catalog if ':step-' in name]
    if len(train_names) != 1024:
        raise ValueError('training state coverage differs')
    if set(fresh_labels) != set(fresh_catalog):
        initial_labels = {f'initial:{code:02d}': _bits(code) for code in range(16)}
        fresh_labels.update(initial_labels)
        if set(fresh_labels) != set(fresh_catalog):
            raise ValueError('fresh sentence coverage differs')
    if set(train_labels_by_name) != set(train_names):
        raise ValueError('training sentence coverage differs')

    fresh_states = {}
    train_states = {}
    for arm in ARMS:
        fresh_states[arm] = collect_fresh_states(initializer, updaters[arm], histories, features,
                                                 refresh_manifest)
        collected = collect_old_states(SplitWriter(initializer, updaters[arm]), histories, features,
                                       old_manifest)
        train_states[arm] = {name: _copy_state(collected[name]) for name in train_names}
    for code in range(16):
        name = f'initial:{code:02d}'
        for arm in ARMS:
            if any(not torch.equal(getattr(fresh_states[arm][name], field),
                                   getattr(fresh_states['uniform'][name], field))
                   for field in ('values', 'valid')):
                raise ValueError('initial fresh states differ between arms')

    output.mkdir(parents=True)
    save_file({**_state_payload(fresh_states['uniform'], 'uniform'),
               **_state_payload(fresh_states['correction_weighted'], 'correction_weighted')},
              str(output / 'fresh_states.safetensors'))
    save_file({**_state_payload(train_states['uniform'], 'uniform'),
               **_state_payload(train_states['correction_weighted'], 'correction_weighted')},
              str(output / 'training_states.safetensors'))
    fresh_payload = load_file(str(output / 'fresh_states.safetensors'))
    training_payload = load_file(str(output / 'training_states.safetensors'))
    for arm in ARMS:
        for name in fresh_states[arm]:
            stored = _state_from_payload(fresh_payload, arm, name)
            if any(not torch.equal(getattr(stored, field), getattr(fresh_states[arm][name], field))
                   for field in ('values', 'valid')):
                raise ValueError('fresh serialized state differs')
        for name in train_states[arm]:
            stored = _state_from_payload(training_payload, arm, name)
            if any(not torch.equal(getattr(stored, field), getattr(train_states[arm][name], field))
                   for field in ('values', 'valid')):
                raise ValueError('training serialized state differs')
    _write_json(output / 'fresh_catalog.json', fresh_catalog)
    _write_json(output / 'training_catalog.json', _catalog_payload(train_catalog, train_names))

    normal = {}
    counts = {'normal': 0, 'no_write': 0, 'swap': 0, 'constant': 0, 'reference': 0}
    read_snapshots = {
        (arm, name): (fresh_payload[f'{arm}/{name}.values'].clone(),
                      fresh_payload[f'{arm}/{name}.valid'].clone())
        for arm in ARMS for name in fresh_catalog
    }
    with ((output / 'predictions.jsonl').open('x') as prediction_handle,
          (output / 'controls.jsonl').open('x') as control_handle,
          (output / 'reference_replays.jsonl').open('x') as reference_handle):
        for arm in ARMS:
            for name in fresh_catalog:
                state = _state_from_payload(fresh_payload, arm, name)
                for query in range(6):
                    row = _read_row(read, state, query)
                    normal[arm, name, query] = row
                    prediction_handle.write(json.dumps({'model': arm, 'state_id': name,
                                                        'query_index': query, **row}) + '\n')
                    counts['normal'] += 1
                    if counts['normal'] % 512 == 0:
                        prediction_handle.flush()
                        print(json.dumps({'normal_reads': counts['normal'], 'total': 37056}), flush=True)
            for stream in refresh_manifest['streams']:
                prefix = f"{stream['id']}/prefix/08"
                for query in range(6):
                    row = _read_row(read, _state_from_payload(fresh_payload, arm, prefix), query)
                    if not _same_fields(row, normal[arm, prefix, query]):
                        raise ValueError('no-write replay differs from normal read')
                    control_handle.write(json.dumps({'model': arm, 'condition': 'no_write',
                                                     'state_id': prefix, 'query_index': query,
                                                     **row}) + '\n')
                    counts['no_write'] += 1
                replicate = stream['id'].rsplit('-', 1)[1]
                donor = f"repeat-{stream['initial_code'] ^ 15:02d}-{replicate}/balanced/08"
                endpoint = f"{stream['id']}/balanced/08"
                for query in range(6):
                    row = _read_row(read, _state_from_payload(fresh_payload, arm, donor), query)
                    if not _same_fields(row, normal[arm, donor, query]):
                        raise ValueError('swap donor replay differs from normal read')
                    control_handle.write(json.dumps({'model': arm, 'condition': 'swap',
                                                     'state_id': endpoint, 'donor_state_id': donor,
                                                     'query_index': query, **row}) + '\n')
                    counts['swap'] += 1
        for condition in ('zero', 'no_memory'):
            state = LatentSlotState(torch.zeros(1, 2, 8),
                                    torch.full((1, 2), condition == 'zero', dtype=torch.bool))
            for query in range(6):
                row = _read_row(read, state, query)
                control_handle.write(json.dumps({'model': '', 'condition': condition,
                                                 'state_id': '', 'query_index': query, **row}) + '\n')
                counts['constant'] += 1
        if len(references) != 96:
            raise ValueError('reference coverage differs')
        for reference in references:
            row = _read_row(read, placement_state(reference['code'], torch.device('cpu'), 'separate_fact0'),
                            reference['query_index'])
            if not _same_fields(row, reference):
                raise ValueError('coordinate reference replay differs')
            reference_handle.write(json.dumps({'code': reference['code'],
                                               'query_index': reference['query_index'], **row}) + '\n')
            counts['reference'] += 1
    if counts != {'normal': 37056, 'no_write': 768, 'swap': 768, 'constant': 12, 'reference': 96}:
        raise ValueError('lineage read coverage differs')
    for code in range(16):
        name = f'initial:{code:02d}'
        for query in range(6):
            if not _same_fields(normal['uniform', name, query],
                                normal['correction_weighted', name, query]):
                raise ValueError('initial read differs between arms')
    if any(not torch.equal(state.values, snapshot[0]) or not torch.equal(state.valid, snapshot[1])
           for (arm, name), snapshot in read_snapshots.items()
           for state in (_state_from_payload(fresh_payload, arm, name),)):
        raise ValueError('reader mutated fresh lineage states')

    training_values = {arm: np.stack([training_payload[f'{arm}/{name}.values'].numpy().reshape(16)
                                      for name in train_names]) for arm in ARMS}
    evaluation_names = list(fresh_catalog)
    evaluation_values = {arm: np.stack([fresh_payload[f'{arm}/{name}.values'].numpy().reshape(16)
                                        for name in evaluation_names]) for arm in ARMS}
    training_labels = np.asarray([train_labels_by_name[name] for name in train_names], dtype=float)
    evaluation_labels = np.asarray([fresh_labels[name] for name in evaluation_names], dtype=float)
    identities = [stream['id'].split('-') for stream in training_manifest['training_streams']]
    folds = np.repeat([int(parts[2]) for parts in identities], 4)
    families = np.asarray([4 * int(parts[2]) + int(parts[3]) for parts in identities])
    if len(training_labels) != 1024 or folds.shape != (1024,):
        raise ValueError('fixed training probe layout differs')
    lm_ok = {arm: np.zeros((len(evaluation_names), 4), dtype=bool) for arm in ARMS}
    for arm in ARMS:
        for index, name in enumerate(evaluation_names):
            for fact in range(4):
                row = normal[arm, name, fact]
                expected = ROOM_PAIRS[fact][int(evaluation_labels[index, fact])]
                lm_ok[arm][index, fact] = row['prediction'].strip().lower() == expected
    probe = _probe(training_values, training_labels, folds, families, evaluation_values,
                   evaluation_labels, lm_ok, evaluation_names, refresh_manifest,
                   training_names=train_names)
    np.savez_compressed(output / 'probe_scores.npz', **probe.pop('arrays'))
    _write_json(output / 'probe_summary.json', probe)
    report = {
        'status': 'complete', 'states': {'fresh_per_model': 3088, 'fresh_total': 6176,
                                          'training_per_model': 1024, 'training_total': 2048},
        'counts': counts, 'total_reads': 38700, 'probe': {
            'alphas': ALPHAS, 'shuffle_seeds': SHUFFLE_SEEDS,
            'bootstrap_seed': BOOTSTRAP_SEED, 'bootstrap_samples': BOOTSTRAP_SAMPLES,
            'accuracy_gates': False,
        },
        'artifacts': ['fresh_states.safetensors', 'training_states.safetensors',
                      'fresh_catalog.json', 'training_catalog.json', 'predictions.jsonl',
                      'controls.jsonl', 'reference_replays.jsonl', 'probe_summary.json',
                      'probe_scores.npz'],
    }
    _write_json(output / 'report.json', report)
    return report
