"""Fit and evaluate the sealed CPU content-recovery diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

from tinymem.research.independent_fact_content_probe import (
    permute_family_labels, predict_ridge, select_ridge,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/diagnostics/independent_fact_content_probe_20260912_v1'
OLD = BASE.parent / 'independent_fact_correction_weight_20260912_v1'
FRESH = BASE.parent / 'independent_fact_repeat_confirmation_20260912_v1'
ARMS = ('uniform', 'correction_weighted')
SETTINGS = {'alphas': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
            'shuffle_seeds': list(range(20260912, 20261011)), 'bootstrap_seed': 20260912,
            'bootstrap_samples': 10000, 'threshold': .5, 'training_states': 1024,
            'evaluation_states': 3088, 'training_steps': 0, 'gpu_reads': 0}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    def convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.generic): return obj.item()
        raise TypeError(type(obj).__name__)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=convert) + '\n')


def require(condition, message):
    if not condition: raise ValueError(message)


def required_files():
    paths = [Path('src/tinymem/research/independent_fact_content_probe.py'),
             Path('scripts/evaluate_independent_fact_content_probe.py'),
             Path('tests/test_independent_fact_content_probe.py'),
             Path('docs/independent_fact_content_probe.md'),
             BASE.relative_to(ROOT) / 'advice_record.json',
             BASE.relative_to(ROOT) / 'claude_advice.json',
             BASE.relative_to(ROOT) / 'freeze_probe.py']
    for base in (OLD, FRESH):
        paths.extend((base / name).relative_to(ROOT) for name in
                     ('declaration.json', 'verification.json', 'results/complete.json',
                      'results/protocol.json', 'results/written_states.safetensors'))
    paths.extend((path.relative_to(ROOT) for path in (
        OLD / 'snapshot/independent_answer_training_data.json',
        FRESH / 'snapshot/independent_fact_inputs.json',
        FRESH / 'data_manifest.json', FRESH / 'results/predictions.jsonl')))
    return paths


def validate_declaration(path):
    d = read(path)
    require(d['settings'] == SETTINGS, 'declared settings differ')
    require(set(d['files']) == {str(p) for p in required_files()}, 'declaration source coverage')
    for name, digest in d['files'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'invalid source path')
        require(sha(ROOT / relative) == digest, 'source or input changed: ' + name)
    for base in (OLD, FRESH):
        seal = read(base / 'results/complete.json'); proof = read(base / 'verification.json')
        require(seal['declaration_sha256'] == sha(base / 'declaration.json') == proof['declaration_sha256'], 'parent declaration identity')
        require(proof['verified'] is True and proof['completion_sha256'] == sha(base / 'results/complete.json')
                and proof['report_sha256'] == sha(base / 'results/report.json'), 'parent proof identity')
        for name, digest in seal['files'].items():
            require(Path(name).name == name and sha(base / 'results' / name) == digest, 'parent result seal')
    return d


def sentence_truth(streams, worlds, entities, *, fresh):
    labels = {}
    for stream in streams:
        truth = {}
        for line in worlds[stream['initial_code']]['cases'][0]['context'].splitlines():
            entity, room = line.removesuffix('.').split(' moved to the '); truth[entity] = room
        sequences = [('prefix', stream['prefix'])] if fresh else [('train', stream['events'])]
        for condition, events in sequences:
            for event in events:
                entity, room = event['text'].removesuffix('.').split(' moved to the ')
                truth[entity] = room
                answers = [truth.get(e, 'unknown') for e in entities]
                require(answers == [q['answer'] for q in worlds[event['after_code']]['cases']], 'sentence truth differs')
                name = f"{stream['id']}/{condition}/{event['step']:02d}" if fresh else f"{stream['id']}:step-{event['step']:02d}"
                labels[name] = answers
        if fresh:
            for condition, events in stream['tails'].items():
                for event in events:
                    entity, room = event['text'].removesuffix('.').split(' moved to the ')
                    require(truth[entity] == room, 'tail statement is not a truthful repeat')
                    labels[f"{stream['id']}/{condition}/{event['step']:02d}"] = [truth.get(e, 'unknown') for e in entities]
    return labels


def load_data():
    worlds = read(FRESH / 'snapshot/independent_fact_inputs.json')['worlds']
    entities = [q['question'].removeprefix('Where is ').removesuffix('?') for q in worlds[0]['cases']]
    pairs = [[worlds[0]['cases'][q]['answer'], worlds[1 << q]['cases'][q]['answer']] for q in range(4)]
    def bits(answers): return [pairs[q].index(answers[q]) for q in range(4)]
    old_manifest = read(OLD / 'snapshot/independent_answer_training_data.json')
    old_streams = old_manifest['training_streams']; manifest = read(FRESH / 'data_manifest.json')
    old_catalog = read(OLD / 'results/protocol.json')['state_catalog']
    catalog = read(FRESH / 'results/protocol.json')['state_catalog']
    old_truth = sentence_truth(old_streams, worlds, entities, fresh=False)
    fresh_truth = sentence_truth(manifest['streams'], worlds, entities, fresh=True)
    fresh_truth.update({f"initial:{w['code']:02d}": [q['answer'] for q in w['cases']] for w in worlds})
    names = list(catalog)
    train_names = [f"{s['id']}:step-{e['step']:02d}" for s in old_streams for e in s['events']]
    require(len(train_names) == len(set(train_names)) == 1024 and set(fresh_truth) == set(names)
            and len(names) == 3088, 'state coverage')
    for n in names:
        require(fresh_truth[n] == [q['answer'] for q in worlds[catalog[n]['code']]['cases']], 'fresh catalog truth')
    for n in train_names:
        require(old_truth[n] == [q['answer'] for q in worlds[old_catalog[n]['code']]['cases']], 'old catalog truth')
    identities = [s['id'].split('-') for s in old_streams]
    require({tuple(map(int, x[1:])) for x in identities} == {
        (w, o, a) for w in range(16) for o in range(4) for a in range(4)}, 'training trajectory layout')
    folds = np.repeat([int(x[2]) for x in identities], 4)
    families = np.array([4 * int(x[2]) + int(x[3]) for x in identities])
    y = np.array([bits(old_truth[n]) for n in train_names]); target = np.array([bits(fresh_truth[n]) for n in names])
    old_payload = load_file(str(OLD / 'results/written_states.safetensors'))
    fresh_payload = load_file(str(FRESH / 'results/written_states.safetensors'))
    lm_rows = [json.loads(line) for line in (FRESH / 'results/predictions.jsonl').read_text().splitlines()]
    lm = {(r['model'], r['state_id'], r['query_index']): r['prediction'].strip().lower() for r in lm_rows}
    require(len(lm) == len(lm_rows) == 37056, 'LM coverage')
    arms = {}
    for arm in ARMS:
        def values(payload, ns):
            tensors = []
            for n in ns:
                key = f'{arm}/{n}'
                valid = payload[key + '.valid'].numpy(); v = payload[key + '.values'].numpy()
                require(valid.shape == (1, 2) and valid.dtype == np.bool_ and valid.all() and v.shape == (1, 2, 8)
                        and v.dtype == np.float32 and np.isfinite(v).all(), 'stored state contract')
                tensors.append(v.reshape(16))
            return np.stack(tensors)
        x = values(old_payload, train_names); evaluation = values(fresh_payload, names)
        hashes = [hashlib.sha256(v.tobytes()).hexdigest() for v in x]
        for fold in range(4):
            require(not set(h for h, f in zip(hashes, folds, strict=True) if f == fold)
                    & set(h for h, f in zip(hashes, folds, strict=True) if f != fold), 'cross-fold tensor duplicate')
        overlaps = [n for n, v in zip(names, evaluation, strict=True) if hashlib.sha256(v.tobytes()).hexdigest() in set(hashes)]
        require(all('/prefix/' in n and not n.endswith('/08') for n in overlaps), 'primary evaluation tensor overlaps fitting')
        lm_ok = np.array([[lm[arm, n, q] == fresh_truth[n][q] for q in range(4)] for n in names])
        arms[arm] = {'x': x, 'evaluation': evaluation, 'lm_ok': lm_ok, 'overlap_names': overlaps,
                     'distinct_train_states': len(set(hashes))}
    return {'arms': arms, 'y': y, 'target': target, 'folds': folds, 'families': families,
            'names': names, 'train_names': train_names, 'manifest': manifest}


def views(data, step=8):
    lookup = {name: i for i, name in enumerate(data['names'])}
    result = {}
    for condition in ('balanced', 'spoken', 'unspoken', 'single_mean'):
        source, end, query, owner = [], [], [], []
        for i, stream in enumerate(data['manifest']['streams']):
            root = lookup[f"{stream['id']}/prefix/08"]
            branches = ['balanced'] if condition == 'balanced' else [f'single{f}' for f in range(4)]
            for branch in branches:
                for q in range(4):
                    if condition == 'spoken' and q != int(branch[-1]): continue
                    if condition == 'unspoken' and q == int(branch[-1]): continue
                    source.append(root); end.append(lookup[f"{stream['id']}/{branch}/{step:02d}"])
                    query.append(q); owner.append(i)
        result[condition] = tuple(np.array(x) for x in (source, end, query, owner))
    return result


def interval(values, draws):
    return np.quantile(np.asarray(values)[draws].mean(1), [.025, .975]).tolist()


def endpoint(scores, target, lm_ok, view, *, draws=None, whole_world=False):
    source, end, q, owner = view
    before = (scores[source, q] >= .5) == target[source, q]
    after = (scores[end, q] >= .5) == target[end, q]
    lm_before, lm_after = lm_ok[source, q], lm_ok[end, q]
    repairs, damage = int((~before & after).sum()), int((before & ~after).sum())
    require(int(after.sum()) - int(before.sum()) == repairs - damage, 'endpoint accounting')
    counts = np.bincount(owner, minlength=64)
    gain = np.bincount(owner, weights=after.astype(int) - before.astype(int), minlength=64) / counts
    lm_gain = np.bincount(owner, weights=lm_after.astype(int) - lm_before.astype(int), minlength=64) / counts
    damaged = lm_before & ~lm_after; repaired = ~lm_before & lm_after
    summary = {'n': len(q), 'before_correct': int(before.sum()), 'after_correct': int(after.sum()),
        'accuracy': float(after.mean()), 'mse': float(np.square(scores[end, q] - target[end, q]).mean()),
        'repairs': repairs, 'damage': damage, 'gain': int(after.sum()) - int(before.sum()),
        'mean_signed_margin': float(((2 * target[end, q] - 1) * (scores[end, q] - .5)).mean()),
        'both_correct': int((lm_after & after).sum()), 'lm_wrong_probe_correct': int((~lm_after & after).sum()),
        'lm_correct_probe_wrong': int((lm_after & ~after).sum()), 'both_wrong': int((~lm_after & ~after).sum()),
        'lm_damage_n': int(damaged.sum()), 'lm_damage_probe_correct': int((damaged & after).sum()),
        'lm_damage_probe_correct_at_both_ends': int((damaged & before & after).sum()),
        'lm_repair_n': int(repaired.sum()), 'lm_repair_probe_correct': int((repaired & after).sum()),
        'per_prefix_gain': gain.tolist(), 'lm_per_prefix_gain': lm_gain.tolist(),
        'per_fact': {str(f): {'n': int((q == f).sum()), 'correct': int(after[q == f].sum())} for f in range(4)}}
    if whole_world:
        summary['complete_worlds'] = int(after.reshape(-1, 4).all(1).sum())
        summary['worlds'] = len(q) // 4
    if draws is not None:
        summary['gain_interval95'] = interval(gain, draws)
        summary['lm_gain_interval95'] = interval(lm_gain, draws)
        summary['probe_minus_lm_gain_interval95'] = interval(gain - lm_gain, draws)
    return summary


def basic(scores, target, indices):
    ok = (scores[indices] >= .5) == target[indices]
    return {'n': int(ok.size), 'correct': int(ok.sum()), 'complete_worlds': int(ok.all(1).sum()),
            'worlds': len(indices), 'per_fact_correct': ok.sum(0).tolist(),
            'mse': float(np.square(scores[indices] - target[indices]).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declaration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'output directory already exists')
    validate_declaration(args.declaration)
    declaration_hash = sha(args.declaration)
    data = load_data(); y = data['y']; target = data['target']; names = data['names']
    require([s['initial_code'] for s in data['manifest']['streams']] == list(np.repeat(np.arange(16), 4)), 'bootstrap strata layout')
    rng = np.random.default_rng(SETTINGS['bootstrap_seed'])
    draws = (rng.integers(0, 4, (SETTINGS['bootstrap_samples'], 16, 4)) + np.arange(16)[None, :, None] * 4).reshape(-1, 64)
    args.output.mkdir(parents=True)
    summary = {'declaration_sha256': declaration_hash, 'models': {}, 'training_steps': 0, 'gpu_reads': 0}
    arrays = {'truth': target, 'training_truth': y, 'folds': data['folds']}
    all_views = {step: views(data, step) for step in range(1, 9)}
    initial = np.array([i for i, n in enumerate(names) if n.startswith('initial:')])
    for arm, values in data['arms'].items():
        x, evaluation, lm_ok = values['x'], values['evaluation'], values['lm_ok']
        model, cv = select_ridge(x, y, data['folds'], alphas=SETTINGS['alphas'])
        scores = predict_ridge(model, evaluation)
        arrays[arm + '_scores'] = scores; arrays[arm + '_lm_correct'] = lm_ok
        symbolic_x = np.zeros_like(x); symbolic_x[:, :4] = y
        symbolic_eval = np.zeros_like(evaluation); symbolic_eval[:, :4] = target
        positive, _ = select_ridge(symbolic_x, y, data['folds'], alphas=SETTINGS['alphas'])
        require(np.array_equal(predict_ridge(positive, symbolic_eval) >= .5, target), 'symbolic positive control')
        mean_scores = np.broadcast_to(y.mean(0), target.shape)
        reference = (evaluation[initial] - model['mean']) / model['scale']
        distances = np.square(((evaluation - model['mean']) / model['scale'])[:, None, :] - reference[None, :, :]).sum(2)
        nearest_scores = target[initial][distances.argmin(1)]
        arrays[arm + '_nearest_scores'] = nearest_scores
        arm_report = {'model': model, 'cv': cv, 'train': basic(predict_ridge(model, x), y, np.arange(1024)),
            'initial': basic(scores, target, initial), 'distinct_train_states': values['distinct_train_states'],
            'overlap_names': values['overlap_names'],
            'overlap_by_prefix_step': {str(step): sum(n.endswith(f'/prefix/{step:02d}') for n in values['overlap_names']) for step in range(1, 9)},
            'symbolic_control_correct': int(target.size),
            'endpoints': {}, 'tail_curves': {}, 'prefix_curve': [], 'controls': []}
        for step in range(1, 9):
            indices = np.array([i for i, n in enumerate(names) if n.endswith(f'/prefix/{step:02d}')])
            arm_report['prefix_curve'].append(basic(scores, target, indices))
        for condition, view in all_views[8].items():
            options = {'draws': draws, 'whole_world': condition in ('balanced', 'single_mean')}
            arm_report['endpoints'][condition] = endpoint(scores, target, lm_ok, view, **options)
            arm_report['endpoints'][condition]['intercept_control'] = endpoint(mean_scores, target, lm_ok, view)
            arm_report['endpoints'][condition]['nearest_teacher'] = endpoint(nearest_scores, target, lm_ok, view)
            arm_report['tail_curves'][condition] = [endpoint(scores, target, lm_ok, all_views[step][condition]) for step in range(1, 9)]
        # This projection describes the fitted linear readout, not a causal mechanism.
        raw_weights = model['weights'] / model['scale'][:, None]
        projections = []
        for condition, view in all_views[8].items():
            root, end, q, _ = view
            change = np.einsum('nd,nd->n', evaluation[end].astype(np.float64) - evaluation[root], raw_weights[:, q].T)
            require(np.allclose(change, scores[end, q] - scores[root, q], rtol=1e-10, atol=1e-12), 'linear displacement identity')
            projections.append({'condition': condition, 'mean_signed_score_change': float(((2 * target[end, q] - 1) * change).mean()),
                'mean_state_displacement_l2': float(np.linalg.norm(evaluation[end].astype(np.float64) - evaluation[root], axis=1).mean())})
        arm_report['displacement'] = projections
        for seed in SETTINGS['shuffle_seeds']:
            shuffled, donor = permute_family_labels(y.reshape(256, 4, 4), data['families'], seed=seed)
            require(np.array_equal(data['folds'][::4][donor], data['folds'][::4]), 'shuffle crosses validation fold')
            null_model, null_cv = select_ridge(x, shuffled.reshape(1024, 4), data['folds'], alphas=SETTINGS['alphas'])
            null_scores = predict_ridge(null_model, evaluation)
            arm_report['controls'].append({'seed': seed, 'alpha': null_model['alpha'],
                'selected_cv_mse': next(s['mean_mse'] for s in null_cv if s['alpha'] == null_model['alpha']),
                'endpoints': {condition: {'accuracy': float(((null_scores[end, q] >= .5) == target[end, q]).mean())}
                    for condition, (_, end, q, _) in all_views[8].items()}})
        arm_report['null_ranges'] = {condition: {'median': float(np.median(a := [r['endpoints'][condition]['accuracy'] for r in arm_report['controls']])),
            'interval95': np.quantile(a, [.025, .975]).tolist()} for condition in all_views[8]}
        summary['models'][arm] = arm_report
        print(json.dumps({'model': arm, 'alpha': model['alpha'], 'status': 'fits_complete'}), flush=True)
    write(args.output / 'summary.json', summary)
    write(args.output / 'state_names.json', {'train': data['train_names'], 'evaluation': names})
    np.savez_compressed(args.output / 'predictions.npz', **arrays)
    validate_declaration(args.declaration)
    require(sha(args.declaration) == declaration_hash, 'declaration changed')
    write(args.output / 'complete.json', {'declaration_sha256': declaration_hash,
        'files': {p.name: sha(p) for p in args.output.iterdir()}})


if __name__ == '__main__':
    main()
