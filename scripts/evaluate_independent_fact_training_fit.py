"""Generated recall on the completed answer writer's exact training paths."""
import argparse
import json
from pathlib import Path
import random

import torch
from safetensors.torch import load_file, save_file

from scripts.fit_independent_fact_answer_training import (
    ARMS, check_fixed_read_path, load_read_path, verify_inputs,
)
from scripts.profile_adapted_readout import write_json
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.independent_fact_answer_protocol import validate_answer_manifest
from tinymem.research.independent_fact_placement import placement_state
from tinymem.research.independent_fact_recurrent_training import pack_recurrent_batch
from tinymem.research.independent_fact_updates import new_update_writer
from tinymem.research.readout_read import read_state_answer
from tinymem.research.study_runtime import REPOSITORY, execution_record, prepare_device, validate_execution
from tinymem.research.update_protocol import file_sha256

GEOMETRIES = ('training_batch16', 'serial_batch1')
SETTINGS = {
    'training_steps': 0, 'parent_job': '13717851', 'reader': 'even',
    'arms': list(ARMS), 'geometries': list(GEOMETRIES), 'streams': 256,
    'horizon': 4, 'states_per_arm_geometry': 1040, 'states': 4160,
    'generated_answers': 24960, 'exact_reference_replays': 96,
    'parent_initial_replays': 192, 'max_new_tokens': 8, 'cpu_threads': 4,
    'persistent_bytes': 66, 'hard_scientific_cutoff': False,
}
READ_FIELDS = ('prediction', 'generated_ids', 'memory_positions', 'native_envelope_tokens', 'input_positions')


def catalog_for(manifest):
    validate_answer_manifest(manifest)
    catalog = {f'initial:{code:02d}': {'code': code, 'step': 0} for code in range(16)}
    for stream in manifest['training_streams']:
        for event in stream['events']:
            catalog[f'{stream["id"]}:step-{event["step"]:02d}'] = {
                'code': event['after_code'], 'step': event['step'], 'stream': stream['id'],
                'target': event['target_fact'], 'action': event['action'],
            }
    if len(catalog) != 1040:
        raise ValueError('training state coverage differs')
    return catalog


def _store(states, name, state):
    saved = LatentSlotState(state.values.detach().clone().contiguous(), state.valid.detach().clone().contiguous())
    if (name in states or saved.values.shape != (1, 2, 8) or saved.nbytes != 66
            or saved.values.dtype != torch.float32 or saved.values.device.type != 'cpu'
            or not saved.valid.all() or not torch.isfinite(saved.values).all()
            or (saved.values.abs() > 1).any()):
        raise ValueError('invalid, duplicate, or nonfinite training-fit state')
    states[name] = saved


def collect_training_states(writer, histories, features, manifest, order):
    """Keep exact training padding and batch membership separate from serial evaluation."""
    catalog = catalog_for(manifest)
    if sorted(order) != list(range(256)):
        raise ValueError('training permutation differs')
    streams = manifest['training_streams']
    states = {geometry: {} for geometry in GEOMETRIES}
    with torch.no_grad():
        for start in range(0, 256, 16):
            selected = [streams[i] for i in order[start:start+16]]
            batch = pack_recurrent_batch(selected, histories, features)
            initial = writer(writer.empty(16), batch.histories, batch.history_valid)
            if start == 0:
                for code in range(16):
                    _store(states['training_batch16'], f'initial:{code:02d}',
                           LatentSlotState(initial.values[code:code+1], initial.valid[code:code+1]))
            else:
                for code in range(16):
                    if not torch.equal(initial.values[code:code+1], states['training_batch16'][f'initial:{code:02d}'].values):
                        raise ValueError('training initializer changed across batches')
            state = LatentSlotState(initial.values.index_select(0, batch.initial_codes),
                                    initial.valid.index_select(0, batch.initial_codes))
            for step in range(4):
                state = writer(state, batch.events[:, step], batch.event_valid[:, step])
                for row, stream_id in enumerate(batch.stream_ids):
                    _store(states['training_batch16'], f'{stream_id}:step-{step+1:02d}',
                           LatentSlotState(state.values[row:row+1], state.valid[row:row+1]))
    with torch.inference_mode():
        for code, hidden in histories.items():
            state = writer(writer.empty(1), hidden.unsqueeze(0), torch.ones(1, len(hidden), dtype=torch.bool))
            _store(states['serial_batch1'], f'initial:{code:02d}', state)
        for stream in streams:
            state = states['serial_batch1'][f'initial:{stream["initial_code"]:02d}']
            for event in stream['events']:
                hidden = features[event['target_fact'], event['new_bit']]
                state = writer(state, hidden.unsqueeze(0), torch.ones(1, len(hidden), dtype=torch.bool))
                _store(states['serial_batch1'], f'{stream["id"]}:step-{event["step"]:02d}', state)
    if any(set(values) != set(catalog) for values in states.values()):
        raise ValueError('training-fit state IDs differ')
    return states


def require_declaration(path):
    declaration = json.loads(path.read_text())
    if (declaration.get('kind') != 'independent_fact_training_fit_declaration_v1'
            or declaration.get('resource_cap_minutes') != 60 or declaration['settings'] != SETTINGS):
        raise ValueError('diagnostic settings differ')
    for name, digest in declaration['file_sha256'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or file_sha256(REPOSITORY / relative) != digest:
            raise ValueError('diagnostic source or input differs: ' + name)
    parent = REPOSITORY / declaration['parent_declaration']
    parent_declaration = json.loads(parent.read_text())
    paths, _ = verify_inputs(parent_declaration)
    parent_results = REPOSITORY / declaration['parent_results']
    complete = json.loads((parent_results / 'complete.json').read_text())
    if complete['declaration_sha256'] != file_sha256(parent):
        raise ValueError('parent completion identity differs')
    for name, digest in complete['files'].items():
        if Path(name).name != name or file_sha256(parent_results / name) != digest:
            raise ValueError('parent result seal differs')
        if declaration['file_sha256'].get(str((parent_results / name).relative_to(REPOSITORY))) != digest:
            raise ValueError('parent result omitted from diagnostic declaration')
    proof = json.loads((REPOSITORY / declaration['parent_proof']).read_text())
    if (proof.get('verified') is not True or proof.get('completion_sha256') != file_sha256(parent_results / 'complete.json')
            or proof.get('report_sha256') != file_sha256(parent_results / 'report.json')
            or proof.get('declaration_sha256') != file_sha256(parent)):
        raise ValueError('parent proof does not bind completed run')
    required = {declaration['parent_declaration'], declaration['parent_proof'],
                str((parent_results / 'complete.json').relative_to(REPOSITORY)),
                'scripts/evaluate_independent_fact_training_fit.py',
                'docs/independent_fact_training_fit.md',
                'artifacts/diagnostics/independent_fact_training_fit_20260911_v1/verify_fit.py'}
    if not required.issubset(declaration['file_sha256']):
        raise ValueError('diagnostic provenance coverage differs')
    return declaration, parent_declaration, paths, parent_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declaration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    declaration_hash = file_sha256(args.declaration)
    d, parent, paths, parent_results = require_declaration(args.declaration)
    manifest = json.loads(paths['manifest'].read_text())
    catalog = catalog_for(manifest)
    order = json.loads((parent_results / 'protocol.json').read_text())['training_order']
    expected_order = list(range(256)); random.Random(1337).shuffle(expected_order)
    if order != expected_order:
        raise ValueError('parent training order differs')
    device = prepare_device('cuda')
    torch.set_num_threads(4)
    execution = execution_record(device)
    validate_execution(execution, expected=json.loads((parent_results / 'protocol.json').read_text())['execution'])
    histories = {int(k): v for k, v in load_file(str(paths['history_features'])).items()}
    features = {tuple(map(int, k.split('.'))): v for k, v in load_file(str(paths['event_features'])).items()}
    parent_states = load_file(str(parent_results / 'written_states.safetensors'))
    args.output.mkdir(parents=True)
    payload = {}
    geometry = {}
    for arm in ARMS:
        writer = new_update_writer(2048, 1337)
        checkpoint = load_file(str(parent_results / (arm + '_final.safetensors')))
        writer.load_state_dict(checkpoint); writer.requires_grad_(False).eval()
        states = collect_training_states(writer, histories, features, manifest, order)
        for code in range(16):
            name = f'initial:{code:02d}'
            for part in ('values', 'valid'):
                if not torch.equal(getattr(states['serial_batch1'][name], part), parent_states[f'{arm}/{name}.{part}']):
                    raise ValueError('serial initial state does not replay parent')
        geometry[arm] = {
            'equal_states': sum(torch.equal(states[GEOMETRIES[0]][n].values, states[GEOMETRIES[1]][n].values) for n in catalog),
            'total_states': len(catalog),
            'max_abs_difference': max(float((states[GEOMETRIES[0]][n].values-states[GEOMETRIES[1]][n].values).abs().max()) for n in catalog),
        }
        payload.update({f'{arm}/{g}/{name}.{part}': getattr(state, part)
                        for g, values in states.items() for name, state in values.items() for part in ('values', 'valid')})
        if any(not torch.equal(v, checkpoint[k]) for k, v in writer.state_dict().items()):
            raise ValueError('writer parameters changed during evaluation')
    save_file(payload, str(args.output / 'written_states.safetensors'))
    write_json(args.output / 'protocol.json', {'declaration_sha256': declaration_hash, 'settings': SETTINGS,
               'execution': execution, 'training_order': order, 'state_catalog': catalog, 'geometry': geometry})
    state_hash = file_sha256(args.output / 'written_states.safetensors')
    reader, bridge, rows, reader_hash = load_read_path(parent, paths, 'even', device)
    bridge_snapshot = {k: v.clone() for k, v in bridge.state_dict().items()}
    payload = load_file(str(args.output / 'written_states.safetensors'), device=str(device))
    before = torch.tensor(rows[0].before_ids, device=device)
    questions = [torch.tensor(q.after_ids, device=device) for q in rows[0].queries]
    def read(state, query):
        return read_state_answer(reader, bridge, state, before, questions[query], max_new_tokens=8)
    old_predictions = {(r['model'], r['state_id'], r['query_index']): r
                       for r in map(json.loads, (parent_results / 'predictions.jsonl').read_text().splitlines())
                       if r['fold'] == 'even' and r['state_id'].startswith('initial:')}
    references = [r for r in map(json.loads, (parent_results / 'reference_replays.jsonl').read_text().splitlines()) if r['fold'] == 'even']
    with (args.output / 'reference_replays.jsonl').open('x') as handle:
        for old in references:
            actual = read(placement_state(old['code'], device, 'separate_fact0'), old['query_index'])
            if any(actual[k] != old[k] for k in READ_FIELDS):
                raise ValueError('fixed reader reference replay differs')
            handle.write(json.dumps({'code': old['code'], 'query_index': old['query_index'], **actual}) + '\n')
    count = 0
    with (args.output / 'predictions.jsonl').open('x') as handle:
        for arm in ARMS:
            for g in GEOMETRIES:
                for name in catalog:
                    key = f'{arm}/{g}/{name}'
                    state = LatentSlotState(payload[key+'.values'], payload[key+'.valid'])
                    for query in range(6):
                        actual = read(state, query)
                        if g == 'serial_batch1' and name.startswith('initial:'):
                            if any(actual[k] != old_predictions[arm, name, query][k] for k in READ_FIELDS):
                                raise ValueError('parent initial answer replay differs')
                        handle.write(json.dumps({'model': arm, 'geometry': g, 'state_id': name, 'query_index': query, **actual}) + '\n')
                        count += 1
                    if count % 600 == 0:
                        handle.flush(); print(json.dumps({'generated_answers': count, 'total': 24960}), flush=True)
                handle.flush()
    check_fixed_read_path(reader, bridge, reader_hash, bridge_snapshot)
    serialized = load_file(str(args.output / 'written_states.safetensors'))
    if any(not torch.equal(value.cpu(), serialized[key]) for key, value in payload.items()):
        raise ValueError('reader mutated serialized state tensors')
    if count != 24960 or len(references) != 96 or state_hash != file_sha256(args.output / 'written_states.safetensors'):
        raise ValueError('completed output coverage or state seal differs')
    require_declaration(args.declaration)
    if file_sha256(args.declaration) != declaration_hash:
        raise ValueError('diagnostic declaration changed')
    write_json(args.output / 'report.json', {'status': 'complete', 'training_steps': 0, 'states': 4160,
               'generated_answers': count, 'reference_replays': len(references), 'parent_initial_replays': 192,
               'writer_unchanged': True, 'reader_unchanged': True, 'bridge_unchanged': True})
    write_json(args.output / 'complete.json', {'declaration_sha256': declaration_hash,
               'files': {p.name: file_sha256(p) for p in args.output.iterdir()}})


if __name__ == '__main__':
    main()
