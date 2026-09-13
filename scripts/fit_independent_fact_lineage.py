"""Repeat the selected memory training lineage from fresh writer initialization."""
import argparse
import json
from pathlib import Path
import random
import time

import torch
from safetensors.torch import load_file, save_file

from scripts.fit_independent_fact_answer_training import check_fixed_read_path, load_read_path
from scripts.profile_adapted_readout import write_json
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.independent_fact_answer_training import train_answer_recurrent_batch
from tinymem.research.independent_fact_joint_update import calibrate_state_weight, train_joint_update_batch
from tinymem.research.independent_fact_correction_weight import train_correction_weight_batch
from tinymem.research.independent_fact_recurrent_training import pack_recurrent_batch
from tinymem.research.independent_fact_repeat_confirmation import build_manifest
from tinymem.research.independent_fact_updates import new_update_writer
from tinymem.research.readout_read import read_state_answer
from tinymem.research.study_runtime import REPOSITORY, execution_record, prepare_device, synchronize, validate_execution
from tinymem.research.update_protocol import file_sha256

SETTINGS = {
    'kind': 'independent_fact_lineage_confirmation_v1', 'seeds': [2027, 2028, 2029],
    'schedule_seed': 1337, 'steps_per_stage': 400, 'batch_streams': 16,
    'optimizer_steps_per_lineage': 1600, 'answer_sequences_per_lineage': 658944,
    'calibration_batches': 16, 'calibration_gradient_fraction': .25,
    'optimizer': {'lr': .001, 'weight_decay': .01, 'betas': [.9, .999], 'eps': 1e-8},
    'reader': 'even', 'cpu_threads': 4, 'max_new_tokens': 8,
    'normal_answers': 37056, 'total_reads': 38700, 'persistent_bytes': 66,
    'hard_scientific_cutoff': False, 'reuse_exploratory_programs': True,
    'probe_collision_policy': 'report_without_exclusion',
}


def frozen(writer):
    writer.zero_grad(set_to_none=True)
    return writer.requires_grad_(False).eval()


def train_lineage(*, output, seed, batches, histories, reader, bridge, worlds, steps_per_stage):
    """Train all dependent stages; each optimizer owns one fresh stage only."""
    if output.exists():
        raise FileExistsError(output)
    if len(batches) != 16 or type(steps_per_stage) is not int or steps_per_stage <= 0:
        raise ValueError('sixteen batches and positive stage length required')
    width = batches[0].histories.shape[-1]
    output.mkdir(parents=True)
    writer = new_update_writer(width, seed)
    save_file(writer.state_dict(), str(output/'fresh_initial.safetensors'))
    record = {'seed': seed, 'optimizer_steps': 0, 'stages': {},
              'fresh_initial_sha256': file_sha256(output/'fresh_initial.safetensors')}
    device = reader.model.device
    with (output/'metrics.jsonl').open('x') as metrics:
        def stage(name, model, kernel, *, start, state_weight=None):
            optimizer = torch.optim.AdamW(model.parameters(), **SETTINGS['optimizer'])
            if optimizer.state:
                raise ValueError('stage optimizer is not fresh')
            for step in range(steps_per_stage):
                synchronize(device); started = time.perf_counter()
                metric = kernel(model, batches[step % 16], optimizer)
                synchronize(device)
                metric.update(stage=name, step=step+1, batch=step % 16, seconds=time.perf_counter()-started)
                metrics.write(json.dumps(metric, allow_nan=False)+'\n'); metrics.flush()
                if step == 0 or (step+1) % 10 == 0:
                    print(json.dumps({'seed': seed, 'stage': name, 'step': step+1,
                                      'answer_ce': metric['answer_ce']}), flush=True)
            frozen(model)
            final = output/f'{name}_final.safetensors'
            save_file(model.state_dict(), str(final))
            record['optimizer_steps'] += steps_per_stage
            record['stages'][name] = {'start_sha256': file_sha256(start),
                'final_sha256': file_sha256(final), 'steps': steps_per_stage, 'state_weight': state_weight}
            write_json(output/f'{name}_progress.json', record)
            return model

        initializer = stage('initial', writer,
            lambda w,b,o: train_answer_recurrent_batch(w,b,o,reader=reader,bridge=bridge,
                                                       worlds=worlds,coordinate_weight=0,query_microbatch=6),
            start=output/'fresh_initial.safetensors')
        teacher_rows = []
        for code in range(16):
            hidden = histories[code].unsqueeze(0)
            with torch.no_grad():
                teacher_rows.append(initializer(initializer.empty(1), hidden, torch.ones(hidden.shape[:2], dtype=torch.bool)))
        teacher = LatentSlotState(torch.cat([s.values for s in teacher_rows]).detach().clone(),
                                  torch.cat([s.valid for s in teacher_rows]).detach().clone())
        save_file({'values': teacher.values, 'valid': teacher.valid}, str(output/'teacher_states.safetensors'))
        original = load_file(str(output/'initial_final.safetensors'))
        joint = new_update_writer(width, seed); joint.load_state_dict(original)
        calibration_optimizer = torch.optim.AdamW(joint.parameters(), **SETTINGS['optimizer'])
        calibration = []
        with (output/'calibration.jsonl').open('x') as handle:
            for index,batch in enumerate(batches):
                metric = train_joint_update_batch(joint,batch,calibration_optimizer,reader=reader,bridge=bridge,
                    worlds=worlds,targets=teacher,state_weight=0,measure_only=True)
                metric.update(batch=index)
                handle.write(json.dumps(metric,allow_nan=False)+'\n'); handle.flush()
                calibration.append(metric)
        if calibration_optimizer.state or any(not torch.equal(v, original[k]) for k,v in joint.state_dict().items()):
            raise ValueError('calibration changed optimizer or updater')
        joint.zero_grad(set_to_none=True)
        weight = calibrate_state_weight(calibration)
        record['state_weight'] = weight
        write_json(output/'calibration.json', {'state_weight': weight, 'optimizer_steps': 0, 'batches': 16})
        del calibration_optimizer
        stage('joint', joint,
            lambda w,b,o: train_joint_update_batch(w,b,o,reader=reader,bridge=bridge,worlds=worlds,
                                                  targets=teacher,state_weight=weight),
            start=output/'initial_final.safetensors', state_weight=weight)
        parent = load_file(str(output/'joint_final.safetensors'))
        arms = {}
        for arm,multiplier in [('uniform',1),('correction_weighted',2)]:
            updater = new_update_writer(width, seed); updater.load_state_dict(parent)
            arms[arm] = stage(arm, updater,
                lambda w,b,o: train_correction_weight_batch(w,b,o,reader=reader,bridge=bridge,worlds=worlds,
                    targets=teacher,state_weight=weight,correction_multiplier=multiplier),
                start=output/'joint_final.safetensors', state_weight=weight)
        if any(not torch.equal(v, original[k]) for k,v in initializer.state_dict().items()):
            raise ValueError('frozen initializer changed')
    write_json(output/'report.json', record)
    return initializer, arms, record


def require_declaration(path):
    d = json.loads(path.read_text())
    if d.get('settings') != SETTINGS or d.get('walltime_minutes') != 1440:
        raise ValueError('lineage declaration settings differ')
    required = {'src/tinymem/__init__.py', 'src/tinymem/research/__init__.py',
                'scripts/fit_independent_fact_lineage.py', 'scripts/evaluate_independent_fact_lineage.py',
                'docs/independent_fact_lineage_confirmation.md', 'independent_answer_training_declaration.json',
                'independent_repeat_confirmation_data.json', 'independent_repeat_confirmation_results/reference_replays.jsonl',
                'independent_repeat_confirmation_declaration.json',
                'independent_repeat_confirmation_results/complete.json',
                'independent_repeat_confirmation_results/report.json',
                'independent_repeat_confirmation_verification.json',
                'src/tinymem/research/independent_fact_joint_update.py',
                'src/tinymem/research/independent_fact_correction_weight.py',
                'src/tinymem/research/independent_fact_repeat_confirmation.py',
                'src/tinymem/research/independent_fact_content_probe.py',
                'scripts/evaluate_independent_fact_content_probe.py',
                'tests/test_independent_fact_lineage_training.py',
                'tests/test_independent_fact_lineage_evaluation.py',
                'artifacts/diagnostics/independent_fact_lineage_confirmation_20260912_v1/fit.slurm',
                'artifacts/diagnostics/independent_fact_lineage_confirmation_20260912_v1/freeze_launch.py'}
    if not required.issubset(d['file_sha256']):
        raise ValueError('lineage source coverage differs')
    for name,digest in d['file_sha256'].items():
        p = Path(name)
        if p.is_absolute() or '..' in p.parts or file_sha256(REPOSITORY/p) != digest:
            raise ValueError('lineage source or input differs: '+name)
    ancestor = json.loads((REPOSITORY/'independent_repeat_confirmation_declaration.json').read_text())
    if any(d['file_sha256'].get(k) != v for k,v in ancestor['file_sha256'].items()):
        raise ValueError('previous runtime dependency closure differs')
    base = REPOSITORY/'independent_repeat_confirmation_results'
    seal = json.loads((base/'complete.json').read_text())
    proof = json.loads((REPOSITORY/'independent_repeat_confirmation_verification.json').read_text())
    if (proof.get('verified') is not True
            or proof['declaration_sha256'] != file_sha256(REPOSITORY/'independent_repeat_confirmation_declaration.json')
            or seal['declaration_sha256'] != proof['declaration_sha256']
            or proof['completion_sha256'] != file_sha256(base/'complete.json')
            or proof['report_sha256'] != file_sha256(base/'report.json')
            or seal['files']['reference_replays.jsonl'] != file_sha256(base/'reference_replays.jsonl')):
        raise ValueError('parent reference proof differs')
    parent = json.loads((REPOSITORY/'independent_answer_training_declaration.json').read_text())
    if any(d['file_sha256'].get(k) != v for k,v in parent['file_sha256'].items()):
        raise ValueError('historical source or feature identity differs')
    return d,parent,{k:REPOSITORY/v for k,v in parent['paths'].items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declaration', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True, choices=SETTINGS['seeds'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    d,parent,paths = require_declaration(args.declaration)
    declaration_hash = file_sha256(args.declaration)
    manifest = json.loads(paths['manifest'].read_text())
    refresh = json.loads((REPOSITORY/'independent_repeat_confirmation_data.json').read_text())
    if refresh != build_manifest(manifest): raise ValueError('refresh manifest differs')
    device = prepare_device('cuda')
    torch.set_num_threads(SETTINGS['cpu_threads'])
    execution = execution_record(device)
    validate_execution(execution, expected=json.loads(paths['profile_protocol'].read_text())['execution'])
    histories = {int(k):v for k,v in load_file(str(paths['history_features'])).items()}
    features = {tuple(map(int,k.split('.'))):v for k,v in load_file(str(paths['event_features'])).items()}
    order = list(range(256)); random.Random(SETTINGS['schedule_seed']).shuffle(order)
    ordered = [manifest['training_streams'][i] for i in order]
    batches = [pack_recurrent_batch(ordered[i:i+16],histories,features) for i in range(0,256,16)]
    reader,bridge,rows,reader_hash = load_read_path(parent,paths,'even',device)
    bridge_snapshot = {k:v.clone() for k,v in bridge.state_dict().items()}
    args.output.mkdir(parents=True)
    write_json(args.output/'protocol.json', {'declaration_sha256':declaration_hash,'settings':SETTINGS,
        'seed':args.seed,'training_order':order,'execution':execution,'frozen_reader_sha256':reader_hash})
    initializer,arms,training = train_lineage(output=args.output/'training',seed=args.seed,batches=batches,
        histories=histories,reader=reader,bridge=bridge,worlds=rows,steps_per_stage=400)
    check_fixed_read_path(reader,bridge,reader_hash,bridge_snapshot)
    from scripts.evaluate_independent_fact_lineage import evaluate_lineage
    before = torch.tensor(rows[0].before_ids,device=device)
    questions = [torch.tensor(q.after_ids,device=device) for q in rows[0].queries]
    def read(state,query):
        actual = LatentSlotState(state.values.to(device).clone(),state.valid.to(device).clone())
        return read_state_answer(reader,bridge,actual,before,questions[query],max_new_tokens=8)
    references = [json.loads(line) for line in (REPOSITORY/'independent_repeat_confirmation_results/reference_replays.jsonl').read_text().splitlines()]
    evaluation = evaluate_lineage(output=args.output/'evaluation',initializer=initializer,updaters=arms,
        histories=histories,features=features,training_manifest=manifest,refresh_manifest=refresh,
        read=read,references=references)
    check_fixed_read_path(reader,bridge,reader_hash,bridge_snapshot)
    require_declaration(args.declaration)
    if file_sha256(args.declaration) != declaration_hash: raise ValueError('declaration changed')
    write_json(args.output/'report.json', {'status':'complete','seed':args.seed,'training':training,'evaluation':evaluation})
    write_json(args.output/'complete.json', {'declaration_sha256':declaration_hash,
        'files':{str(p.relative_to(args.output)):file_sha256(p) for p in args.output.rglob('*') if p.is_file()}})


if __name__ == '__main__':
    main()
