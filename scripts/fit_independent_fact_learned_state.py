"""Repair recurrent updates using states from a fixed answer-trained initializer."""
import argparse
import json
from pathlib import Path
import random
import time

import torch
from safetensors.torch import load_file, save_file

from scripts.evaluate_independent_fact_training_fit import READ_FIELDS, require_declaration as require_fit
from scripts.fit_independent_fact_answer_training import check_fixed_read_path, load_read_path
from scripts.fit_independent_fact_recurrent_training import collect_states, state_catalog
from scripts.profile_adapted_readout import write_json
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.independent_fact_learned_state import train_learned_state_batch
from tinymem.research.independent_fact_placement import placement_state
from tinymem.research.independent_fact_recurrent_training import pack_recurrent_batch
from tinymem.research.independent_fact_updates import new_update_writer
from tinymem.research.readout_read import read_state_answer
from tinymem.research.study_runtime import REPOSITORY, execution_record, prepare_device, validate_execution
from tinymem.research.update_protocol import file_sha256

ARMS = ('reset','recurrent')
SETTINGS = {'kind':'learned_state_update_repair_v1','arms':list(ARMS),'seed':1337,
    'steps_per_arm':400,'batch_streams':16,'training_horizon':4,'cpu_threads':4,
    'optimizer':{'lr':.001,'weight_decay':.01,'betas':[.9,.999],'eps':1e-8},
    'loss':'mean16_learned_target_mse_over_four_updates','initial_encoder_frozen':True,
    'reader':'even','states_per_model':1680,'new_normal_answers':20160,
    'swap_answers':384,'constant_answers':12,'exact_replays':96,'parent_initial_replays':192,
    'reused_baseline_answers':10080,'persistent_bytes':66,'max_new_tokens':8,
    'hard_scientific_cutoff':False,'full_answer_ce_continuation_control':False}


class SplitWriter:
    """Use the frozen initial writer only for empty state; otherwise use the updater."""
    def __init__(self, initializer, updater):
        self.initializer, self.updater = initializer, updater

    def empty(self, batch_size):
        return self.initializer.empty(batch_size)

    def __call__(self, state, hidden, valid):
        if not state.valid.any():
            return self.initializer(state,hidden,valid)
        if not state.valid.all():
            raise ValueError('split evaluation requires all-empty or all-valid states')
        return self.updater(state,hidden,valid)


def require_declaration(path):
    d=json.loads(path.read_text())
    if d.get('settings')!=SETTINGS or d.get('walltime_minutes')!=60:
        raise ValueError('repair settings differ')
    for name,digest in d['file_sha256'].items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts or file_sha256(REPOSITORY/relative)!=digest:
            raise ValueError('repair source or input differs: '+name)
    prior_path=REPOSITORY/d['fit_declaration']
    prior, parent, paths, parent_results=require_fit(prior_path)
    fit_results=REPOSITORY/d['fit_results']
    seal=json.loads((fit_results/'complete.json').read_text())
    if seal['declaration_sha256']!=file_sha256(prior_path):raise ValueError('training-fit parent identity differs')
    for name,digest in seal['files'].items():
        relative=str((fit_results/name).relative_to(REPOSITORY))
        if Path(name).name!=name or d['file_sha256'].get(relative)!=digest or file_sha256(fit_results/name)!=digest:
            raise ValueError('training-fit parent result differs')
    proof=json.loads((REPOSITORY/d['fit_proof']).read_text())
    if (proof.get('verified') is not True or proof.get('completion_sha256')!=file_sha256(fit_results/'complete.json')
            or proof.get('report_sha256')!=file_sha256(fit_results/'report.json') or proof.get('declaration_sha256')!=file_sha256(prior_path)):
        raise ValueError('training-fit proof differs')
    needed={d['fit_declaration'],d['fit_proof'],str((fit_results/'complete.json').relative_to(REPOSITORY)),
        'scripts/fit_independent_fact_learned_state.py','src/tinymem/research/independent_fact_learned_state.py',
        'docs/independent_fact_learned_state_repair.md',
        'artifacts/diagnostics/independent_fact_update_repair_20260911_v1/verify_repair.py'}
    if not needed.issubset(d['file_sha256']):raise ValueError('repair declaration coverage differs')
    return d,parent,paths,parent_results,fit_results


def baseline_records(parent_results,fit_results,catalog):
    result={}
    for row in map(json.loads,(parent_results/'predictions.jsonl').read_text().splitlines()):
        if row['model']=='answer_only' and row['fold']=='even' and row['state_id'] in catalog:
            result[row['state_id'],row['query_index']]={k:row[k] for k in READ_FIELDS}
    for row in map(json.loads,(fit_results/'predictions.jsonl').read_text().splitlines()):
        if row['model']=='answer_only' and row['geometry']=='serial_batch1':
            key=row['state_id'],row['query_index']; actual={k:row[k] for k in READ_FIELDS}
            if key in result and result[key]!=actual:raise ValueError('baseline overlap differs')
            result[key]=actual
    if set(result)!={(name,q) for name in catalog for q in range(6)}:raise ValueError('baseline answer coverage differs')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declaration',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    digest=file_sha256(args.declaration)
    d,parent,paths,parent_results,fit_results=require_declaration(args.declaration)
    manifest=json.loads(paths['manifest'].read_text())
    evaluation={'evaluation_streams':manifest['training_streams']+manifest['evaluation_streams']}
    catalog=state_catalog(evaluation)
    if len(catalog)!=1680:raise ValueError('repair state coverage differs')
    device=prepare_device('cuda');torch.set_num_threads(4)
    execution=execution_record(device)
    validate_execution(execution,expected=json.loads((parent_results/'protocol.json').read_text())['execution'])
    histories={int(k):v for k,v in load_file(str(paths['history_features'])).items()}
    features={tuple(map(int,k.split('.'))):v for k,v in load_file(str(paths['event_features'])).items()}
    checkpoint=load_file(str(parent_results/'answer_only_final.safetensors'))
    initializer=new_update_writer(2048,1337);initializer.load_state_dict(checkpoint);initializer.requires_grad_(False).eval()
    original_states=load_file(str(parent_results/'written_states.safetensors'))
    fit_states=load_file(str(fit_results/'written_states.safetensors'))
    baseline=collect_states(initializer,histories,features,evaluation)
    for name,state in baseline.items():
        source=fit_states if name.startswith('train-') else original_states
        key=f'answer_only/serial_batch1/{name}' if name.startswith('train-') else f'answer_only/{name}'
        if any(not torch.equal(getattr(state,p),source[key+'.'+p]) for p in ('values','valid')):
            raise ValueError('baseline state replay differs: '+name)
    targets=LatentSlotState(torch.cat([baseline[f'initial:{code:02d}'].values for code in range(16)]),
                           torch.cat([baseline[f'initial:{code:02d}'].valid for code in range(16)]))
    baseline_answers=baseline_records(parent_results,fit_results,catalog)
    order=list(range(256));random.Random(1337).shuffle(order)
    ordered=[manifest['training_streams'][i] for i in order]
    batches=[pack_recurrent_batch(ordered[i:i+16],histories,features) for i in range(0,256,16)]
    args.output.mkdir(parents=True)
    save_file({'values':targets.values,'valid':targets.valid},str(args.output/'teacher_states.safetensors'))
    write_json(args.output/'protocol.json',{'declaration_sha256':digest,'settings':SETTINGS,'execution':execution,
        'training_order':order,'state_catalog':catalog,'initializer_parameters':sum(v.numel() for v in checkpoint.values()),
        'updater_parameters':sum(v.numel() for v in checkpoint.values()),'baseline_cpu_states_replayed':1680})
    with (args.output/'baseline_predictions.jsonl').open('x') as handle:
        for (name,q),record in baseline_answers.items():handle.write(json.dumps({'state_id':name,'query_index':q,**record})+'\n')
    states={}
    with (args.output/'metrics.jsonl').open('x') as handle:
        for arm in ARMS:
            updater=new_update_writer(2048,1337);updater.load_state_dict(checkpoint)
            optimizer=torch.optim.AdamW(updater.parameters(),**SETTINGS['optimizer'])
            for step in range(400):
                start=time.perf_counter()
                metric=train_learned_state_batch(updater,batches[step%16],optimizer,targets=targets,arm=arm)
                metric.update(arm=arm,step=step+1,batch=step%16,seconds=time.perf_counter()-start)
                handle.write(json.dumps(metric,allow_nan=False)+'\n');handle.flush()
                if (step+1)%40==0:print(json.dumps(metric),flush=True)
            updater.zero_grad(set_to_none=True);updater.requires_grad_(False).eval()
            save_file(updater.state_dict(),str(args.output/(arm+'_final.safetensors')))
            restored=new_update_writer(2048,1337);restored.load_state_dict(load_file(str(args.output/(arm+'_final.safetensors'))));restored.requires_grad_(False).eval()
            model_states=collect_states(SplitWriter(initializer,restored),histories,features,evaluation)
            states.update({f'{arm}/{name}':state for name,state in model_states.items()})
            if any(not torch.equal(v,checkpoint[k]) for k,v in initializer.state_dict().items()):raise ValueError('initializer changed')
    save_file({f'{key}.{part}':getattr(state,part) for key,state in states.items() for part in ('values','valid')},str(args.output/'written_states.safetensors'))
    reader,bridge,rows,reader_hash=load_read_path(parent,paths,'even',device)
    bridge_snapshot={k:v.clone() for k,v in bridge.state_dict().items()}
    payload=load_file(str(args.output/'written_states.safetensors'),device=str(device))
    before=torch.tensor(rows[0].before_ids,device=device)
    questions=[torch.tensor(q.after_ids,device=device) for q in rows[0].queries]
    def read(state,q):return read_state_answer(reader,bridge,state,before,questions[q],max_new_tokens=8)
    def stored(key):return LatentSlotState(payload[key+'.values'],payload[key+'.valid'])
    references=[r for r in map(json.loads,(parent_results/'reference_replays.jsonl').read_text().splitlines()) if r['fold']=='even']
    with (args.output/'reference_replays.jsonl').open('x') as handle:
        for old in references:
            actual=read(placement_state(old['code'],device,'separate_fact0'),old['query_index'])
            if any(actual[k]!=old[k] for k in READ_FIELDS):raise ValueError('exact reference replay differs')
            handle.write(json.dumps({'code':old['code'],'query_index':old['query_index'],**actual})+'\n')
    counts={'normal':0,'swap':0,'constant':0,'reference':len(references)}
    with (args.output/'predictions.jsonl').open('x') as handle,(args.output/'controls.jsonl').open('x') as controls:
        for arm in ARMS:
            for name in catalog:
                for q in range(6):
                    actual=read(stored(f'{arm}/{name}'),q)
                    if name.startswith('initial:') and any(actual[k]!=baseline_answers[name,q][k] for k in READ_FIELDS):raise ValueError('frozen initial answer differs')
                    handle.write(json.dumps({'model':arm,'state_id':name,'query_index':q,**actual})+'\n');counts['normal']+=1
                if counts['normal']%600==0:handle.flush();print(json.dumps(counts),flush=True)
            for stream in manifest['evaluation_streams']:
                name=stream['id']+':step-16'; donor=f'fresh-{stream["initial_code"]^15:02d}-{stream["program"]}:step-16'
                for q in range(6):
                    actual=read(stored(f'{arm}/{donor}'),q)
                    controls.write(json.dumps({'model':arm,'condition':'swap','state_id':name,'donor_state_id':donor,'query_index':q,**actual})+'\n');counts['swap']+=1
            handle.flush();controls.flush()
        for condition in ('zero','no_memory'):
            state=LatentSlotState(torch.zeros(1,2,8,device=device),torch.full((1,2),condition=='zero',dtype=torch.bool,device=device))
            for q in range(6):
                controls.write(json.dumps({'condition':condition,'query_index':q,**read(state,q)})+'\n');counts['constant']+=1
    check_fixed_read_path(reader,bridge,reader_hash,bridge_snapshot)
    if any(not torch.equal(v.cpu(),getattr(states[k.rsplit('.',1)[0]],k.rsplit('.',1)[1])) for k,v in payload.items()):raise ValueError('state mutated during read')
    if counts!={'normal':20160,'swap':384,'constant':12,'reference':96}:raise ValueError('repair answer coverage differs')
    require_declaration(args.declaration)
    if file_sha256(args.declaration)!=digest:raise ValueError('repair declaration changed')
    write_json(args.output/'report.json',{'status':'complete','training_steps':800,'states':3360,'counts':counts,
        'baseline_answers_reused':10080,'baseline_cpu_states_replayed':1680,'parent_initial_replays':192,
        'initializer_unchanged':True,'reader_unchanged':True,'bridge_unchanged':True})
    write_json(args.output/'complete.json',{'declaration_sha256':digest,'files':{p.name:file_sha256(p) for p in args.output.iterdir()}})


if __name__=='__main__':main()
