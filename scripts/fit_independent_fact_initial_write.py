"""One joint text-initialization and update fit with fixed readers."""
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

from scripts.profile_adapted_readout import write_json
from tinymem.memory.readout_interface import ReadoutBridge
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.adapted_readout import configure_read_adapter, frozen_history_features
from tinymem.research.independent_fact_data import build_worlds, encode_worlds
from tinymem.research.independent_fact_placement import placement_state
from tinymem.research.independent_fact_updates import build_update_cases, new_update_writer, pack_update_batch
from tinymem.research.independent_fact_initial_write import pack_initial_update_batch, train_initial_update_batch
from tinymem.research.readout_experiment import _reader_hash
from tinymem.research.readout_read import read_state_answer
from tinymem.research.study_runtime import REPOSITORY, execution_record, prepare_device, validate_execution
from tinymem.research.update_protocol import file_sha256, load_shared_reader, shared_reader_identity

SETTINGS = {'seed':1337, 'steps':400, 'batch_size':16, 'lr':0.001, 'weight_decay':0.01,
            'betas':[0.9,0.999], 'eps':1e-8, 'clip_norm':1.0, 'hidden_width':64,
            'loss':'half_initial_mse16_plus_half_updated_mse16_full_gradient', 'writer_device':'cpu', 'cpu_threads':4,
            'schedule':'one_python_random1337_permutation_cycled_four_batches',
            'checkpoint':'final_only', 'max_new_tokens':8, 'training_cases':64, 'heldout_cases':64,
            'generated_records':3264, 'reference_replays':192, 'persistent_bytes':66,
            'hard_scientific_cutoff':False, 'automatic_followup':False, 'initial_histories':16, 'training_initial_codes':list(range(15)),
            'initial_state_input':'empty_then_own_raw_state', 'canonical_start_diagnostic':True}


def require_inputs(args, d):
    for key,path in {'placement_report_sha256':args.placement/'report.json',
                     'placement_complete_sha256':args.placement/'complete.json',
                     'placement_declaration_sha256':args.placement_declaration,
                     'placement_proof_sha256':args.placement_proof, 'input_sha256':args.inputs}.items():
        if file_sha256(path) != d[key]:
            raise ValueError('selected input changed: ' + key)
    proof = json.loads(args.placement_proof.read_text())
    old = json.loads(args.placement_declaration.read_text())
    if (proof != {'verified':True,'artifact_files':40,'training_steps':800,'predictions':1008,
                  'fixed_sequences':3024,'auxiliary_sequences':1,'report_sha256':d['placement_report_sha256'],
                  'declaration_sha256':d['placement_declaration_sha256']}
            or old['reader'] != d['reader'] or old['execution'] != d['execution']
            or old['input_sha256'] != d['input_sha256']):
        raise ValueError('verified placement inputs required')
    seal = json.loads((args.placement/'complete.json').read_text())
    if len(seal['files']) != 40 or any(file_sha256(args.placement/name) != h for name,h in seal['files'].items()):
        raise ValueError('placement artifact changed')
    if any(file_sha256(REPOSITORY/name) != h for name,h in d['source_sha256'].items()):
        raise ValueError('execution source changed')
    return json.loads((args.placement/'report.json').read_text())


def require_update_inputs(args, d):
    proof = json.loads(args.updates_proof.read_text())
    expected = {'verified':True,'artifact_files':11,'training_steps':400,'generated_records':1536,
                'reference_replays':192,'report_sha256':d['updates_report_sha256'],
                'declaration_sha256':d['updates_declaration_sha256']}
    seal = json.loads((args.updates/'complete.json').read_text())
    files = {'protocol.json','event_features.safetensors','events.json','writer_initial.safetensors',
             'metrics.jsonl','writer_final.safetensors','written_states.safetensors','state_metrics.json',
             'predictions.jsonl','reference_replays.jsonl','report.json'}
    if (proof != expected or file_sha256(args.updates_proof) != d['updates_proof_sha256']
            or file_sha256(args.updates/'report.json') != d['updates_report_sha256']
            or file_sha256(args.updates/'complete.json') != d['updates_complete_sha256']
            or seal.get('kind') != 'independent_fact_updates_complete_v1'
            or set(seal) != {'kind','files'} or set(seal['files']) != files
            or {p.name for p in args.updates.iterdir()} != files | {'complete.json'}
            or any(file_sha256(args.updates/name) != h for name,h in seal['files'].items())):
        raise ValueError('verified conditional update inputs changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('placement','placement-declaration','placement-proof','inputs','updates','updates-proof','output','declaration'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    d = json.loads(args.declaration.read_text())
    declaration_hash = file_sha256(args.declaration)
    if d['settings'] != SETTINGS or d['reader'] != shared_reader_identity():
        raise ValueError('settings or reader identity changed')
    prior = require_inputs(args,d)
    require_update_inputs(args,d)
    device = prepare_device('cuda')
    torch.set_num_threads(4)
    execution = execution_record(device)
    validate_execution(execution,expected=d['execution'])
    cases = build_update_cases()
    training = [case for case in cases if case.split == 'train']
    random.Random(1337).shuffle(training)
    if len(training) != 64:
        raise ValueError('training coverage differs')
    args.output.mkdir(parents=True)
    write_json(args.output/'protocol.json',{'declaration_sha256':declaration_hash,'settings':SETTINGS,
        'execution':execution,'source_sha256':d['source_sha256'],'cases':[asdict(c) for c in cases],
        'training_order':[c.case_id for c in training], 'scope':'text_initialization_and_update_with_privileged_state_supervision'})
    feature_reader = load_shared_reader(d['reader'],device)
    original_hash = _reader_hash(feature_reader)
    if original_hash != prior['fits']['separate_fact0']['even']['reader_initial_sha256']:
        raise ValueError('feature reader differs from original qualified reader')
    features = {tuple(map(int,k.split('.'))):v for k,v in load_file(str(args.updates/'event_features.safetensors')).items()}
    for name in ('event_features.safetensors','events.json'):
        shutil.copyfile(args.updates/name,args.output/name)
    histories, history_records = {}, []
    for world in build_worlds():
        text = world.cases[0].context
        ids = feature_reader.tokenizer.encode(text,add_special_tokens=False)
        histories[world.code] = frozen_history_features(feature_reader,ids)
        history_records.append({'code':world.code,'history_text':text,'token_ids':ids})
    if len(histories) != 16 or _reader_hash(feature_reader) != original_hash:
        raise ValueError('history feature coverage or original reader changed')
    save_file({str(code):v for code,v in histories.items()},str(args.output/'history_features.safetensors'))
    write_json(args.output/'histories.json',history_records)
    del feature_reader
    gc.collect(); torch.cuda.empty_cache()
    writer = new_update_writer(2048,1337)
    save_file(writer.state_dict(),str(args.output/'writer_initial.safetensors'))
    optimizer = torch.optim.AdamW(writer.parameters(),lr=0.001,weight_decay=0.01,betas=(0.9,0.999),eps=1e-8)
    batches = [pack_initial_update_batch(training[i:i+16],histories,features) for i in range(0,64,16)]
    with (args.output/'metrics.jsonl').open('x') as handle:
        for step in range(400):
            start = time.perf_counter()
            metric = train_initial_update_batch(writer,batches[step%4],optimizer)
            metric.update(step=step+1,seconds=time.perf_counter()-start,batch=step%4)
            handle.write(json.dumps(metric,allow_nan=False)+'\n'); handle.flush()
            if (step+1)%40 == 0:
                print(json.dumps(metric),flush=True)
    writer.zero_grad(set_to_none=True)
    writer.requires_grad_(False).eval()
    save_file(writer.state_dict(),str(args.output/'writer_final.safetensors'))
    restored = new_update_writer(2048,1337)
    restored.load_state_dict(load_file(str(args.output/'writer_final.safetensors')))
    restored.requires_grad_(False).eval()
    if any(not torch.equal(v,restored.state_dict()[k]) for k,v in writer.state_dict().items()):
        raise ValueError('writer checkpoint reload differs')
    states, state_metrics = {}, []
    def check_state(name, state, reference, target):
        if (not torch.equal(state.values,reference.values) or not torch.equal(state.valid,reference.valid)
                or state.nbytes != 66 or state.values.shape != (1,2,8) or not bool(state.valid.all())
                or not bool(torch.isfinite(state.values).all()) or bool((state.values.abs()>1).any())):
            raise ValueError('state or reload check failed')
        states[name] = state
        errors = (state.values-target).abs().flatten().tolist()
        state_metrics.append({'state_id':name,'absolute_errors':errors,
                              'state_mse':sum(float(x)**2 for x in errors)/16})
    with torch.inference_mode():
        for code, hidden in histories.items():
            valid = torch.ones(1,hidden.shape[0],dtype=torch.bool)
            state = restored(restored.empty(1),hidden.unsqueeze(0),valid)
            reference = writer(writer.empty(1),hidden.unsqueeze(0),valid)
            check_state(f'initial:{code:02d}',state,reference,
                        placement_state(code,torch.device('cpu'),'separate_fact0').values)
        for case in cases:
            canonical, hidden, valid, target = pack_update_batch([case],features)
            for route, before in (('learned',states[f'initial:{case.before_code:02d}']),('canonical',canonical)):
                before_copy = before.values.clone()
                state = restored(before,hidden,valid)
                reference = writer(before,hidden,valid)
                check_state(f'{route}:{case.case_id}',state,reference,target)
                if not torch.equal(before.values,before_copy):
                    raise ValueError('writer mutated initial state')
    save_file({f'{name}.{part}':getattr(state,part) for name,state in states.items()
               for part in ('values','valid')},str(args.output/'written_states.safetensors'))
    write_json(args.output/'state_metrics.json',state_metrics)
    frozen_files = {p.name:file_sha256(p) for p in args.output.iterdir() if p.is_file()}
    generated, replays, reader_hashes = 0,0,{}
    with (args.output/'predictions.jsonl').open('x') as handle, (args.output/'reference_replays.jsonl').open('x') as replay_handle:
        for fold in ('even','odd'):
            directory = args.placement/'separate_fact0'/fold
            reader = load_shared_reader(d['reader'],device)
            if _reader_hash(reader) != original_hash:
                raise ValueError('original reader changed')
            set_peft_model_state_dict(reader.model,load_file(str(directory/'reader_adapter/adapter_model.safetensors'),device=str(device)),adapter_name='default')
            configure_read_adapter(reader,trainable=False)
            reader_hash = _reader_hash(reader)
            if reader_hash != prior['fits']['separate_fact0'][fold]['reader_final_sha256']:
                raise ValueError('frozen update reader differs')
            bridge = ReadoutBridge(2048,'affine').to(device)
            bridge.load_state_dict({k.removeprefix('bridge.'):v for k,v in load_file(str(directory/'final.safetensors'),device=str(device)).items()})
            bridge.requires_grad_(False).eval()
            bridge_before = {k:v.clone() for k,v in bridge.state_dict().items()}
            rows = encode_worlds(reader,build_worlds())
            if json.loads(json.dumps([asdict(r) for r in rows])) != json.loads(args.inputs.read_text())['encodings']:
                raise ValueError('native read inputs changed')
            def read(state, query, *, reader=reader, bridge=bridge, rows=rows):
                return read_state_answer(reader,bridge,state,torch.tensor(rows[0].before_ids,device=device),
                    torch.tensor(rows[0].queries[query].after_ids,device=device),max_new_tokens=8)
            saved = [json.loads(line) for line in (directory/'predictions.jsonl').read_text().splitlines()]
            for record in saved:
                if record['condition'] != 'normal':
                    continue
                actual = read(placement_state(record['code'],device,'separate_fact0'),record['query_index'])
                if any(actual[key] != record[key] for key in actual):
                    raise ValueError('frozen reference generation differs')
                replay_handle.write(json.dumps({'fold':fold,'code':record['code'],'query_index':record['query_index'],**actual})+'\n')
                replays += 1
            payload = load_file(str(args.output/'written_states.safetensors'),device=str(device))
            for name in states:
                state = LatentSlotState(payload[name+'.values'],payload[name+'.valid'])
                for query in range(6):
                    output = read(state,query)
                    handle.write(json.dumps({'fold':fold,'state_id':name,'query_index':query,**output})+'\n')
                    generated += 1
            if (_reader_hash(reader) != reader_hash or any(not torch.equal(v,bridge_before[k]) for k,v in bridge.state_dict().items())
                    or any(not torch.equal(v.cpu(),getattr(states[name.rsplit('.',1)[0]],name.rsplit('.',1)[1])) for name,v in payload.items())):
                raise ValueError('read-only evaluation changed model or state')
            reader_hashes[fold] = reader_hash
            print(json.dumps({'fold':fold,'generated_total':generated,'reference_replays':replays}),flush=True)
            del read,reader,bridge,payload,state
            gc.collect(); torch.cuda.empty_cache()
    if generated != 3264 or replays != 192:
        raise ValueError('read coverage differs')
    if any(file_sha256(args.output/name) != h for name,h in frozen_files.items()):
        raise ValueError('frozen writer artifact changed during evaluation')
    require_inputs(args,d)
    require_update_inputs(args,d)
    validate_execution(execution_record(device),expected=d['execution'])
    if file_sha256(args.declaration) != declaration_hash:
        raise ValueError('declaration changed')
    write_json(args.output/'report.json',{'status':'complete','training_steps':400,'generated_records':generated,
        'reference_replays':replays,'writer_parameters':sum(p.numel() for p in writer.parameters()),
        'writer_checkpoint_reload_exact':True,'reader_unchanged':True,'bridge_unchanged':True,'state_unchanged_during_reads':True,
        'feature_reader_sha256':original_hash,'reader_sha256':reader_hashes,'hard_scientific_cutoff':False})
    write_json(args.output/'complete.json',{'kind':'independent_fact_initial_write_complete_v1','files':{p.name:file_sha256(p) for p in args.output.iterdir() if p.is_file()}})


if __name__ == '__main__':
    main()
