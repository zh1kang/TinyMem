"""Matched reset and recurrent training followed by uninterrupted evaluation."""
import argparse
from collections import Counter
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

from scripts.evaluate_independent_fact_recurrence import require_initial_inputs
from scripts.fit_independent_fact_initial_write import require_inputs, require_update_inputs
from scripts.profile_adapted_readout import write_json
from tinymem.memory.readout_interface import ReadoutBridge
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.adapted_readout import configure_read_adapter
from tinymem.research.independent_fact_data import build_worlds, encode_worlds
from tinymem.research.independent_fact_placement import placement_state
from tinymem.research.independent_fact_updates import build_update_cases, new_update_writer
from tinymem.research.readout_experiment import _reader_hash
from tinymem.research.readout_read import read_state_answer
from tinymem.research.study_runtime import execution_record, prepare_device, validate_execution
from tinymem.research.update_protocol import file_sha256, load_shared_reader, shared_reader_identity

MODELS = ['baseline', *[f'{arm}-{seed}' for seed in (1337,1338,1339) for arm in ('reset','recurrent')]]
SETTINGS = {
    'models': MODELS, 'schedule_seeds':[1337,1338,1339], 'steps_per_arm':400,
    'training_steps':2400, 'batch_streams':16, 'training_horizon':4, 'evaluation_horizon':16,
    'training_streams':256, 'evaluation_streams':32, 'states_per_model':656,
    'generated_records':55104, 'reference_replays':192, 'baseline_answer_matches':1728,
    'lr':0.001, 'weight_decay':0.01, 'betas':[0.9,0.999], 'eps':1e-8, 'clip_norm':1.0,
    'writer_device':'cpu', 'cpu_threads':4, 'persistent_bytes':66, 'max_new_tokens':8,
    'loss':'half_all16_initial_mse16_plus_half_mean_four_updates_mse16',
    'schedule':'one_seeded_permutation_cycled_25_passes', 'checkpoint':'final_only',
    'evaluation_raw_feedback':True, 'hard_scientific_cutoff':False, 'automatic_followup':False,
}


def validate_manifest(manifest):
    worlds = build_worlds()
    train, evaluation = manifest['training_streams'], manifest['evaluation_streams']
    if len(train) != 256 or len(evaluation) != 32:
        raise ValueError('stream coverage differs')
    counts = Counter()
    for streams, horizon in ((train,4),(evaluation,16)):
        if len({s['id'] for s in streams}) != len(streams):
            raise ValueError('duplicate stream ID')
        for stream in streams:
            code = stream['initial_code']
            if type(code) is not int or not 0 <= code < 16 or len(stream['events']) != horizon:
                raise ValueError('invalid stream start or horizon')
            for step,event in enumerate(stream['events'],1):
                fact, bit = event['target_fact'], event['new_bit']
                if type(fact) is not int or not 0 <= fact < 4 or type(bit) is not int or bit not in (0,1):
                    raise ValueError('invalid event assignment')
                after = (code & ~(1 << fact)) | (bit << fact)
                text = worlds[after].cases[0].context.splitlines()[fact]
                action = 'R' if code == after else 'C'
                if event != {'step':step,'before_code':code,'after_code':after,'target_fact':fact,
                             'new_bit':bit,'action':action,'text':text}:
                    raise ValueError('event truth or text differs')
                if horizon == 4:
                    counts[code,fact,bit] += 1
                code = after
            if horizon == 16 and (stream['events'][7]['after_code'] != stream['initial_code'] ^ 15
                                  or code != stream['initial_code'] ^ 15
                                  or any(e['action'] != 'R' for e in stream['events'][8:])):
                raise ValueError('evaluation endpoint or repetition tail differs')
    if len(counts) != 128 or set(counts.values()) != {8}:
        raise ValueError('training transition exposure differs')
    for step in range(4):
        coverage = Counter((s['events'][step]['target_fact'],s['events'][step]['action']) for s in train)
        if len(coverage) != 8 or set(coverage.values()) != {32}:
            raise ValueError('training position exposure differs')
    orders = {tuple(e['target_fact'] for e in s['events']) for s in train}
    if orders != {(0,1,3,2),(1,2,0,3),(2,3,1,0),(3,0,2,1)}:
        raise ValueError('training orders differ')
    for stream in evaluation:
        targets = [e['target_fact'] for e in stream['events']]
        if any(tuple(targets[i:i+4]) in orders for i in range(13)):
            raise ValueError('evaluation order overlaps training')


def state_catalog(manifest):
    catalog = {f'initial:{code:02d}': {'code':code,'kind':'initial'} for code in range(16)}
    for case in build_update_cases():
        catalog[f'one:{case.case_id}'] = {'code':case.after_code,'kind':'one',
                                          'before_code':case.before_code,'fact':case.target_fact}
    for stream in manifest['evaluation_streams']:
        for event in stream['events']:
            catalog[f'{stream["id"]}:step-{event["step"]:02d}'] = {
                'code':event['after_code'],'kind':'stream','stream_id':stream['id'],
                'step':event['step'],'before_code':event['before_code'],'fact':event['target_fact']}
    return catalog


def collect_states(writer, histories, features, manifest):
    """Both arms use this same raw-feedback evaluation path."""
    states = {}
    def write(before, hidden):
        snapshot = before.values.clone()
        hidden = hidden.unsqueeze(0)
        result = writer(before,hidden,torch.ones(hidden.shape[:2],dtype=torch.bool))
        if (not torch.equal(before.values,snapshot) or result.nbytes != 66
                or result.values.shape != (1,2,8) or result.values.dtype != torch.float32
                or not result.valid.all() or not torch.isfinite(result.values).all()
                or bool((result.values.abs()>1).any())):
            raise ValueError('invalid or mutating writer state')
        return result
    with torch.inference_mode():
        for code in range(16):
            states[f'initial:{code:02d}'] = write(writer.empty(1),histories[code])
        for case in build_update_cases():
            states[f'one:{case.case_id}'] = write(states[f'initial:{case.before_code:02d}'],
                                                  features[case.target_fact,case.new_bit])
        for stream in manifest['evaluation_streams']:
            state = states[f'initial:{stream["initial_code"]:02d}']
            for event in stream['events']:
                state = write(state,features[event['target_fact'],event['new_bit']])
                states[f'{stream["id"]}:step-{event["step"]:02d}'] = state
    if set(states) != set(state_catalog(manifest)):
        raise ValueError('evaluation state coverage differs')
    return states


def main():
    from tinymem.research.independent_fact_recurrent_training import pack_recurrent_batch, train_recurrent_batch
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('placement','placement-declaration','placement-proof','inputs','updates','updates-proof',
                 'initial','initial-proof','output','declaration','manifest'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    d = json.loads(args.declaration.read_text())
    declaration_hash = file_sha256(args.declaration)
    if d['settings'] != SETTINGS or d['reader'] != shared_reader_identity() or file_sha256(args.manifest) != d['data_manifest_sha256']:
        raise ValueError('declaration settings or manifest differ')
    prior = require_inputs(args,d)
    require_update_inputs(args,d); require_initial_inputs(args,d)
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    device = prepare_device('cuda')
    torch.set_num_threads(4)
    execution = execution_record(device)
    validate_execution(execution,expected=d['execution'])
    args.output.mkdir(parents=True)
    histories = {int(k):v for k,v in load_file(str(args.initial/'history_features.safetensors')).items()}
    features = {tuple(map(int,k.split('.'))):v for k,v in load_file(str(args.initial/'event_features.safetensors')).items()}
    original = load_file(str(args.initial/'writer_final.safetensors'))
    orders = {}
    for seed in SETTINGS['schedule_seeds']:
        order = list(range(256)); random.Random(seed).shuffle(order); orders[str(seed)] = order
    catalog = state_catalog(manifest)
    write_json(args.output/'protocol.json',{'declaration_sha256':declaration_hash,'settings':SETTINGS,
        'execution':execution,'source_sha256':d['source_sha256'],'training_orders':orders,'state_catalog':catalog})
    shutil.copyfile(args.manifest,args.output/'data_manifest.json')
    states, state_metrics = {}, []
    with (args.output/'metrics.jsonl').open('x') as metrics:
        for model in MODELS:
            writer = new_update_writer(2048,1337)
            writer.load_state_dict(original)
            if model != 'baseline':
                arm,seed_text = model.split('-')
                ordered = [manifest['training_streams'][i] for i in orders[seed_text]]
                batches = [pack_recurrent_batch(ordered[i:i+16],histories,features) for i in range(0,256,16)]
                optimizer = torch.optim.AdamW(writer.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8)
                for step in range(400):
                    start = time.perf_counter()
                    metric = train_recurrent_batch(writer,batches[step%16],optimizer,arm=arm)
                    metric.update(model=model,step=step+1,batch=step%16,seconds=time.perf_counter()-start)
                    metrics.write(json.dumps(metric,allow_nan=False)+'\n'); metrics.flush()
                    if (step+1)%40 == 0:
                        print(json.dumps(metric),flush=True)
                del optimizer,batches
            writer.zero_grad(set_to_none=True)
            writer.requires_grad_(False).eval()
            checkpoint = args.output/f'writer_{model}.safetensors'
            save_file(writer.state_dict(),str(checkpoint))
            restored = new_update_writer(2048,1337)
            restored.load_state_dict(load_file(str(checkpoint)))
            restored.requires_grad_(False).eval()
            if any(not torch.equal(v,restored.state_dict()[k]) for k,v in writer.state_dict().items()):
                raise ValueError('checkpoint reload differs')
            model_states = collect_states(restored,histories,features,manifest)
            for name,state in model_states.items():
                key = model+'/'+name
                states[key] = state
                target = placement_state(catalog[name]['code'],torch.device('cpu'),'separate_fact0').values
                errors = (state.values-target).abs().flatten().tolist()
                state_metrics.append({'model':model,'state_id':name,'absolute_errors':errors,
                                      'state_mse':sum(float(x)**2 for x in errors)/16})
            del writer,restored,model_states
    save_file({f'{name}.{part}':getattr(state,part) for name,state in states.items()
               for part in ('values','valid')},str(args.output/'written_states.safetensors'))
    write_json(args.output/'state_metrics.json',state_metrics)
    frozen = {p.name:file_sha256(p) for p in args.output.iterdir()}
    old_reads = {(r['fold'],r['state_id'],r['query_index']):r for r in
                 map(json.loads,(args.initial/'predictions.jsonl').read_text().splitlines())}
    generated = replays = matches = 0
    reader_hashes = {}
    with (args.output/'predictions.jsonl').open('x') as handle, (args.output/'reference_replays.jsonl').open('x') as replay:
        for fold in ('even','odd'):
            directory = args.placement/'separate_fact0'/fold
            reader = load_shared_reader(d['reader'],device)
            if _reader_hash(reader) != prior['fits']['separate_fact0'][fold]['reader_initial_sha256']:
                raise ValueError('original reader differs')
            set_peft_model_state_dict(reader.model,load_file(str(directory/'reader_adapter/adapter_model.safetensors'),device=str(device)),adapter_name='default')
            configure_read_adapter(reader,trainable=False)
            reader_hash = _reader_hash(reader)
            if reader_hash != prior['fits']['separate_fact0'][fold]['reader_final_sha256']:
                raise ValueError('read adapter differs')
            bridge = ReadoutBridge(2048,'affine').to(device)
            bridge.load_state_dict({k.removeprefix('bridge.'):v for k,v in load_file(str(directory/'final.safetensors'),device=str(device)).items()})
            bridge.requires_grad_(False).eval()
            bridge_before = {k:v.clone() for k,v in bridge.state_dict().items()}
            rows = encode_worlds(reader,build_worlds())
            if json.loads(json.dumps([asdict(r) for r in rows])) != json.loads(args.inputs.read_text())['encodings']:
                raise ValueError('read encodings differ')
            def read(state,query,*,reader=reader,bridge=bridge,rows=rows):
                return read_state_answer(reader,bridge,state,torch.tensor(rows[0].before_ids,device=device),
                                         torch.tensor(rows[0].queries[query].after_ids,device=device),max_new_tokens=8)
            for record in map(json.loads,(directory/'predictions.jsonl').read_text().splitlines()):
                if record['condition'] != 'normal':
                    continue
                actual = read(placement_state(record['code'],device,'separate_fact0'),record['query_index'])
                if any(actual[k] != record[k] for k in actual):
                    raise ValueError('exact-state reference differs')
                replay.write(json.dumps({'fold':fold,'code':record['code'],'query_index':record['query_index'],**actual})+'\n')
                replays += 1
            payload = load_file(str(args.output/'written_states.safetensors'),device=str(device))
            for model in MODELS:
                for name in catalog:
                    key = model+'/'+name
                    state = LatentSlotState(payload[key+'.values'],payload[key+'.valid'])
                    for query in range(6):
                        actual = read(state,query)
                        if model == 'baseline' and catalog[name]['kind'] in ('initial','one'):
                            old_name = name if name.startswith('initial:') else 'learned:'+name.removeprefix('one:')
                            old = old_reads[fold,old_name,query]
                            if any(actual[k] != old[k] for k in actual):
                                raise ValueError('frozen baseline answer differs')
                            matches += 1
                        handle.write(json.dumps({'model':model,'fold':fold,'state_id':name,'query_index':query,**actual})+'\n')
                        generated += 1
                handle.flush()
                print(json.dumps({'fold':fold,'model':model,'generated':generated}),flush=True)
            if (_reader_hash(reader) != reader_hash or any(not torch.equal(v,bridge_before[k]) for k,v in bridge.state_dict().items())
                    or any(not torch.equal(v.cpu(),getattr(states[k.rsplit('.',1)[0]],k.rsplit('.',1)[1])) for k,v in payload.items())):
                raise ValueError('evaluation mutated fixed reader, bridge, or state')
            reader_hashes[fold] = reader_hash
            del read,reader,bridge,payload,state
            gc.collect(); torch.cuda.empty_cache()
    if (generated,replays,matches) != (55104,192,1728):
        raise ValueError('generated coverage differs')
    require_inputs(args,d); require_update_inputs(args,d); require_initial_inputs(args,d)
    validate_execution(execution_record(device),expected=d['execution'])
    if (file_sha256(args.declaration) != declaration_hash or file_sha256(args.manifest) != d['data_manifest_sha256']
            or any(file_sha256(args.output/name) != digest for name,digest in frozen.items())):
        raise ValueError('frozen execution inputs changed')
    write_json(args.output/'report.json',{'status':'complete','training_steps':2400,'models':MODELS,
        'generated_records':generated,'reference_replays':replays,'baseline_answer_matches':matches,
        'states':len(states),'reader_sha256':reader_hashes,'reader_unchanged':True,'bridge_unchanged':True,
        'state_unchanged_during_reads':True,'hard_scientific_cutoff':False})
    write_json(args.output/'complete.json',{'kind':'independent_fact_recurrent_training_complete_v1',
        'files':{p.name:file_sha256(p) for p in args.output.iterdir()}})


if __name__ == '__main__':
    main()
