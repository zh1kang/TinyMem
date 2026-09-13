"""Evaluate eight raw recurrent writes with the completed joint writer."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import shutil

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file, save_file

from scripts.profile_adapted_readout import write_json
from scripts.fit_independent_fact_initial_write import require_inputs, require_update_inputs
from tinymem.memory.readout_interface import ReadoutBridge
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.adapted_readout import configure_read_adapter
from tinymem.research.independent_fact_data import build_worlds, encode_worlds
from tinymem.research.independent_fact_placement import placement_state
from tinymem.research.independent_fact_recurrence import build_recurrence_streams
from tinymem.research.independent_fact_updates import new_update_writer
from tinymem.research.readout_experiment import _reader_hash
from tinymem.research.readout_read import read_state_answer
from tinymem.research.study_runtime import execution_record, prepare_device, validate_execution
from tinymem.research.update_protocol import file_sha256, load_shared_reader, shared_reader_identity

SETTINGS = {'training_steps':0,'streams':64,'steps_per_stream':8,'post_states':512,
            'families':['repeat','toggle'],'orders':['forward','reverse'],
            'writer_device':'cpu','cpu_threads':4,'max_new_tokens':8,'persistent_bytes':66,
            'generated_records':6144,'reference_replays':192,'first_step_answer_matches':768,
            'raw_state_feedback':True,'hard_scientific_cutoff':False,'automatic_followup':False}


def require_initial_inputs(args, d):
    expected = {'verified':True,'artifact_files':13,'training_steps':400,'generated_records':3264,
                'reference_replays':192,'report_sha256':d['initial_report_sha256'],
                'declaration_sha256':d['initial_declaration_sha256']}
    files = {'protocol.json','event_features.safetensors','events.json','history_features.safetensors',
             'histories.json','writer_initial.safetensors','metrics.jsonl','writer_final.safetensors',
             'written_states.safetensors','state_metrics.json','predictions.jsonl','reference_replays.jsonl','report.json'}
    seal = json.loads((args.initial/'complete.json').read_text())
    if (json.loads(args.initial_proof.read_text()) != expected
            or file_sha256(args.initial_proof) != d['initial_proof_sha256']
            or file_sha256(args.initial/'report.json') != d['initial_report_sha256']
            or file_sha256(args.initial/'complete.json') != d['initial_complete_sha256']
            or set(seal) != {'kind','files'} or set(seal['files']) != files
            or seal['kind'] != 'independent_fact_initial_write_complete_v1'
            or {p.name for p in args.initial.iterdir()} != files | {'complete.json'}
            or any(file_sha256(args.initial/name) != h for name,h in seal['files'].items())):
        raise ValueError('initial-write inputs changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('placement','placement-declaration','placement-proof','inputs','updates','updates-proof',
                 'initial','initial-proof','output','declaration'):
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
    require_initial_inputs(args,d)
    device = prepare_device('cuda')
    torch.set_num_threads(4)
    execution = execution_record(device)
    validate_execution(execution,expected=d['execution'])
    streams = build_recurrence_streams()
    args.output.mkdir(parents=True)
    write_json(args.output/'protocol.json',{'declaration_sha256':declaration_hash,'settings':SETTINGS,
        'execution':execution,'source_sha256':d['source_sha256'],'streams':[asdict(s) for s in streams],
        'scope':'frozen_writer_eight_step_recurrence'})
    writer = new_update_writer(2048,1337)
    writer.load_state_dict(load_file(str(args.initial/'writer_final.safetensors')))
    writer.requires_grad_(False).eval()
    writer_before = {k:v.clone() for k,v in writer.state_dict().items()}
    shutil.copyfile(args.initial/'writer_final.safetensors',args.output/'writer.safetensors')
    features = load_file(str(args.initial/'event_features.safetensors'))
    histories = load_file(str(args.initial/'history_features.safetensors'))
    old_payload = load_file(str(args.initial/'written_states.safetensors'))
    initial_states, states, state_metrics = {}, {}, []
    with torch.inference_mode():
        for code in range(16):
            hidden = histories[str(code)].unsqueeze(0)
            valid = torch.ones(hidden.shape[:2],dtype=torch.bool)
            state = writer(writer.empty(1),hidden,valid)
            name = f'initial:{code:02d}'
            if (not torch.equal(state.values,old_payload[name+'.values'])
                    or not torch.equal(state.valid,old_payload[name+'.valid'])):
                raise ValueError('initial-state reconstruction differs')
            initial_states[code] = state
        for stream in streams:
            initial = initial_states[stream.initial_code]
            previous = LatentSlotState(initial.values.clone(),initial.valid.clone())
            for event in stream.events:
                hidden = features[f'{event.target_fact}.{event.new_bit}'].unsqueeze(0)
                valid = torch.ones(hidden.shape[:2],dtype=torch.bool)
                before = previous.values.clone()
                state = writer(previous,hidden,valid)
                if (state.values.shape != (1,2,8) or state.nbytes != 66 or not bool(state.valid.all())
                        or not bool(torch.isfinite(state.values).all()) or bool((state.values.abs()>1).any())
                        or not torch.equal(previous.values,before)):
                    raise ValueError('state ownership or bounds differ')
                if event.step == 1:
                    name = f'learned:code-{event.before_code:02d}:fact-{event.target_fact}:value-{event.new_bit}'
                    if (not torch.equal(state.values,old_payload[name+'.values'])
                            or not torch.equal(state.valid,old_payload[name+'.valid'])):
                        raise ValueError('first recurrent step differs from saved one-step state')
                name = f'{stream.stream_id}:step-{event.step:02d}'
                states[name] = state
                target = placement_state(event.after_code,torch.device('cpu'),'separate_fact0').values
                errors = (state.values-target).abs().flatten().tolist()
                state_metrics.append({'state_id':name,'absolute_errors':errors,
                    'state_mse':sum(float(x)**2 for x in errors)/16,
                    'absolute_change_from_previous':(state.values-previous.values).abs().flatten().tolist(),
                    'absolute_change_from_initial':(state.values-initial.values).abs().flatten().tolist()})
                previous = state
    if len(states) != 512 or any(not torch.equal(v,writer_before[k]) for k,v in writer.state_dict().items()):
        raise ValueError('state count or frozen writer differs')
    save_file({f'{name}.{part}':getattr(state,part) for name,state in states.items()
               for part in ('values','valid')},str(args.output/'written_states.safetensors'))
    write_json(args.output/'state_metrics.json',state_metrics)
    frozen_files = {p.name:file_sha256(p) for p in args.output.iterdir() if p.is_file()}
    old_reads = {(r['fold'],r['state_id'],r['query_index']):r for r in
                 (json.loads(line) for line in (args.initial/'predictions.jsonl').read_text().splitlines())}
    original_hash = prior['fits']['separate_fact0']['even']['reader_initial_sha256']
    generated, replays, first_matches, reader_hashes = 0,0,0,{}
    with (args.output/'predictions.jsonl').open('x') as handle, (args.output/'reference_replays.jsonl').open('x') as replay_handle:
        for fold in ('even','odd'):
            directory = args.placement/'separate_fact0'/fold
            reader = load_shared_reader(d['reader'],device)
            if _reader_hash(reader) != original_hash:
                raise ValueError('original reader differs')
            set_peft_model_state_dict(reader.model,load_file(str(directory/'reader_adapter/adapter_model.safetensors'),
                device=str(device)),adapter_name='default')
            configure_read_adapter(reader,trainable=False)
            reader_hash = _reader_hash(reader)
            if reader_hash != prior['fits']['separate_fact0'][fold]['reader_final_sha256']:
                raise ValueError('frozen read adapter differs')
            bridge = ReadoutBridge(2048,'affine').to(device)
            bridge.load_state_dict({k.removeprefix('bridge.'):v for k,v in
                load_file(str(directory/'final.safetensors'),device=str(device)).items()})
            bridge.requires_grad_(False).eval()
            bridge_before = {k:v.clone() for k,v in bridge.state_dict().items()}
            rows = encode_worlds(reader,build_worlds())
            if json.loads(json.dumps([asdict(r) for r in rows])) != json.loads(args.inputs.read_text())['encodings']:
                raise ValueError('read prompt inputs differ')
            def read(state, query, *, reader=reader, bridge=bridge, rows=rows):
                return read_state_answer(reader,bridge,state,torch.tensor(rows[0].before_ids,device=device),
                    torch.tensor(rows[0].queries[query].after_ids,device=device),max_new_tokens=8)
            initial_gpu = load_file(str(args.initial/'written_states.safetensors'),device=str(device))
            for code in range(16):
                name = f'initial:{code:02d}'
                state = LatentSlotState(initial_gpu[name+'.values'],initial_gpu[name+'.valid'])
                for query in range(6):
                    actual = read(state,query)
                    old = old_reads[fold,name,query]
                    if any(actual[k] != old[k] for k in actual):
                        raise ValueError('initial reference answer differs')
                    replay_handle.write(json.dumps({'fold':fold,'state_id':name,'query_index':query,**actual})+'\n')
                    replays += 1
            payload = load_file(str(args.output/'written_states.safetensors'),device=str(device))
            for stream in streams:
                for event in stream.events:
                    name = f'{stream.stream_id}:step-{event.step:02d}'
                    state = LatentSlotState(payload[name+'.values'],payload[name+'.valid'])
                    for query in range(6):
                        actual = read(state,query)
                        if event.step == 1:
                            key = f'learned:code-{event.before_code:02d}:fact-{event.target_fact}:value-{event.new_bit}'
                            old = old_reads[fold,key,query]
                            if any(actual[k] != old[k] for k in actual):
                                raise ValueError('first-step generated answer differs')
                            first_matches += 1
                        handle.write(json.dumps({'fold':fold,'state_id':name,'query_index':query,**actual})+'\n')
                        generated += 1
            if (_reader_hash(reader) != reader_hash or any(not torch.equal(v,bridge_before[k]) for k,v in bridge.state_dict().items())
                    or any(not torch.equal(v.cpu(),getattr(states[name.rsplit('.',1)[0]],name.rsplit('.',1)[1]))
                           for name,v in payload.items())
                    or any(not torch.equal(v.cpu(),old_payload[name]) for name,v in initial_gpu.items())):
                raise ValueError('read-only evaluation changed reader, bridge, or state')
            reader_hashes[fold] = reader_hash
            print(json.dumps({'fold':fold,'generated_total':generated,'reference_replays':replays}),flush=True)
            del read,reader,bridge,payload,initial_gpu,state
            gc.collect(); torch.cuda.empty_cache()
    if (generated,replays,first_matches) != (6144,192,768):
        raise ValueError('read coverage differs')
    require_inputs(args,d); require_update_inputs(args,d); require_initial_inputs(args,d)
    validate_execution(execution_record(device),expected=d['execution'])
    if (file_sha256(args.declaration) != declaration_hash
            or any(file_sha256(args.output/name) != h for name,h in frozen_files.items())
            or any(not torch.equal(v,writer_before[k]) for k,v in writer.state_dict().items())):
        raise ValueError('frozen execution inputs changed')
    write_json(args.output/'report.json',{'status':'complete','training_steps':0,'streams':64,'post_states':512,
        'generated_records':generated,'reference_replays':replays,'first_step_answer_matches':first_matches,
        'initial_states_reproduced':16,'first_step_states_reproduced':64,'writer_unchanged':True,
        'reader_unchanged':True,'bridge_unchanged':True,'state_unchanged_during_reads':True,
        'reader_sha256':reader_hashes,'hard_scientific_cutoff':False})
    write_json(args.output/'complete.json',{'kind':'independent_fact_recurrence_complete_v1',
        'files':{p.name:file_sha256(p) for p in args.output.iterdir() if p.is_file()}})


if __name__ == '__main__':
    main()
