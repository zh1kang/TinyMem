"""Two fixed complementary-parity fits, with evidence review separate from targets."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file, save_file

from scripts.profile_adapted_readout import frozen_hash, write_json
from scripts.qualify_independent_facts import sources as reference_sources
from tinymem.memory.readout_interface import ReadoutBridge
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.adapted_readout import configure_read_adapter
from tinymem.research.independent_fact_data import build_worlds, encode_worlds, oracle_state
from tinymem.research.independent_fact_fit_evaluation import evaluate_codes, summarize_codes
from tinymem.research.independent_fact_training import train_fact_step
from tinymem.research.paired_readout_evaluation import verify_state_payload
from tinymem.research.readout_experiment import _reader_hash
from tinymem.research.study_runtime import REPOSITORY, allocation_metrics, execution_record, prepare_device, synchronize, validate_execution
from tinymem.research.update_protocol import file_sha256, load_shared_reader, shared_reader_identity

SETTINGS = {'training_parities':[0,1], 'steps_per_fit':200, 'lr':.001, 'weight_decay':.01,
            'betas':[.9,.999], 'eps':1e-8, 'clip_norm':1.0, 'reader_mode':'eval', 'gradient_checkpointing':False,
            'checkpoint_selection':'final_only', 'max_new_tokens':8, 'persistent_bytes':66,
            'writer':'none', 'queries_per_world':6, 'reliability_target':.95,
            'control_gap_target':.4, 'hard_scientific_cutoff':False, 'seed_replications':1}


def sources():
    names=('src/tinymem/research/independent_fact_training.py','src/tinymem/research/independent_fact_fit_evaluation.py','scripts/fit_independent_facts.py')
    return {**reference_sources(), **{name:file_sha256(REPOSITORY/name) for name in names}}


def require_reference(directory, proof_path, declaration):
    report=json.loads((directory/'report.json').read_text())
    proof=json.loads(proof_path.read_text())
    protocol=json.loads((directory/'protocol.json').read_text())
    if (file_sha256(directory/'report.json')!=declaration['reference_report_sha256']
            or file_sha256(directory/'complete.json')!=declaration['reference_complete_sha256']
            or file_sha256(proof_path)!=declaration['reference_verification_sha256']
            or proof['verified'] is not True or proof['report_sha256']!=declaration['reference_report_sha256']
            or proof['declaration_sha256']!=declaration['reference_declaration_sha256']
            or proof['verifier_sha256']!=declaration['reference_verifier_v2_sha256']
            or proof['original_verifier_sha256']!=declaration['reference_original_verifier_sha256']
            or protocol['declaration_sha256']!=declaration['reference_declaration_sha256']
            or protocol['reader']!=declaration['reader'] or protocol['input_sha256']!=declaration['input_sha256']
            or report['status']!='complete' or report['training_steps']!=0 or report['reader_unchanged'] is not True
            or declaration['reference_interpretation']['training_informative'] is not True):
        raise ValueError('completed verified reference and explicit evidence interpretation are required')
    seal=json.loads((directory/'complete.json').read_text())
    if seal['kind']!='independent_fact_reference_complete_v1' or set(seal['files'])!={'protocol.json','report.json','predictions.jsonl'}:
        raise ValueError('reference completion coverage differs')
    for name,digest in seal['files'].items():
        if file_sha256(directory/name)!=digest:
            raise ValueError('reference artifact changed')
    return report


def verify_reloaded_bridge(original, restored):
    expected, actual = original.state_dict(), restored.state_dict()
    if set(expected)!=set(actual) or any(expected[k].dtype!=actual[k].dtype or expected[k].shape!=actual[k].shape
                                         or not torch.equal(expected[k],actual[k]) for k in expected):
        raise ValueError('bridge checkpoint tensor reload differs')


def verify_evaluation_identity(reader, expected_reader, expected_states, states):
    hashes = verify_state_payload(expected_states,{f'{c}.{part}':getattr(state,part) for c,state in states.items() for part in ('values','valid')})
    if _reader_hash(reader)!=expected_reader:
        raise ValueError('reader changed during evaluation')
    return hashes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','reference','reference-verification','initial-bridge','output','declaration'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    d=json.loads(args.declaration.read_text())
    declaration_hash=file_sha256(args.declaration)
    identity=shared_reader_identity()
    if (d['settings']!=SETTINGS or d['source_sha256']!=sources() or d['reader']!=identity
            or d['input_sha256']!=file_sha256(args.inputs) or d['initial_bridge_sha256']!=file_sha256(args.initial_bridge)):
        raise ValueError('fit settings, source, reader, inputs, or initial bridge differ')
    reference=require_reference(args.reference,args.reference_verification,d)
    device=prepare_device('cuda')
    execution=execution_record(device)
    validate_execution(execution,expected=d['execution'])
    initial={k.removeprefix('bridge.'):v for k,v in load_file(str(args.initial_bridge)).items() if k.startswith('bridge.')}
    expected_states={}
    for code in range(16):
        state=oracle_state(code,torch.device('cpu'))
        expected_states[f'{code}.values'],expected_states[f'{code}.valid']=state.values,state.valid
    args.output.mkdir(parents=True,exist_ok=False)
    save_file(expected_states,str(args.output/'states.safetensors'))
    write_json(args.output/'protocol.json',{'kind':'independent_fact_fit_v1','declaration_sha256':declaration_hash,
               'settings':SETTINGS,'execution':execution,'source_sha256':sources(),'reader':identity,
               'input_sha256':d['input_sha256'],'reference_report_sha256':d['reference_report_sha256'],
               'reference_verification_sha256':d['reference_verification_sha256'],'reference_interpretation':d['reference_interpretation'],
               'initial_bridge_sha256':d['initial_bridge_sha256'],'scope':'fixed_questions_unseen_combinations_only'})
    summaries={}
    for parity,name in ((0,'even'),(1,'odd')):
        reader=load_shared_reader(identity,device)
        original_reader=_reader_hash(reader)
        if original_reader!=reference['reader_sha256']:
            raise ValueError('fresh fit reader differs from measured original reference')
        rows=encode_worlds(reader,build_worlds())
        if json.loads(json.dumps([asdict(r) for r in rows]))!=json.loads(args.inputs.read_text())['encodings']:
            raise ValueError('native fit encodings differ from fixed inputs')
        training=[r for r in rows if r.code.bit_count()%2==parity]
        if len(training)!=8:
            raise ValueError('wrong training split size')
        directory=args.output/name
        directory.mkdir()
        adapters=configure_read_adapter(reader,trainable=True)
        base_before=frozen_hash(reader)
        bridge=ReadoutBridge(2048,'affine')
        bridge.load_state_dict(initial)
        if any(not torch.equal(value,initial[key]) for key,value in bridge.state_dict().items()):
            raise ValueError('initial bridge values differ')
        bridge.to(device)
        payload=load_file(str(args.output/'states.safetensors'),device=str(device))
        verify_state_payload(expected_states,payload)
        states={c:LatentSlotState(payload[f'{c}.values'],payload[f'{c}.valid']) for c in range(16)}
        parameters=[*bridge.parameters(),*adapters]
        optimizer=torch.optim.AdamW(parameters,lr=SETTINGS['lr'],weight_decay=SETTINGS['weight_decay'],betas=tuple(SETTINGS['betas']),eps=SETTINGS['eps'])
        save_file({'bridge.'+k:v for k,v in bridge.state_dict().items()},str(directory/'initial.safetensors'))
        if parity==1 and file_sha256(directory/'initial.safetensors')!=file_sha256(args.output/'even/initial.safetensors'):
            raise ValueError('complementary fit initialization differs')
        metrics=[]
        with (directory/'metrics.jsonl').open('x') as handle:
            for step in range(200):
                row=training[step%8]
                synchronize(device)
                start=time.perf_counter()
                metric=train_fact_step(reader,bridge,states[row.code],row.code,row.before_ids,row.queries,optimizer,adapter_parameters=adapters)
                synchronize(device)
                metric.update(step=step+1,history_id=row.history_id,seconds=time.perf_counter()-start,**allocation_metrics(device))
                metrics.append(metric)
                handle.write(json.dumps(metric,sort_keys=True)+'\n'); handle.flush()
                if step==0 or (step+1)%20==0:
                    print(json.dumps({'fit':name,**metric}),flush=True)
        if frozen_hash(reader)!=base_before:
            raise ValueError('frozen base or buffers changed')
        save_file({'bridge.'+k:v for k,v in bridge.state_dict().items()},str(directory/'final.safetensors'))
        reader.model.save_pretrained(directory/'reader_adapter',save_embedding_layers=False)
        checkpoint_paths={'bridge':directory/'final.safetensors','adapter':directory/'reader_adapter/adapter_model.safetensors'}
        checkpoint_hashes={name:file_sha256(path) for name,path in checkpoint_paths.items()}
        trained_reader=_reader_hash(reader)
        reader.model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for parameter in adapters:
                parameter.zero_()
        set_peft_model_state_dict(reader.model,load_file(str(directory/'reader_adapter/adapter_model.safetensors'),device=str(device)),adapter_name='default')
        configure_read_adapter(reader,trainable=False)
        if _reader_hash(reader)!=trained_reader:
            raise ValueError('adapter checkpoint reload differs')
        restored=ReadoutBridge(2048,'affine').to(device)
        restored.load_state_dict({k.removeprefix('bridge.'):v for k,v in load_file(str(directory/'final.safetensors'),device=str(device)).items()})
        restored.requires_grad_(False).eval()
        verify_reloaded_bridge(bridge,restored)
        with torch.inference_mode():
            for code,state in states.items():
                if not torch.equal(bridge(state),restored(state)):
                    raise ValueError('bridge checkpoint reload differs')
        state_hashes=verify_state_payload(expected_states,{f'{c}.{part}':getattr(state,part) for c,state in states.items() for part in ('values','valid')})
        predictions=evaluate_codes(reader,restored,rows,states)
        after_states=verify_evaluation_identity(reader,trained_reader,expected_states,states)
        if {name:file_sha256(path) for name,path in checkpoint_paths.items()}!=checkpoint_hashes:
            raise ValueError('saved checkpoint changed during reload or evaluation')
        if after_states!=state_hashes:
            raise ValueError('state changed during evaluation')
        with (directory/'predictions.jsonl').open('x') as handle:
            for record in predictions:
                handle.write(json.dumps(record,sort_keys=True)+'\n')
        summaries[name]={**summarize_codes(predictions,rows,parity),'completed_steps':200,'training_seconds':sum(m['seconds'] for m in metrics),
                         'reader_initial_sha256':original_reader,'reader_final_sha256':trained_reader,'frozen_base_unchanged':True,
                         'checkpoint_reload_exact':True,'final_checkpoint_sha256':checkpoint_hashes,'state_payload_sha256':state_hashes,'state_unchanged':True,
                         'shared_parameters':{'encoder':0,'bridge':sum(p.numel() for p in bridge.parameters()),'adapter':sum(p.numel() for p in adapters)}}
        write_json(directory/'summary.json',summaries[name])
        print(json.dumps({'fit':name,'summary':summaries[name]}),flush=True)
        del reader,bridge,restored,optimizer,parameters,adapters,states,payload,state,parameter
        gc.collect(); torch.cuda.empty_cache()
    if sources()!=d['source_sha256'] or file_sha256(args.declaration)!=declaration_hash or file_sha256(args.inputs)!=d['input_sha256'] or file_sha256(args.initial_bridge)!=d['initial_bridge_sha256']:
        raise ValueError('fit execution input changed')
    require_reference(args.reference,args.reference_verification,d)
    validate_execution(execution_record(device),expected=execution)
    write_json(args.output/'report.json',{'status':'complete','fits':summaries,'training_steps':400,'predictions':504,
               'pooled_heldout_known':{'correct':sum(s['splits']['heldout']['scores']['normal']['known']['correct'] for s in summaries.values()),'total':64},
               'independent_seed_replications':1,'scientific_decision':'review_generalization_controls_and_errors','hard_scientific_cutoff':False})
    write_json(args.output/'complete.json',{'kind':'independent_fact_fit_complete_v1','files':{str(p.relative_to(args.output)):file_sha256(p) for p in args.output.rglob('*') if p.is_file()}})


if __name__=='__main__':
    main()
