"""Matched four-fit placement study with fixed-budget training and answer scores."""
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
from scripts.audit_independent_fact_scores import sources as margin_sources, CANDIDATES
from scripts.fit_independent_facts import SETTINGS as FIT_SETTINGS, require_reference, verify_reloaded_bridge, verify_evaluation_identity
from tinymem.memory.readout_interface import ReadoutBridge
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.adapted_readout import configure_read_adapter
from tinymem.research.independent_fact_data import build_worlds, encode_worlds
from tinymem.research.independent_fact_fit_evaluation import evaluate_codes, summarize_codes
from tinymem.research.independent_fact_placement import placement_state, train_placement_step
from tinymem.research.independent_fact_placement_scores import candidate_sequences, score_complete_answer
from tinymem.research.paired_readout_evaluation import verify_state_payload
from tinymem.research.readout_experiment import _reader_hash
from tinymem.research.study_runtime import REPOSITORY, allocation_metrics, execution_record, prepare_device, synchronize, validate_execution
from tinymem.research.update_protocol import file_sha256, load_shared_reader, shared_reader_identity


LAYOUTS = ('control', 'separate_fact0')
SETTINGS = {**FIT_SETTINGS, 'layouts':list(LAYOUTS),
            'fit_order':[['control','even'],['control','odd'],['separate_fact0','even'],['separate_fact0','odd']],
            'fixed_candidates':list(CANDIDATES), 'candidate_score':'sum_token_log_probabilities_including_eos',
            'candidate_tie_rule':'first_in_declared_order', 'auxiliary_policy':'saved_ids_append_native_eos_only_if_missing',
            'empty_answer_policy':'score_native_eos_alone', 'expected_fixed_sequences':3024,
            'maximum_auxiliary_sequences':432, 'control_reproduction':'exact_metrics_checkpoints_predictions_and_scores',
            'automatic_followup':False}


def sources():
    names = ('src/tinymem/research/independent_fact_placement.py',
             'src/tinymem/research/independent_fact_placement_scores.py', 'scripts/fit_independent_fact_placement.py')
    return {**margin_sources(), **{name:file_sha256(REPOSITORY/name) for name in names}}


def require_baseline(args, declaration):
    paths = {'baseline_declaration_sha256':args.baseline_declaration,
             'baseline_proof_sha256':args.baseline_proof, 'margin_proof_sha256':args.margin_proof,
             'baseline_report_sha256':args.baseline/'report.json',
             'baseline_complete_sha256':args.baseline/'complete.json', 'baseline_scores_sha256':args.baseline_scores,
             'margin_declaration_sha256':args.margin_declaration,
             'margin_report_sha256':args.baseline_scores.parent/'report.json',
             'margin_complete_sha256':args.baseline_scores.parent/'complete.json',
             'margin_replays_sha256':args.baseline_scores.parent/'replays.jsonl',
             'margin_verifier_sha256':REPOSITORY/'artifacts/diagnostics/independent_fact_margins_20260910_v1/verify_margins.py'}
    if any(file_sha256(path)!=declaration[key] for key,path in paths.items()):
        raise ValueError('completed baseline identity differs')
    old = json.loads(args.baseline_declaration.read_text())
    proof = json.loads(args.baseline_proof.read_text())
    margin = json.loads(args.margin_proof.read_text())
    report = json.loads((args.baseline/'report.json').read_text())
    seal = json.loads((args.baseline/'complete.json').read_text())
    if (old['settings']!=FIT_SETTINGS or old['reader']!=declaration['reader'] or old['execution']!=declaration['execution']
            or old['input_sha256']!=declaration['input_sha256'] or old['initial_bridge_sha256']!=declaration['initial_bridge_sha256']
            or proof!={'verified':True,'artifact_files':19,'training_steps':400,'predictions':504,'generation_records_per_fit':252,
                       'report_sha256':declaration['baseline_report_sha256'],'declaration_sha256':declaration['baseline_declaration_sha256']}
            or margin!={'verified':True,'artifact_files':4,'training_steps':0,'greedy_replays':216,'exact_replays':216,
                        'fixed_sequences':1512,'auxiliary_sequences':1,'total_sequences':1513,
                        'scores_sha256':declaration['baseline_scores_sha256'],'replays_sha256':declaration['margin_replays_sha256'],
                        'report_sha256':declaration['margin_report_sha256'],'declaration_sha256':declaration['margin_declaration_sha256']}
            or report['status']!='complete' or report['training_steps']!=400 or report['predictions']!=504
            or seal['kind']!='independent_fact_fit_complete_v1' or len(seal['files'])!=19):
        raise ValueError('verified fixed baseline is required')
    if any(file_sha256(args.baseline/name)!=digest for name,digest in seal['files'].items()):
        raise ValueError('completed baseline artifact changed')
    margin_declaration=json.loads(args.margin_declaration.read_text())
    margin_seal=json.loads((args.baseline_scores.parent/'complete.json').read_text())
    if (margin_declaration['reader']!=declaration['reader'] or margin_declaration['execution']!=declaration['execution']
            or margin_declaration['fit_report_sha256']!=declaration['baseline_report_sha256']
            or margin_seal['kind']!='independent_fact_margins_complete_v1'
            or set(margin_seal['files'])!={'protocol.json','report.json','replays.jsonl','scores.jsonl'}
            or any(file_sha256(args.baseline_scores.parent/name)!=digest for name,digest in margin_seal['files'].items())):
        raise ValueError('completed margin provenance differs')
    return report


def require_control_reproduction(directory, arm, metrics, predictions, summary, baseline, baseline_scores):
    previous = json.loads((baseline/arm/'summary.json').read_text())
    if (summary['final_checkpoint_sha256']!=previous['final_checkpoint_sha256']
            or summary['reader_final_sha256']!=previous['reader_final_sha256']
            or summary['reader_initial_sha256']!=previous['reader_initial_sha256']):
        raise ValueError('control checkpoint reproduction differs')
    old_metrics = [json.loads(line) for line in (baseline/arm/'metrics.jsonl').read_text().splitlines()]
    signature = lambda m:{k:v for k,v in m.items() if k!='seconds' and not k.startswith('cuda_')}
    if [signature(m) for m in metrics]!=[signature(m) for m in old_metrics]:
        raise ValueError('control training metric reproduction differs')
    old_predictions = [json.loads(line) for line in (baseline/arm/'predictions.jsonl').read_text().splitlines()]
    if predictions!=old_predictions:
        raise ValueError('control prediction reproduction differs')
    old_scores = []
    for row in map(json.loads, baseline_scores.read_text().splitlines()):
        if row['arm']!=arm:
            continue
        row.pop('arm')
        if row['role']=='saved_greedy':
            row['role']='saved_greedy_completion'
        row['appended_eos']=False
        old_scores.append(row)
    current = [json.loads(line) for line in (directory/'scores.jsonl').read_text().splitlines()]
    if current!=old_scores:
        raise ValueError('control complete-score reproduction differs')


@torch.inference_mode()
def score_predictions(reader, bridge, rows, states, predictions, output):
    selected = [r for r in predictions if r['condition'] in ('normal','zero','no_memory')]
    expected = {('normal',c,i) for c in range(16) for i in range(6)} | {(cond,None,i) for cond in ('zero','no_memory') for i in range(6)}
    if len(selected)!=108 or {(r['condition'],r['code'],r['query_index']) for r in selected}!=expected:
        raise ValueError('scored prediction coverage differs')
    device = reader.model.device
    controls = {cond:LatentSlotState(torch.zeros(1,2,8,device=device),torch.full((1,2),cond=='zero',dtype=torch.bool,device=device)) for cond in ('zero','no_memory')}
    eos = reader.tokenizer.eos_token_id
    fixed = {label:reader.tokenizer.encode(label,add_special_tokens=False)+[eos] for label in CANDIDATES}
    counts = {'fixed_sequences':0, 'auxiliary_sequences':0}
    with output.open('x') as handle:
        for saved in selected:
            condition,code,index = saved['condition'],saved['code'],saved['query_index']
            row = rows[code if condition=='normal' else 0]
            state = states[code] if condition=='normal' else controls[condition]
            before,after = torch.tensor(row.before_ids,device=device),torch.tensor(row.queries[index].after_ids,device=device)
            memory = bridge(state)
            if (saved['memory_positions']!=memory.shape[0] or saved['native_envelope_tokens']!=before.numel()+after.numel()
                    or saved['input_positions']!=before.numel()+memory.shape[0]+after.numel()):
                raise ValueError('score prompt dimensions differ from generated input')
            for label,role,appended,ids in candidate_sequences(fixed,saved['generated_ids'],saved['prediction'],eos):
                score = score_complete_answer(reader,before,memory,after,torch.tensor(ids,device=device))
                record = {'condition':condition,'code':code,'query_index':index,'candidate':label,'role':role,'appended_eos':appended,**score}
                handle.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')
                counts['fixed_sequences' if role=='fixed_candidate' else 'auxiliary_sequences']+=1
    if counts['fixed_sequences']!=756 or not 0<=counts['auxiliary_sequences']<=108:
        raise ValueError('complete-score count differs')
    return {**counts,'scores_sha256':file_sha256(output)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','reference','reference-verification','initial-bridge','baseline','baseline-declaration','baseline-proof','margin-proof','margin-declaration','baseline-scores','output','declaration'):
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
    require_baseline(args,d)
    device=prepare_device('cuda')
    execution=execution_record(device)
    validate_execution(execution,expected=d['execution'])
    initial={k.removeprefix('bridge.'):v for k,v in load_file(str(args.initial_bridge)).items() if k.startswith('bridge.')}
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'protocol.json',{'kind':'independent_fact_placement_v1','declaration_sha256':declaration_hash,
               'settings':SETTINGS,'execution':execution,'source_sha256':sources(),'reader':identity,
               'input_sha256':d['input_sha256'],'reference_report_sha256':d['reference_report_sha256'],
               'reference_verification_sha256':d['reference_verification_sha256'],'reference_interpretation':d['reference_interpretation'],
               'initial_bridge_sha256':d['initial_bridge_sha256'],'baseline_report_sha256':d['baseline_report_sha256'],
               'baseline_proof_sha256':d['baseline_proof_sha256'],'margin_proof_sha256':d['margin_proof_sha256'],
               'scope':'post_hoc_matched_placement_on_inspected_code_tables'})
    summaries={layout:{} for layout in LAYOUTS}
    for layout in LAYOUTS:
        expected_states={}
        for code in range(16):
            state=placement_state(code,torch.device('cpu'),layout)
            expected_states[f'{code}.values'],expected_states[f'{code}.valid']=state.values,state.valid
        layout_root=args.output/layout
        layout_root.mkdir()
        save_file(expected_states,str(layout_root/'states.safetensors'))
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
            directory=layout_root/name
            directory.mkdir()
            adapters=configure_read_adapter(reader,trainable=True)
            base_before=frozen_hash(reader)
            bridge=ReadoutBridge(2048,'affine')
            bridge.load_state_dict(initial)
            if any(not torch.equal(value,initial[key]) for key,value in bridge.state_dict().items()):
                raise ValueError('initial bridge values differ')
            bridge.to(device)
            payload=load_file(str(layout_root/'states.safetensors'),device=str(device))
            verify_state_payload(expected_states,payload)
            states={c:LatentSlotState(payload[f'{c}.values'],payload[f'{c}.valid']) for c in range(16)}
            parameters=[*bridge.parameters(),*adapters]
            optimizer=torch.optim.AdamW(parameters,lr=SETTINGS['lr'],weight_decay=SETTINGS['weight_decay'],betas=tuple(SETTINGS['betas']),eps=SETTINGS['eps'])
            save_file({'bridge.'+k:v for k,v in bridge.state_dict().items()},str(directory/'initial.safetensors'))
            if parity==1 and file_sha256(directory/'initial.safetensors')!=file_sha256(layout_root/'even/initial.safetensors'):
                raise ValueError('complementary fit initialization differs')
            metrics=[]
            with (directory/'metrics.jsonl').open('x') as handle:
                for step in range(200):
                    row=training[step%8]
                    synchronize(device)
                    start=time.perf_counter()
                    metric=train_placement_step(reader,bridge,states[row.code],row.code,row.before_ids,row.queries,optimizer,adapter_parameters=adapters,layout=layout)
                    synchronize(device)
                    metric.update(step=step+1,history_id=row.history_id,seconds=time.perf_counter()-start,**allocation_metrics(device))
                    metrics.append(metric)
                    handle.write(json.dumps(metric,sort_keys=True)+'\n'); handle.flush()
                    if step==0 or (step+1)%20==0:
                        print(json.dumps({'layout':layout,'fit':name,**metric}),flush=True)
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
            summaries[layout][name]={**summarize_codes(predictions,rows,parity),'completed_steps':200,'training_seconds':sum(m['seconds'] for m in metrics),
                             'reader_initial_sha256':original_reader,'reader_final_sha256':trained_reader,'frozen_base_unchanged':True,
                             'checkpoint_reload_exact':True,'final_checkpoint_sha256':checkpoint_hashes,'state_payload_sha256':state_hashes,'state_unchanged':True,
                             'shared_parameters':{'encoder':0,'bridge':sum(p.numel() for p in bridge.parameters()),'adapter':sum(p.numel() for p in adapters)}}
            scoring=score_predictions(reader,restored,rows,states,predictions,directory/'scores.jsonl')
            if verify_evaluation_identity(reader,trained_reader,expected_states,states)!=state_hashes:
                raise ValueError('scoring state identity differs')
            verify_reloaded_bridge(bridge,restored)
            if {key:file_sha256(path) for key,path in checkpoint_paths.items()}!=checkpoint_hashes:
                raise ValueError('checkpoint changed during scoring')
            summaries[layout][name]['scoring']=scoring
            if layout=='control':
                require_control_reproduction(directory,name,metrics,predictions,summaries[layout][name],args.baseline,args.baseline_scores)
                summaries[layout][name]['control_reproduction_exact']=True
            write_json(directory/'summary.json',summaries[layout][name])
            print(json.dumps({'layout':layout,'fit':name,'summary':summaries[layout][name]}),flush=True)
            del reader,bridge,restored,optimizer,parameters,adapters,states,payload,state,parameter
            gc.collect(); torch.cuda.empty_cache()
    if sources()!=d['source_sha256'] or file_sha256(args.declaration)!=declaration_hash or file_sha256(args.inputs)!=d['input_sha256'] or file_sha256(args.initial_bridge)!=d['initial_bridge_sha256']:
        raise ValueError('fit execution input changed')
    require_reference(args.reference,args.reference_verification,d)
    validate_execution(execution_record(device),expected=execution)
    require_baseline(args,d)
    write_json(args.output/'report.json',{'status':'complete','fits':summaries,'training_steps':800,'predictions':1008,
               'fixed_sequences':sum(s['scoring']['fixed_sequences'] for fits in summaries.values() for s in fits.values()),
               'auxiliary_sequences':sum(s['scoring']['auxiliary_sequences'] for fits in summaries.values() for s in fits.values()),
               'control_reproduction_exact':True,'independent_seed_replications':1,
               'scientific_decision':'review_matched_placement_effects','hard_scientific_cutoff':False})
    write_json(args.output/'complete.json',{'kind':'independent_fact_placement_complete_v1','files':{str(p.relative_to(args.output)):file_sha256(p) for p in args.output.rglob('*') if p.is_file()}})



if __name__=='__main__':
    main()
