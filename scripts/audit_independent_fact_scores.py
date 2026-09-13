"""Replay saved greedy reads and score complete candidate answers without training."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file

from scripts.fit_independent_facts import sources as fit_sources
from scripts.profile_adapted_readout import write_json
from tinymem.memory.readout_interface import ReadoutBridge
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.adapted_readout import configure_read_adapter
from tinymem.research.independent_fact_data import build_worlds, encode_worlds, oracle_state
from tinymem.research.independent_fact_scores import prefix_answer_log_probs
from tinymem.research.paired_readout_evaluation import verify_state_payload
from tinymem.research.readout_experiment import _reader_hash
from tinymem.research.readout_read import read_state_answer
from tinymem.research.study_runtime import REPOSITORY, allocation_metrics, execution_record, prepare_device, synchronize, validate_execution
from tinymem.research.update_protocol import file_sha256, shared_reader_identity, load_shared_reader

CANDIDATES=('bathroom','bedroom','garden','hallway','kitchen','office','unknown')
SETTINGS={'training_steps':0,'candidates':list(CANDIDATES),'score':'sum_token_log_probabilities_including_eos',
          'max_new_tokens':8,'expected_replays':216,'expected_fixed_sequences':1512,'expected_auxiliary_sequences':1,
          'candidate_tie_rule':'first_in_declared_order','scientific_scope':'post_hoc_diagnostic_of_existing_finite_code_tables',
          'original_greedy_scores_unchanged':True,'automatic_followup':False}


def sources():
    names=('scripts/audit_independent_fact_scores.py','src/tinymem/research/independent_fact_scores.py')
    return {**fit_sources(),**{name:file_sha256(REPOSITORY/name) for name in names}}


def require_fit(root, declaration_path, proof_path, declaration):
    expected={'fit_declaration_sha256':declaration_path,'fit_proof_sha256':proof_path,
              'fit_report_sha256':root/'report.json','fit_complete_sha256':root/'complete.json'}
    if any(file_sha256(path)!=declaration[key] for key,path in expected.items()):
        raise ValueError('selected completed fit identity differs')
    fit_declaration=json.loads(declaration_path.read_text())
    proof=json.loads(proof_path.read_text())
    report=json.loads((root/'report.json').read_text())
    seal=json.loads((root/'complete.json').read_text())
    if (proof!={'verified':True,'artifact_files':19,'training_steps':400,'predictions':504,'generation_records_per_fit':252,
                'report_sha256':declaration['fit_report_sha256'],'declaration_sha256':declaration['fit_declaration_sha256']}
            or report['status']!='complete' or report['training_steps']!=400 or report['predictions']!=504
            or fit_declaration['reader']!=declaration['reader'] or fit_declaration['execution']!=declaration['execution']
            or fit_declaration['input_sha256']!=declaration['input_sha256']
            or seal['kind']!='independent_fact_fit_complete_v1' or len(seal['files'])!=19):
        raise ValueError('verified completed fit is required')
    for name,digest in seal['files'].items():
        if file_sha256(root/name)!=digest:
            raise ValueError('saved fit artifact changed')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('fit','fit-declaration','fit-proof','inputs','output','declaration'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    d=json.loads(args.declaration.read_text())
    declaration_hash=file_sha256(args.declaration)
    identity=shared_reader_identity()
    if (d['settings']!=SETTINGS or d['source_sha256']!=sources() or d['reader']!=identity
            or d['input_sha256']!=file_sha256(args.inputs)):
        raise ValueError('margin audit source, settings, reader, or inputs differ')
    fit=require_fit(args.fit,args.fit_declaration,args.fit_proof,d)
    device=prepare_device('cuda')
    execution=execution_record(device)
    validate_execution(execution,expected=d['execution'])
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'protocol.json',{'kind':'independent_fact_margins_v1','declaration_sha256':declaration_hash,
               'settings':SETTINGS,'execution':execution,'source_sha256':sources(),'reader':identity,
               'input_sha256':d['input_sha256'],'fit_report_sha256':d['fit_report_sha256'],
               'fit_declaration_sha256':d['fit_declaration_sha256'],'fit_proof_sha256':d['fit_proof_sha256']})
    counts={'greedy_replays':0,'exact_replays':0,'fixed_sequences':0,'auxiliary_sequences':0}
    arms={}
    expected_payload={f'{c}.{part}':getattr(oracle_state(c,torch.device('cpu')),part) for c in range(16) for part in ('values','valid')}
    with (args.output/'replays.jsonl').open('x') as replay_handle, (args.output/'scores.jsonl').open('x') as score_handle:
        for arm in ('even','odd'):
            start=time.perf_counter()
            reader=load_shared_reader(identity,device)
            if _reader_hash(reader)!=fit['fits'][arm]['reader_initial_sha256']:
                raise ValueError('original reader identity differs')
            set_peft_model_state_dict(reader.model,load_file(str(args.fit/arm/'reader_adapter/adapter_model.safetensors'),device=str(device)),adapter_name='default')
            configure_read_adapter(reader,trainable=False)
            reader_hash=_reader_hash(reader)
            if reader_hash!=fit['fits'][arm]['reader_final_sha256']:
                raise ValueError('saved adapted reader differs')
            bridge=ReadoutBridge(2048,'affine').to(device)
            bridge_payload={k.removeprefix('bridge.'):v for k,v in load_file(str(args.fit/arm/'final.safetensors'),device=str(device)).items()}
            bridge.load_state_dict(bridge_payload)
            bridge.requires_grad_(False).eval()
            rows=encode_worlds(reader,build_worlds())
            if json.loads(json.dumps([asdict(r) for r in rows]))!=json.loads(args.inputs.read_text())['encodings']:
                raise ValueError('native score inputs differ from the fit')
            payload=load_file(str(args.fit/'states.safetensors'),device=str(device))
            state_hashes=verify_state_payload(expected_payload,payload)
            states={c:LatentSlotState(payload[f'{c}.values'],payload[f'{c}.valid']) for c in range(16)}
            states['zero']=LatentSlotState(torch.zeros(1,2,8,device=device),torch.ones(1,2,dtype=torch.bool,device=device))
            states['no_memory']=LatentSlotState(torch.zeros(1,2,8,device=device),torch.zeros(1,2,dtype=torch.bool,device=device))
            selected=[r for r in map(json.loads,(args.fit/arm/'predictions.jsonl').read_text().splitlines()) if r['condition'] in ('normal','zero','no_memory')]
            if len(selected)!=108 or {(r['condition'],r['code'],r['query_index']) for r in selected}!={('normal',c,i) for c in range(16) for i in range(6)}|{(cond,None,i) for cond in ('zero','no_memory') for i in range(6)}:
                raise ValueError('saved read coverage differs')
            candidate_ids={label:reader.tokenizer.encode(label,add_special_tokens=False)+[reader.tokenizer.eos_token_id] for label in CANDIDATES}
            prompts=[]
            for saved in selected:
                condition,code,index=saved['condition'],saved['code'],saved['query_index']
                row=rows[code if condition=='normal' else 0]
                state=states[code if condition=='normal' else condition]
                before=torch.tensor(row.before_ids,device=device)
                after=torch.tensor(row.queries[index].after_ids,device=device)
                replay=read_state_answer(reader,bridge,state,before,after,max_new_tokens=8)
                for key in ('generated_ids','prediction','input_positions','memory_positions','native_envelope_tokens'):
                    if replay[key]!=saved[key]:
                        raise ValueError('saved greedy replay differs: '+key)
                record={'arm':arm,'condition':condition,'code':code,'query_index':index,**replay}
                replay_handle.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')
                counts['greedy_replays']+=1;counts['exact_replays']+=1
                prompts.append((record,before,after,state))
            replay_handle.flush()
            for record,before,after,state in prompts:
                with torch.inference_mode():
                    memory=bridge(state)
                key={k:record[k] for k in ('arm','condition','code','query_index')}
                candidates=[(label,'fixed_candidate',ids) for label,ids in candidate_ids.items()]
                if record['generated_ids'] not in candidate_ids.values():
                    candidates.append((record['prediction'],'saved_greedy',record['generated_ids']))
                for label,role,ids in candidates:
                    scored=prefix_answer_log_probs(reader,before,memory,after,torch.tensor(ids,device=device))
                    score_handle.write(json.dumps({**key,'candidate':label,'role':role,**scored},sort_keys=True,allow_nan=False)+'\n')
                    counts['fixed_sequences' if role=='fixed_candidate' else 'auxiliary_sequences']+=1
            score_handle.flush()
            if (_reader_hash(reader)!=reader_hash
                    or any(bridge.state_dict()[k].dtype!=v.dtype or not torch.equal(bridge.state_dict()[k],v) for k,v in bridge_payload.items())
                    or verify_state_payload(expected_payload,payload)!=state_hashes):
                raise ValueError('reader, bridge, or state changed during scoring')
            synchronize(device)
            arms[arm]={'reader_sha256':reader_hash,'reader_unchanged':True,'bridge_unchanged':True,'state_unchanged':True,
                       'state_payload_sha256':state_hashes,'seconds':time.perf_counter()-start,**allocation_metrics(device)}
            print(json.dumps({'arm':arm,**counts,**arms[arm]}),flush=True)
            del reader,bridge,bridge_payload,rows,payload,states,selected,prompts,record,before,after,state,memory
            gc.collect();torch.cuda.empty_cache()
    if counts!={'greedy_replays':216,'exact_replays':216,'fixed_sequences':1512,'auxiliary_sequences':1}:
        raise ValueError('completed diagnostic coverage differs')
    if sources()!=d['source_sha256'] or file_sha256(args.declaration)!=declaration_hash or file_sha256(args.inputs)!=d['input_sha256']:
        raise ValueError('margin audit execution input changed')
    require_fit(args.fit,args.fit_declaration,args.fit_proof,d)
    validate_execution(execution_record(device),expected=execution)
    write_json(args.output/'report.json',{'status':'complete','training_steps':0,**counts,'total_sequences':1513,'arms':arms,
               'scores_sha256':file_sha256(args.output/'scores.jsonl'),'replays_sha256':file_sha256(args.output/'replays.jsonl'),
               'scientific_scope':SETTINGS['scientific_scope'],'original_greedy_scores_unchanged':True})
    write_json(args.output/'complete.json',{'kind':'independent_fact_margins_complete_v1',
               'files':{p.name:file_sha256(p) for p in args.output.iterdir() if p.is_file()}})


if __name__=='__main__':
    main()
