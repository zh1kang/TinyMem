"""Finite code-table evaluation and dependent coordinate comparisons."""
import torch

from tinymem.evaluation.longmemeval import normalized_answer
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.independent_fact_data import nearest_training_code
from tinymem.research.independent_fact_evaluation import full_text_records, summarize_reference
from tinymem.research.readout_read import read_state_answer


@torch.inference_mode()
def evaluate_codes(reader, bridge, rows, states):
    device = reader.model.device
    records = []

    def read(row, query_index, state):
        return read_state_answer(reader, bridge, state, torch.tensor(row.before_ids, device=device),
                                 torch.tensor(row.queries[query_index].after_ids, device=device), max_new_tokens=8)

    for row in rows:
        for i in range(6):
            records.append({'condition':'normal', 'code':row.code, 'query_index':i, **read(row, i, states[row.code])})
    for condition in ('zero','no_memory'):
        state = LatentSlotState(torch.zeros(1,2,8,device=device), torch.full((1,2), condition=='zero',dtype=torch.bool,device=device))
        for i in range(6):
            records.append({'condition':condition,'code':None,'query_index':i,**read(rows[0],i,state)})
    for recipient in (0,1):
        for axis in range(4):
            donor = recipient ^ (1 << axis)
            for i in range(6):
                records.append({'condition':'donor_replay','code':donor,'recipient_code':recipient,'axis':axis,
                                'query_index':i,**read(rows[recipient],i,states[donor])})
    records.extend(full_text_records(reader, rows))
    return records


def summarize_codes(records, rows, training_parity):
    if type(training_parity) is not int or training_parity not in (0,1):
        raise ValueError('training parity must be zero or one')
    if [r.code for r in rows] != list(range(16)):
        raise ValueError('expected all sixteen codes in fixed order')
    normal = [r for r in records if r['condition']=='normal']
    controls = [r for r in records if r['condition'] in ('zero','no_memory')]
    replay = [r for r in records if r['condition']=='donor_replay']
    text = [r for r in records if r['condition']=='full_text']
    if len(records)!=252 or len(normal)!=96 or len(controls)!=12 or len(replay)!=48 or len(text)!=96:
        raise ValueError('fit prediction coverage differs')
    if {(r['code'],r['query_index']) for r in normal}!={(c,i) for c in range(16) for i in range(6)}:
        raise ValueError('normal code coverage differs')
    if {(r['condition'],r['code'],r['query_index']) for r in controls}!={(c,None,i) for c in ('zero','no_memory') for i in range(6)}:
        raise ValueError('unique control coverage differs')
    if {(r['recipient_code'],r['axis'],r['code'],r['query_index']) for r in replay}!={(c,a,c^(1<<a),i) for c in (0,1) for a in range(4) for i in range(6)}:
        raise ValueError('direct donor replay coverage differs')
    table = {(r['code'],r['query_index']):r for r in normal}
    baseline = {(r['condition'],r['query_index']):r for r in controls}
    for r in replay:
        expected = table[r['code'],r['query_index']]
        if r['generated_ids']!=expected['generated_ids'] or r['prediction']!=expected['prediction']:
            raise ValueError('identical donor read inputs changed generated output')
    truth = {(row.code,i):q.answer for row in rows for i,q in enumerate(row.queries)}
    correct = {key:normalized_answer(r['prediction'])==truth[key] for key,r in table.items()}
    splits = {}
    for name, parity in (('train',training_parity),('heldout',1-training_parity)):
        codes = [c for c in range(16) if c.bit_count()%2==parity]
        scores = {}
        for condition in ('normal','zero','no_memory','nearest_training_world'):
            values = {}
            for c in codes:
                for i in range(6):
                    prediction = (table[c,i]['prediction'] if condition=='normal' else
                                  truth[nearest_training_code(c,training_parity),i] if condition=='nearest_training_world' else
                                  baseline[condition,i]['prediction'])
                    values[c,i] = normalized_answer(prediction)==truth[c,i]
            scores[condition] = {}
            for label,indices in (('known',range(4)),('absent',range(4,6))):
                n=sum(values[c,i] for c in codes for i in indices)
                total=len(codes)*len(indices)
                scores[condition][label]={'correct':n,'total':total,'accuracy':n/total,'reliability_target_met':n/total>=.95}
            scores[condition]['all_six_correct_worlds']=sum(all(values[c,i] for i in range(6)) for c in codes)
        per_fact = {}
        for i in range(4):
            per_fact[str(i)]={str(bit):{'correct':sum(correct[c,i] for c in codes if (c>>i)&1==bit),'total':4} for bit in (0,1)}
        splits[name]={'codes':codes,'scores':scores,'per_fact_and_value':per_fact,
                      'normal_known_gaps':{c:scores['normal']['known']['accuracy']-scores[c]['known']['accuracy'] for c in ('zero','no_memory','nearest_training_world')}}
    edges = []
    for left in range(16):
        for axis in range(4):
            right=left^(1<<axis)
            if left>right:
                continue
            train,held=(left,right) if left.bit_count()%2==training_parity else (right,left)
            edges.append({'left_code':left,'right_code':right,'axis':axis,
                          'target_answer_changed':normalized_answer(table[left,axis]['prediction'])!=normalized_answer(table[right,axis]['prediction']),
                          'target_both_correct':correct[left,axis] and correct[right,axis],
                          'untouched_known_answer_changes':sum(normalized_answer(table[left,i]['prediction'])!=normalized_answer(table[right,i]['prediction']) for i in range(4) if i!=axis),
                          'untouched_known_both_correct':sum(correct[left,i] and correct[right,i] for i in range(4) if i!=axis),
                          'untouched_train_to_heldout_correct_to_wrong':sum(correct[train,i] and not correct[held,i] for i in range(4) if i!=axis),
                          'untouched_heldout_to_train_correct_to_wrong':sum(correct[held,i] and not correct[train,i] for i in range(4) if i!=axis),
                          'absent_answer_changes':sum(normalized_answer(table[left,i]['prediction'])!=normalized_answer(table[right,i]['prediction']) for i in (4,5)),
                          'absent_both_correct':sum(correct[left,i] and correct[right,i] for i in (4,5))})
    totals = {key:sum(edge[key] for edge in edges) for key in edges[0] if key not in ('left_code','right_code','axis')}
    complements = []
    for left in range(8):
        right = left ^ 15
        complements.append({'left_code':left, 'right_code':right,
                            'known_answer_changes':sum(normalized_answer(table[left,i]['prediction'])!=normalized_answer(table[right,i]['prediction']) for i in range(4)),
                            'known_both_correct':sum(correct[left,i] and correct[right,i] for i in range(4)),
                            'absent_answer_changes':sum(normalized_answer(table[left,i]['prediction'])!=normalized_answer(table[right,i]['prediction']) for i in (4,5)),
                            'absent_both_correct':sum(correct[left,i] and correct[right,i] for i in (4,5))})
    complement_totals = {key:sum(pair[key] for pair in complements) for key in complements[0] if key not in ('left_code','right_code')}
    adapted_text = summarize_reference(text,rows)
    adapted_text['scientific_decision'] = 'review_full_text_interference'
    return {'training_parity':training_parity,'splits':splits,'coordinate_edges':edges,
            'coordinate_totals':{'unique_edges':32,**totals},'adapted_full_text':adapted_text,
            'complement_pairs':complements,'complement_totals':{'unique_pairs':8,**complement_totals},
            'generation_records':252,'derived_controls_are_independent_samples':False,
            'control_exact_token_agreement_out_of_six':sum(baseline['zero',i]['generated_ids']==baseline['no_memory',i]['generated_ids'] for i in range(6)),
            'reliability_target':.95,'continuity_control_gap_target':.4,'hard_scientific_cutoff':False,
            'scientific_decision':'review_generalization_controls_and_errors',
            'scope':'unseen_combinations_for_fixed_queries_and_room_pairs'}
