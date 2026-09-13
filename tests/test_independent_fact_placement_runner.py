from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from test_readout_runner import tiny_reader
from scripts.fit_independent_fact_placement import require_control_reproduction, score_predictions
from tinymem.memory.readout_interface import ReadoutBridge
from tinymem.research.independent_fact_placement import placement_state


def _write(path, value):
    path.write_text(json.dumps(value))


def _reproduction_fixture(tmp_path):
    baseline = tmp_path/'baseline'
    (baseline/'even').mkdir(parents=True)
    current = tmp_path/'current'
    current.mkdir()
    summary = {'final_checkpoint_sha256':{'bridge':'b','adapter':'a'}, 'reader_initial_sha256':'r0', 'reader_final_sha256':'r1'}
    metrics = [{'step':1,'answer_ce':1.5,'seconds':2.0,'cuda_allocated_bytes':123}]
    predictions = [{'condition':'normal','code':0,'query_index':0,'prediction':'company','generated_ids':[11,0]}]
    score = {'condition':'normal','code':0,'query_index':0,'candidate':'company','role':'saved_greedy_completion','appended_eos':False,
             'answer_ids':[11,0],'token_log_probs':[-1.,-.1],'sequence_log_probability':-1.1,'mean_answer_ce':.55}
    _write(baseline/'even/summary.json',summary)
    (baseline/'even/metrics.jsonl').write_text(json.dumps(metrics[0])+'\n')
    (baseline/'even/predictions.jsonl').write_text(json.dumps(predictions[0])+'\n')
    old_score = {k:v for k,v in score.items() if k!='appended_eos'}
    old_score.update(arm='even',role='saved_greedy')
    (tmp_path/'old_scores.jsonl').write_text(json.dumps(old_score)+'\n')
    (current/'scores.jsonl').write_text(json.dumps(score)+'\n')
    return current,metrics,predictions,summary,baseline,tmp_path/'old_scores.jsonl'


def test_reproduction_ignores_cost_but_preserves_behavior(tmp_path):
    directory,metrics,predictions,summary,baseline,scores = _reproduction_fixture(tmp_path)
    metrics[0].update(seconds=99.0,cuda_allocated_bytes=456)
    require_control_reproduction(directory,'even',metrics,predictions,summary,baseline,scores)


@pytest.mark.parametrize('changed', ['metric','prediction','checkpoint','score'])
def test_reproduction_rejects_changed_behavior(tmp_path, changed):
    directory,metrics,predictions,summary,baseline,scores = _reproduction_fixture(tmp_path)
    if changed=='metric':
        metrics[0]['answer_ce'] += .01
    elif changed=='prediction':
        predictions[0]['generated_ids'] = [12,0]
    elif changed=='checkpoint':
        summary['reader_final_sha256'] = 'changed'
    else:
        value=json.loads((directory/'scores.jsonl').read_text());value['token_log_probs'][0] -= .1
        (directory/'scores.jsonl').write_text(json.dumps(value)+'\n')
    with pytest.raises(ValueError,match='reproduction'):
        require_control_reproduction(directory,'even',metrics,predictions,summary,baseline,scores)


def test_scoring_covers_all_native_prompts_and_retains_auxiliary_answers(tiny_reader,tmp_path):
    queries = [SimpleNamespace(after_ids=(7,8)) for _ in range(6)]
    rows = [SimpleNamespace(before_ids=(1,2),queries=queries) for _ in range(16)]
    states = {c:placement_state(c,torch.device('cpu'),'separate_fact0') for c in range(16)}
    keys = [('normal',c,i) for c in range(16) for i in range(6)] + [(cond,None,i) for cond in ('zero','no_memory') for i in range(6)]
    predictions = [{'condition':cond,'code':c,'query_index':i,'generated_ids':[200,0],'prediction':'extra',
                    'memory_positions':0 if cond=='no_memory' else 2,'native_envelope_tokens':4,
                    'input_positions':4 if cond=='no_memory' else 6} for cond,c,i in keys]
    bridge = ReadoutBridge(16,'affine').requires_grad_(False).eval()
    result=score_predictions(tiny_reader,bridge,rows,states,predictions,tmp_path/'scores.jsonl')
    output=[json.loads(line) for line in (tmp_path/'scores.jsonl').read_text().splitlines()]
    assert result['fixed_sequences']==756
    assert result['auxiliary_sequences']==108
    assert len(output)==864
    assert all(r['answer_ids']==[200,0] and not r['appended_eos'] for r in output if r['role']=='saved_greedy_completion')
    malformed=deepcopy(predictions);malformed[0]['input_positions'] += 1
    with pytest.raises(ValueError,match='dimensions'):
        score_predictions(tiny_reader,bridge,rows,states,malformed,tmp_path/'invalid.jsonl')
