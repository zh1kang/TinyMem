"""Behavioral checks for the full run's replay and evaluation boundaries."""
from copy import deepcopy

import pytest
import torch
from independent_fact_fixtures import recurrent_manifest

from scripts.fit_independent_fact_answer_training import check_profile_prefix
from scripts.fit_independent_fact_recurrent_training import collect_states
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.independent_fact_answer_protocol import (
    build_answer_manifest,
    state_catalog,
)


def test_profile_replay_checks_values_but_not_timing_or_allocation():
    expected={'answer_ce':2.5,'coordinate_mse':None,'gradient_norm':0.01,
              'answer_sequences':480,'step':1,'warmup':True,'seconds':40.0,'cuda_peak_allocated_bytes':1000}
    actual={k:v for k,v in expected.items() if k not in ('step','warmup','seconds','cuda_peak_allocated_bytes')}
    check_profile_prefix(actual,expected)
    changed=deepcopy(actual);changed['answer_ce']+=1e-8
    with pytest.raises(ValueError,match='exactly replay'):
        check_profile_prefix(changed,expected)
    changed=deepcopy(actual);changed.pop('gradient_norm')
    with pytest.raises(ValueError,match='exactly replay'):
        check_profile_prefix(changed,expected)


class AveragingWriter:
    def empty(self,batch):
        return LatentSlotState(torch.zeros(batch,2,8),torch.zeros(batch,2,dtype=torch.bool))

    def __call__(self,state,hidden,valid):
        values=(state.values+hidden[:,0,:].reshape(-1,2,8))/2
        return LatentSlotState(values,torch.ones_like(state.valid))


def test_full_catalog_uses_raw_recurrence_and_opposite_truth_swap():
    previous=recurrent_manifest()
    manifest=build_answer_manifest(previous)
    combined={'evaluation_streams':manifest['old_continuity_streams']+manifest['evaluation_streams']}
    histories={c:torch.full((1,16),c/16) for c in range(16)}
    features={(f,b):torch.full((1,16),(2*f+b)/8) for f in range(4) for b in range(2)}
    states=collect_states(AveragingWriter(),histories,features,combined)
    catalog=state_catalog(manifest)
    assert set(states)==set(catalog) and len(states)==1168
    for stream in combined['evaluation_streams']:
        value=stream['initial_code']/32
        for event in stream['events']:
            value=(value+(2*event['target_fact']+event['new_bit'])/8)/2
            assert torch.equal(states[f'{stream["id"]}:step-{event["step"]:02d}'].values,torch.full((1,2,8),value))
    for stream in manifest['evaluation_streams']:
        donor=f'fresh-{stream["initial_code"] ^ 15:02d}-{stream["id"].rsplit("-",1)[1]}:step-16'
        recipient=stream['id']+':step-16'
        assert catalog[recipient]['code'] ^ catalog[donor]['code']==15
        assert donor in states
