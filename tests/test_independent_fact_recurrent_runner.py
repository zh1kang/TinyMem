"""Behavioral checks for the matched study's evaluation boundary."""
import copy

import pytest
import torch
from independent_fact_fixtures import recurrent_manifest

from scripts.fit_independent_fact_recurrent_training import (
    collect_states,
    state_catalog,
    validate_manifest,
)
from tinymem.memory.recurrent_slots import LatentSlotState


def test_manifest_rejects_changed_truth_and_training_overlap():
    manifest = recurrent_manifest()
    validate_manifest(manifest)
    changed = copy.deepcopy(manifest)
    changed['evaluation_streams'][0]['events'][0]['after_code'] ^= 2
    with pytest.raises(ValueError,match='truth'):
        validate_manifest(changed)
    changed = copy.deepcopy(manifest)
    changed['training_streams'][0]['events'][0]['text'] = 'bad event'
    with pytest.raises(ValueError,match='truth'):
        validate_manifest(changed)


def test_catalog_has_all_initial_single_and_sequence_states():
    manifest = recurrent_manifest()
    catalog = state_catalog(manifest)
    assert len(catalog) == 656
    assert sum(r['kind']=='initial' for r in catalog.values()) == 16
    assert sum(r['kind']=='one' for r in catalog.values()) == 128
    assert sum(r['kind']=='stream' for r in catalog.values()) == 512


class AveragingWriter:
    def empty(self, batch):
        return LatentSlotState(torch.zeros(batch,2,8),torch.zeros(batch,2,dtype=torch.bool))

    def __call__(self, state, hidden, valid):
        value = (state.values + hidden[:,0,:].reshape(-1,2,8))/2
        return LatentSlotState(value,torch.ones_like(state.valid))


def test_evaluation_carries_raw_previous_outputs_for_every_step():
    manifest = recurrent_manifest()
    histories = {code:torch.full((1,16),code/16) for code in range(16)}
    features = {(fact,bit):torch.full((1,16),(2*fact+bit)/8) for fact in range(4) for bit in range(2)}
    states = collect_states(AveragingWriter(),histories,features,manifest)
    for stream in manifest['evaluation_streams']:
        expected = stream['initial_code']/32
        for event in stream['events']:
            expected = (expected + (2*event['target_fact']+event['new_bit'])/8)/2
            actual = states[f'{stream["id"]}:step-{event["step"]:02d}']
            assert torch.equal(actual.values,torch.full((1,2,8),expected))
    assert all(torch.equal(v,torch.full((1,16),code/16)) for code,v in histories.items())


def test_synthetic_evaluation_keeps_two_mixed_programs():
    streams = recurrent_manifest()["evaluation_streams"]
    first, second = streams[:2]
    first_program = [(event["target_fact"], event["action"]) for event in first["events"][:8]]
    second_program = [(event["target_fact"], event["action"]) for event in second["events"][:8]]
    assert second_program == list(reversed(first_program))
    assert {action for _, action in first_program} == {"C", "R"}
    assert all(event["action"] == "R" for stream in streams for event in stream["events"][8:])
