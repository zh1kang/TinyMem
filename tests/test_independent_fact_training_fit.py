"""Training-path coverage and recurrent state ownership checks."""
import random

import pytest
import torch
from independent_fact_fixtures import answer_manifest

from scripts.evaluate_independent_fact_training_fit import (
    GEOMETRIES,
    catalog_for,
    collect_training_states,
)
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.independent_fact_updates import new_update_writer


@pytest.fixture
def inputs():
    manifest = answer_manifest()
    generator = torch.Generator().manual_seed(271)
    histories = {code: torch.randn(2+code%3, 4, generator=generator) for code in range(16)}
    features = {(fact, bit): torch.randn(1+fact+bit, 4, generator=generator) for fact in range(4) for bit in range(2)}
    order = list(range(256)); random.Random(1337).shuffle(order)
    return manifest, histories, features, order


def test_all_training_paths_use_raw_prior_states_and_preserve_inputs(inputs):
    manifest, histories, features, order = inputs
    writer = new_update_writer(4, 77).requires_grad_(False).eval()
    saved = {k: v.clone() for k, v in writer.state_dict().items()}
    source = {k: v.clone() for k, v in features.items()}
    states = collect_training_states(writer, histories, features, manifest, order)
    assert all(len(states[g]) == 1040 for g in GEOMETRIES)
    assert not any(name.startswith(('fresh-', 'eval-')) for name in states[GEOMETRIES[0]])
    with torch.no_grad():
        for stream in manifest['training_streams']:
            hidden = histories[stream['initial_code']].unsqueeze(0)
            state = writer(writer.empty(1), hidden, torch.ones(hidden.shape[:2], dtype=torch.bool))
            for event in stream['events']:
                hidden = features[event['target_fact'], event['new_bit']].unsqueeze(0)
                state = writer(state, hidden, torch.ones(hidden.shape[:2], dtype=torch.bool))
                name = f'{stream["id"]}:step-{event["step"]:02d}'
                assert torch.equal(states['serial_batch1'][name].values, state.values)
    assert all(torch.equal(v, saved[k]) for k, v in writer.state_dict().items())
    assert all(torch.equal(v, source[k]) for k, v in features.items())
    assert len({s.values.data_ptr() for g in GEOMETRIES for s in states[g].values()}) == 2080


def test_batch_geometry_retains_parent_groups_and_masked_padding(inputs):
    manifest, histories, features, order = inputs
    writer = new_update_writer(4, 31).requires_grad_(False).eval()
    states = collect_training_states(writer, histories, features, manifest, order)
    # Construct the original tensor geometry independently of pack_recurrent_batch.
    h = torch.zeros(16, max(len(v) for v in histories.values()), 4)
    hv = torch.zeros(h.shape[:2], dtype=torch.bool)
    for code, values in histories.items():
        h[code, :len(values)] = values; hv[code, :len(values)] = True
    with torch.no_grad():
        initial = writer(writer.empty(16), h, hv)
        for offset in range(0, 256, 16):
            streams = [manifest['training_streams'][i] for i in order[offset:offset+16]]
            indices = torch.tensor([s['initial_code'] for s in streams])
            state = LatentSlotState(initial.values[indices], initial.valid[indices])
            for step in range(4):
                hidden = torch.zeros(16, max(len(v) for v in features.values()), 4)
                valid = torch.zeros(hidden.shape[:2], dtype=torch.bool)
                for row, stream in enumerate(streams):
                    event = stream['events'][step]
                    values = features[event['target_fact'], event['new_bit']]
                    hidden[row, :len(values)] = values; valid[row, :len(values)] = True
                state = writer(state, hidden, valid)
                for row, stream in enumerate(streams):
                    saved = states['training_batch16'][f'{stream["id"]}:step-{step+1:02d}']
                    assert torch.equal(saved.values, state.values[row:row+1])
                    assert saved.nbytes == 66


def test_invalid_schedule_and_truth_fail_before_evaluation(inputs):
    manifest, histories, features, _order = inputs
    writer = new_update_writer(4, 77)
    with pytest.raises(ValueError, match='permutation'):
        collect_training_states(writer, histories, features, manifest, [0]*256)
    manifest['training_streams'][0]['events'][0]['after_code'] ^= 1
    with pytest.raises(ValueError):
        catalog_for(manifest)
