"""Truth and branch-ownership checks for frozen truthful-refresh evaluation."""
from collections import Counter
from copy import deepcopy

import pytest
import torch

from test_independent_fact_answer_training import _features
from tinymem.research.independent_fact_data import ENTITIES, ROOM_PAIRS
from tinymem.research.independent_fact_updates import new_update_writer


def test_manifest_is_fresh_balanced_and_truth_preserving():
    from tinymem.research.independent_fact_repeat_confirmation import build_manifest, state_catalog

    prior = {'evaluation_streams': [], 'old_continuity_streams': []}
    manifest = build_manifest(prior)
    assert manifest == build_manifest(prior)
    assert len(manifest['streams']) == 64
    assert Counter(s['initial_code'] for s in manifest['streams']) == {c: 4 for c in range(16)}
    programs = set()
    for stream in manifest['streams']:
        signature = tuple((e['target_fact'], e['action']) for e in stream['prefix'])
        assert Counter(signature) == {(f, a): 1 for f in range(4) for a in 'CR'}
        programs.add(signature)
        truth = {ENTITIES[f]: ROOM_PAIRS[f][(stream['initial_code'] >> f) & 1] for f in range(4)}
        for event in stream['prefix']:
            entity, room = event['text'].removesuffix('.').split(' moved to the ')
            truth[entity] = room
            code = sum(ROOM_PAIRS[f].index(truth[ENTITIES[f]]) << f for f in range(4))
            assert code == event['after_code']
        assert code == stream['initial_code'] ^ 15
        assert set(stream['tails']) == {'balanced', 'single0', 'single1', 'single2', 'single3'}
        for condition, events in stream['tails'].items():
            assert len(events) == 8
            assert [e['step'] for e in events] == list(range(1, 9))
            for event in events:
                entity, room = event['text'].removesuffix('.').split(' moved to the ')
                assert truth[entity] == room
                assert event['before_code'] == event['after_code'] == code
            counts = Counter(e['target_fact'] for e in events)
            assert counts == ({f: 2 for f in range(4)} if condition == 'balanced' else {int(condition[-1]): 8})
    assert len(programs) == 64
    assert len(state_catalog(manifest)) == 3088
    excluded = {'evaluation_streams': [{'events': manifest['streams'][0]['prefix']}], 'old_continuity_streams': []}
    changed = build_manifest(excluded)
    old_signature = tuple((e['target_fact'], e['action']) for e in manifest['streams'][0]['prefix'])
    assert all(tuple((e['target_fact'], e['action']) for e in s['prefix']) != old_signature for s in changed['streams'])


def test_all_tails_branch_from_same_state_and_match_direct_replay():
    from tinymem.research.independent_fact_repeat_confirmation import build_manifest, collect_states, state_catalog

    manifest = build_manifest({'evaluation_streams': [], 'old_continuity_streams': []})
    histories, features = _features(16)
    initializer = new_update_writer(16, 321).requires_grad_(False).eval()
    updater = new_update_writer(16, 654).requires_grad_(False).eval()
    snapshots = [deepcopy(m.state_dict()) for m in (initializer, updater)]
    states = collect_states(initializer, updater, histories, features, manifest)
    assert set(states) == set(state_catalog(manifest))
    assert len({s.values.data_ptr() for s in states.values()}) == len(states)
    stream = manifest['streams'][0]
    initial = states[f"initial:{stream['initial_code']:02d}"]
    state = initial
    for event in stream['prefix']:
        hidden = features[event['target_fact'], event['new_bit']].unsqueeze(0)
        with torch.no_grad():
            state = updater(state, hidden, torch.ones(hidden.shape[:2], dtype=torch.bool))
    root = state
    for condition, events in stream['tails'].items():
        state = root
        for event in events:
            hidden = features[event['target_fact'], event['new_bit']].unsqueeze(0)
            with torch.no_grad():
                state = updater(state, hidden, torch.ones(hidden.shape[:2], dtype=torch.bool))
            actual = states[f"{stream['id']}/{condition}/{event['step']:02d}"]
            assert torch.equal(actual.values, state.values)
            assert actual.nbytes == 66
    reordered = deepcopy(manifest)
    for s in reordered['streams']:
        s['tails'] = dict(reversed(list(s['tails'].items())))
    other = collect_states(initializer, updater, histories, features, reordered)
    assert all(torch.equal(state.values, other[name].values) for name, state in states.items())
    for module, snapshot in zip((initializer, updater), snapshots, strict=True):
        assert all(torch.equal(v, snapshot[k]) for k, v in module.state_dict().items())
        assert all(p.grad is None for p in module.parameters())


def test_collection_rejects_trainable_checkpoint():
    from tinymem.research.independent_fact_repeat_confirmation import build_manifest, collect_states

    manifest = build_manifest({'evaluation_streams': [], 'old_continuity_streams': []})
    histories, features = _features(16)
    with pytest.raises(ValueError, match='frozen'):
        collect_states(new_update_writer(16, 1), new_update_writer(16, 2), histories, features, manifest)
