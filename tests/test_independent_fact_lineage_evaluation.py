"""Behavioral checks for the fixed frozen lineage evaluation."""

import json

import pytest
import torch

from test_independent_fact_answer_training import _features
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.independent_fact_data import ENTITIES, ROOM_PAIRS
from tinymem.research.independent_fact_updates import new_update_writer


def _fake_reader(state, query):
    fact = query % 4
    slot, offset = (0, 0) if fact == 0 else (1, fact)
    bit = 0 if state is None else int(float(state.values[0, slot, offset]) > 0)
    return {
        'prediction': ROOM_PAIRS[query % 4][bit],
        'generated_ids': [query],
        'memory_positions': 2,
        'native_envelope_tokens': 1,
        'input_positions': 3,
    }


def _references():
    return [
        {**{**_fake_reader(None, query),
            'prediction': ROOM_PAIRS[query % 4][(code >> (query % 4)) & 1]},
         'code': code, 'query_index': query}
        for code in range(16) for query in range(6)
    ]


def _training_manifest():
    orders = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))
    actions = (('C', 'C', 'C', 'C'), ('R', 'R', 'R', 'R'),
               ('C', 'R', 'C', 'R'), ('R', 'C', 'R', 'C'))
    streams = []
    for code in range(16):
        for order, facts in enumerate(orders):
            for action, pattern in enumerate(actions):
                current = code
                events = []
                for step, (fact, kind) in enumerate(zip(facts, pattern, strict=True), 1):
                    before = current
                    if kind == 'C':
                        current ^= 1 << fact
                    bit = (current >> fact) & 1
                    events.append({
                        'step': step, 'before_code': before, 'after_code': current,
                        'target_fact': fact, 'new_bit': bit, 'action': kind,
                        'text': f'{ENTITIES[fact]} moved to the {ROOM_PAIRS[fact][bit]}.',
                    })
                streams.append({'id': f'train-{code:02d}-{order}-{action}',
                                'initial_code': code, 'events': events})
    return {'training_streams': streams}


class _CollapsedWriter(torch.nn.Module):
    slots = 2
    memory_width = 8

    def empty(self, batch_size):
        return LatentSlotState(torch.zeros(batch_size, 2, 8),
                               torch.zeros(batch_size, 2, dtype=torch.bool))

    def forward(self, state, hidden, valid):
        del hidden, valid
        return LatentSlotState(torch.zeros_like(state.values), torch.ones_like(state.valid))


def _collapsed_state_reader(state, query):
    del state
    return {
        'prediction': ROOM_PAIRS[query % 4][0], 'generated_ids': [query],
        'memory_positions': 2, 'native_envelope_tokens': 1, 'input_positions': 3,
    }


def test_lineage_evaluation_writes_full_read_and_probe_artifacts(tmp_path):
    from scripts.evaluate_independent_fact_lineage import ARMS, evaluate_lineage

    from tinymem.research.independent_fact_repeat_confirmation import build_manifest

    training = _training_manifest()
    refresh = build_manifest({'evaluation_streams': [], 'old_continuity_streams': []})
    histories, features = _features(16)
    initializer = new_update_writer(16, 321).requires_grad_(False).eval()
    updaters = {arm: new_update_writer(16, seed).requires_grad_(False).eval()
                for arm, seed in zip(ARMS, (654, 987), strict=True)}

    summary = evaluate_lineage(
        output=tmp_path / 'lineage', initializer=initializer, updaters=updaters,
        histories=histories, features=features, training_manifest=training,
        refresh_manifest=refresh, read=_fake_reader, references=_references())

    assert summary['counts'] == {
        'normal': 37056, 'no_write': 768, 'swap': 768,
        'constant': 12, 'reference': 96,
    }
    assert summary['states'] == {
        'fresh_per_model': 3088, 'fresh_total': 6176,
        'training_per_model': 1024, 'training_total': 2048,
    }
    output = tmp_path / 'lineage'
    assert (output / 'fresh_states.safetensors').is_file()
    assert (output / 'training_states.safetensors').is_file()
    assert (output / 'probe_scores.npz').is_file()
    assert len((output / 'predictions.jsonl').read_text().splitlines()) == 37056
    assert len((output / 'controls.jsonl').read_text().splitlines()) == 1548
    assert len((output / 'reference_replays.jsonl').read_text().splitlines()) == 96
    probe = json.loads((output / 'probe_summary.json').read_text())
    assert set(probe['models']) == set(ARMS)
    assert probe['models']['uniform']['fit_eval_overlap_including_branches'] >= 0
    assert len(probe['models']['uniform']['controls']) == 99


def test_lineage_evaluation_rejects_existing_output(tmp_path):
    from scripts.evaluate_independent_fact_lineage import evaluate_lineage

    output = tmp_path / 'already-there'
    output.mkdir()
    initializer = new_update_writer(16, 1).requires_grad_(False).eval()
    with pytest.raises(FileExistsError):
        evaluate_lineage(output=output, initializer=initializer,
                         updaters={'uniform': initializer, 'correction_weighted': initializer},
                         histories={}, features={}, training_manifest={}, refresh_manifest={},
                         read=_fake_reader, references=[])


def test_lineage_evaluation_reports_finite_collapsed_states(tmp_path):
    from scripts.evaluate_independent_fact_lineage import ARMS, evaluate_lineage
    from tinymem.research.independent_fact_repeat_confirmation import build_manifest

    training = _training_manifest()
    refresh = build_manifest({'evaluation_streams': [], 'old_continuity_streams': []})
    histories, features = _features(16)
    initializer = _CollapsedWriter().eval()
    updaters = {arm: _CollapsedWriter().eval() for arm in ARMS}
    references = [{**_collapsed_state_reader(None, query), 'code': code, 'query_index': query}
                  for code in range(16) for query in range(6)]

    summary = evaluate_lineage(
        output=tmp_path / 'collapsed', initializer=initializer, updaters=updaters,
        histories=histories, features=features, training_manifest=training,
        refresh_manifest=refresh, read=_collapsed_state_reader, references=references)

    assert summary['counts']['normal'] == 37056
    probe = json.loads((tmp_path / 'collapsed' / 'probe_summary.json').read_text())
    assert probe['models']['uniform']['crossfold_tensor_collisions'] > 0
    assert probe['models']['uniform']['fit_eval_overlap_including_branches'] > 0
    assert probe['models']['uniform']['primary_overlap_count'] == 64 + 64*5*8
    assert probe['models']['uniform']['symbolic_control_correct'] == 12352
    assert probe['models']['uniform']['overlap_by_prefix_step'] == {str(step):64 for step in range(1,9)}


def test_lineage_evaluation_rejects_mutating_read_replay(tmp_path):
    from scripts.evaluate_independent_fact_lineage import ARMS, evaluate_lineage
    from tinymem.research.independent_fact_repeat_confirmation import build_manifest

    def mutating_reader(state, query):
        state.values.add_(1)
        return _fake_reader(state, query)

    training = _training_manifest()
    refresh = build_manifest({'evaluation_streams': [], 'old_continuity_streams': []})
    histories, features = _features(16)
    initializer = new_update_writer(16, 321).requires_grad_(False).eval()
    updaters = {arm: new_update_writer(16, seed).requires_grad_(False).eval()
                for arm, seed in zip(ARMS, (654, 987), strict=True)}
    with pytest.raises(ValueError, match='replay|mutated'):
        evaluate_lineage(
            output=tmp_path / 'mutating', initializer=initializer, updaters=updaters,
            histories=histories, features=features, training_manifest=training,
            refresh_manifest=refresh, read=mutating_reader, references=_references())
