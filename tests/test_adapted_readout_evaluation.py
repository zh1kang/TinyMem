"""Recall controls must reject a reader that ignores the stored state."""
from copy import deepcopy

import pytest
import torch

from test_readout_runner import tiny_reader
from tinymem.memory.readout_interface import OneShotEncoder, ReadoutBridge
from tinymem.research.adapted_readout_evaluation import evaluate_states, summarize_readiness
from tinymem.research.adapted_readout import frozen_history_features
from tinymem.research.readout_runner import EncodedBefore, ReadoutQuery


def test_gate_rejects_perfect_question_memorization_without_memory_dependence():
    records = []
    for history in range(8):
        for condition in ('normal', 'shuffled', 'zero', 'no_memory', 'full_text'):
            for q in range(10):
                answer = 'kitchen' if q < 8 else 'unknown'
                records.append(dict(history_id=str(history), case_id=f'{history}:{q}', condition=condition,
                                    category='update_known' if q < 8 else 'update_missing', answer=answer, prediction=answer))
    assert summarize_readiness(records)['readiness_passed'] is False
    dependent = deepcopy(records)
    for row in dependent:
        if row['condition'] in ('shuffled', 'zero', 'no_memory') and row['category'] == 'update_known':
            row['prediction'] = 'unknown'
    assert summarize_readiness(dependent)['readiness_passed'] is True
    with pytest.raises(ValueError, match='coverage'):
        summarize_readiness(dependent[:-1])


def test_state_evaluator_never_reads_history_tokens(tiny_reader):
    encoder, bridge = OneShotEncoder(16), ReadoutBridge(16, 'affine')
    rows, states = [], {}
    for i in range(2):
        hidden = frozen_history_features(tiny_reader, (3, 4, 5 + i))
        states[str(i)] = encoder(hidden.unsqueeze(0), torch.ones(1, 3, dtype=torch.bool))
        rows.append(EncodedBefore(str(i), ('a', 'b'), ('c', 'd'), ('e', 'f'), (1, 2),
                                  (99999,), (ReadoutQuery(str(i), 'update_known', 'kitchen', (7, 8), (9, 0)),)))
    # History tokens deliberately cannot be embedded by this model.
    result = evaluate_states(tiny_reader, bridge, states, rows, max_new_tokens=1)
    assert len(result) == 8
    assert {r['condition'] for r in result} == {'normal', 'zero', 'no_memory', 'shuffled'}
    assert all(r['donor_history_id'] != r['history_id'] for r in result if r['condition'] == 'shuffled')
    assert all(r['persistent_bytes'] == 66 for r in result)
