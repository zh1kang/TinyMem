"""Paired swaps require donor answers, not accuracy against the wrong history."""
from copy import deepcopy
import hashlib
import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from test_memory_updates import episode
from test_readout_runner import tiny_reader
from tinymem.research.paired_readout_data import build_pairs, encode_pairs
from tinymem.research.paired_readout_evaluation import summarize_pairs, verify_state_payload
from scripts.fit_paired_readout import require_verified_qualification


def records_for(rows):
    result = []
    for i, row in enumerate(rows):
        donor = rows[i ^ 1]
        baseline = rows[i // 2 * 2]
        for condition in ('normal', 'shuffled', 'zero', 'no_memory', 'full_text'):
            for j, q in enumerate(row.queries):
                supported = donor.queries[j] if condition == 'shuffled' else q
                prediction = baseline.queries[j].answer if condition in ('zero', 'no_memory') else supported.answer
                result.append(dict(history_id=row.history_id, case_id=q.case_id, category=q.category,
                                   answer=q.answer, condition=condition, prediction=prediction,
                                   donor_history_id=donor.history_id if condition == 'shuffled' else row.history_id,
                                   donor_answer=supported.answer,
                                   donor_answer_matches_recipient=supported.answer == q.answer))
    return result


def test_correct_donor_answers_pass_and_question_only_lookup_fails(tiny_reader):
    rows = encode_pairs(tiny_reader, build_pairs([episode(i) for i in range(4)]))
    records = records_for(rows)
    summary = summarize_pairs(records, rows)
    assert summary['binding_passed'] is True
    assert summary['scores']['shuffled']['known_accuracy'] == 0
    assert summary['swapped_donor_known_accuracy'] == 1
    assert summary['scores']['no_memory']['known_accuracy'] == .5
    lookup = {(r['case_id'], r['condition']): r['prediction'] for r in records}
    ignored = deepcopy(records)
    for row in ignored:
        if row['condition'] != 'full_text':
            row['prediction'] = lookup[row['case_id'], 'no_memory']
    assert summarize_pairs(ignored, rows)['binding_passed'] is False
    bad = deepcopy(records)
    next(r for r in bad if r['condition'] == 'shuffled')['donor_answer'] = 'unknown'
    with pytest.raises(ValueError, match='donor'):
        summarize_pairs(bad, rows)
    with pytest.raises(ValueError, match='coverage'):
        summarize_pairs(records[:-1], rows)


def test_fit_requires_independent_positive_verification(tmp_path):
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    protocol = {'declaration_sha256': 'qualification-declaration', 'reader': {'model': 'fixture'},
                'data_protocol_sha256': 'source-data'}
    report = {'status': 'complete', 'qualified': True, 'known_correct': 62, 'known_total': 64,
              'missing_correct': 16, 'missing_total': 16, 'training_steps': 0, 'reader_unchanged': True}
    for name, value in {'protocol.json': protocol, 'report.json': report, 'paired_cases.json': [],
                        'encodings.json': [], 'input_audit.json': {}, 'predictions.jsonl': []}.items():
        (tmp_path / name).write_text(json.dumps(value))
    (tmp_path / 'complete.json').write_text(json.dumps({'kind': 'paired_readout_qualification_complete_v1',
               'files': {p.name: sha(p) for p in tmp_path.iterdir()}}))
    verification = {'verified': True, 'qualified': True, 'report_sha256': sha(tmp_path / 'report.json'),
                    'declaration_sha256': 'qualification-declaration'}
    path = tmp_path / 'independent.json'
    path.write_text(json.dumps(verification))
    declaration = {'qualification_report_sha256': verification['report_sha256'],
                   'qualification_verification_sha256': sha(path),
                   'qualification_complete_sha256': sha(tmp_path / 'complete.json'),
                   'qualification_declaration_sha256': 'qualification-declaration',
                   'reader': protocol['reader'], 'data_protocol_sha256': 'source-data'}
    assert require_verified_qualification(tmp_path, path, declaration) == report
    verification['verified'] = False
    path.write_text(json.dumps(verification))
    declaration['qualification_verification_sha256'] = sha(path)
    with pytest.raises(ValueError, match='independent positive'):
        require_verified_qualification(tmp_path, path, declaration)


def test_serialized_state_rejects_changed_values_validity_and_dtype(tmp_path):
    expected = {'history.values': torch.randn(1, 2, 8), 'history.valid': torch.ones(1, 2, dtype=torch.bool)}
    path = tmp_path / 'states.safetensors'
    save_file(expected, str(path))
    restored = load_file(str(path))
    assert set(verify_state_payload(expected, restored)) == set(expected)
    for key in expected:
        changed = {name: tensor.clone() for name, tensor in restored.items()}
        changed[key].flatten()[0] = 0
        with pytest.raises(ValueError, match='serialized state tensor differs'):
            verify_state_payload(expected, changed)
    restored['history.values'] = restored['history.values'].double()
    with pytest.raises(ValueError, match='serialized state tensor differs'):
        verify_state_payload(expected, restored)
