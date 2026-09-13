"""Paired conflicting swaps test which history supplies each answer."""
from collections.abc import Mapping, Sequence
import hashlib

import torch

from tinymem.evaluation.longmemeval import normalized_answer
from tinymem.memory.readout_interface import ReadoutBridge
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.adapted_readout_evaluation import evaluate_states, summarize_readiness
from tinymem.research.paired_readout_data import audit_pairs
from tinymem.research.pretrained import PretrainedReader
from tinymem.research.readout_runner import EncodedBefore


def verify_state_payload(expected: Mapping[str, torch.Tensor], restored: Mapping[str, torch.Tensor]) -> dict[str, str]:
    """Compare actual serialization values and validity, then anchor tensor bytes."""
    if set(expected) != set(restored):
        raise ValueError('serialized state tensor coverage differs')
    hashes = {}
    for name, source in expected.items():
        saved = restored[name]
        if source.dtype != saved.dtype or source.shape != saved.shape or not torch.equal(source.cpu(), saved.cpu()):
            raise ValueError('serialized state tensor differs: ' + name)
        hashes[name] = hashlib.sha256(source.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    return hashes


def evaluate_pairs(reader: PretrainedReader, bridge: ReadoutBridge,
                   states: Mapping[str, LatentSlotState], rows: Sequence[EncodedBefore]) -> list[dict]:
    audit_pairs(rows)
    if set(states) != {row.history_id for row in rows}:
        raise ValueError('paired state coverage differs')
    records = []
    for index in range(0, 8, 2):
        pair = rows[index:index + 2]
        records.extend(evaluate_states(reader, bridge, {row.history_id: states[row.history_id] for row in pair}, pair))
    return records


def summarize_pairs(records: Sequence[dict], rows: Sequence[EncodedBefore]) -> dict:
    audit = audit_pairs(rows)
    summary = summarize_readiness(records)
    queries = {q.case_id: (row, q) for row in rows for q in row.queries}
    if {record['case_id'] for record in records} != set(queries):
        raise ValueError('prediction coverage differs from paired queries')
    by_id = {row.history_id: row for row in rows}
    donor_correct = []
    for record in records:
        row, query = queries[record['case_id']]
        if (record['history_id'], record['category'], record['answer']) != (row.history_id, query.category, query.answer):
            raise ValueError('prediction labels differ from paired queries')
        if record['condition'] == 'full_text':
            continue
        donor_id = audit['donors'][row.history_id] if record['condition'] == 'shuffled' else row.history_id
        donor_query = next(q for q in by_id[donor_id].queries if q.after_ids == query.after_ids)
        if (record['donor_history_id'] != donor_id or record['donor_answer'] != donor_query.answer
                or record['donor_answer_matches_recipient'] != (donor_query.answer == query.answer)):
            raise ValueError('paired donor identity or answer differs')
        if record['condition'] == 'shuffled' and query.category == 'update_known':
            donor_correct.append(normalized_answer(record['prediction']) == normalized_answer(donor_query.answer))
    donor_accuracy = sum(donor_correct) / 64
    scores, gaps = summary['scores'], summary['known_accuracy_gaps']
    passed = (summary['full_text_passed'] and scores['normal']['known_accuracy'] >= .95
              and scores['normal']['missing_accuracy'] >= .95 and donor_accuracy >= .95
              and min(gaps[c] for c in ('zero', 'no_memory')) >= .4)
    summary.pop('readiness_passed')
    return {**summary, 'binding_passed': passed, 'swapped_donor_known_accuracy': donor_accuracy,
            'evidence_kind': 'paired_training_history_fit_only'}
