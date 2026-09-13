"""Read explicit state tables without reconstructing features from an adapted reader."""
from collections import defaultdict
from collections.abc import Mapping, Sequence

import torch

from tinymem.evaluation.longmemeval import normalized_answer
from tinymem.memory.readout_interface import ReadoutBridge, check_readout_state
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.pretrained import PretrainedReader
from tinymem.research.readout_controls import controlled_state, state_donors
from tinymem.research.readout_evaluation import _score_answer
from tinymem.research.readout_read import read_state_answer
from tinymem.research.readout_runner import EncodedBefore


@torch.inference_mode()
def evaluate_states(
    reader: PretrainedReader, bridge: ReadoutBridge, states: Mapping[str, LatentSlotState],
    rows: Sequence[EncodedBefore], *, max_new_tokens: int = 8,
) -> list[dict]:
    """History tokens and features are not inputs to any compressed read."""
    ids = [row.history_id for row in rows]
    donors = state_donors(ids)
    if set(states) != set(ids):
        raise ValueError('state coverage differs from evaluation histories')
    for state in states.values():
        check_readout_state(state)
    results = []
    device = reader.model.device
    for row in rows:
        before = torch.tensor(row.before_ids, device=device)
        for condition in ('normal', 'zero', 'no_memory', 'shuffled'):
            donor = donors[row.history_id] if condition == 'shuffled' else row.history_id
            state = controlled_state(states[donor], 'normal' if condition == 'shuffled' else condition)
            memory = bridge(state)
            for query in row.queries:
                after = torch.tensor(query.after_ids, device=device)
                generated = read_state_answer(reader, bridge, state, before, after, max_new_tokens=max_new_tokens)
                donor_queries = next(source.queries for source in rows if source.history_id == donor)
                donor_answer = next((q.answer for q in donor_queries if q.after_ids == query.after_ids), 'unknown')
                results.append({**_score_answer(reader, before, memory, after, query, generated),
                                'history_id': row.history_id, 'condition': condition,
                                'donor_history_id': donor, 'persistent_bytes': 66,
                                'donor_answer': donor_answer,
                                'donor_answer_matches_recipient': donor_answer == query.answer})
    return results


def summarize_readiness(records: Sequence[dict]) -> dict:
    """Fixed eight-history fit gate, with strict coverage and independently scored text."""
    conditions = ('normal', 'shuffled', 'zero', 'no_memory', 'full_text')
    groups, seen, labels, case_histories = defaultdict(list), set(), {}, {}
    for row in records:
        condition, case = row['condition'], row['case_id']
        if condition not in conditions or (condition, case) in seen:
            raise ValueError('unexpected or duplicate coverage')
        seen.add((condition, case))
        category = row['category']
        if category not in ('update_known', 'update_missing'):
            raise ValueError('unexpected category')
        label = (row['history_id'], category, row['answer'])
        if case in labels and labels[case] != label:
            raise ValueError('inconsistent query labels')
        labels[case] = label
        case_histories[case] = row['history_id']
        groups[condition, category].append((case, normalized_answer(row['prediction']) == normalized_answer(row['answer'])))
    histories = set(case_histories.values())
    if len(histories) != 8 or len(labels) != 80:
        raise ValueError('expected eight-history, eighty-query coverage')
    expected_cases = set(labels)
    for condition in conditions:
        if {case for name, case in seen if name == condition} != expected_cases:
            raise ValueError('incomplete condition coverage')
    for history in histories:
        if sum(h == history and c == 'update_known' for h, c, _ in labels.values()) != 8 or sum(h == history and c == 'update_missing' for h, c, _ in labels.values()) != 2:
            raise ValueError('expected eight known and two absent queries per history')
    scores = {}
    for condition in conditions:
        correct_by_case = dict(groups[condition, 'update_known'] + groups[condition, 'update_missing'])
        scores[condition] = {
            'known_accuracy': sum(ok for _, ok in groups[condition, 'update_known']) / 64,
            'missing_accuracy': sum(ok for _, ok in groups[condition, 'update_missing']) / 16,
            'all_ten_correct_histories': sum(all(ok for case, ok in correct_by_case.items() if case_histories[case] == history) for history in histories),
            'per_history_known_accuracy': {history: sum(ok for case, ok in groups[condition, 'update_known'] if case_histories[case] == history) / 8 for history in sorted(histories)},
        }
    gaps = {condition: scores['normal']['known_accuracy'] - scores[condition]['known_accuracy']
            for condition in ('shuffled', 'zero', 'no_memory')}
    full_text = scores['full_text']['known_accuracy'] >= 0.95 and scores['full_text']['missing_accuracy'] >= 0.95
    passed = (full_text and scores['normal']['known_accuracy'] >= 0.95
              and scores['normal']['missing_accuracy'] >= 0.95 and min(gaps.values()) >= 0.5)
    return {'scores': scores, 'known_accuracy_gaps': gaps, 'full_text_passed': full_text,
            'readiness_passed': passed, 'evidence_kind': 'consumed_training_history_fit_only'}
