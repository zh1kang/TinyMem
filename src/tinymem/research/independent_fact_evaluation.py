"""Exact-count reference and finite-code readout summaries, without scientific cutoffs."""
import torch

from tinymem.evaluation.longmemeval import normalized_answer
from tinymem.research.prefix_reader import generate_prefix_answer
from tinymem.research.readout_evaluation import _score_answer


@torch.inference_mode()
def full_text_records(reader, rows):
    if any(m.training for m in reader.model.modules()) or any(p.requires_grad or p.grad is not None for p in reader.model.parameters()):
        raise ValueError('reference reader must be frozen and in evaluation mode')
    device = reader.model.device
    memory = reader.model.get_input_embeddings().weight.new_empty((0, reader.model.config.hidden_size))
    records = []
    for row in rows:
        before = torch.tensor(row.before_ids + row.history_ids, device=device)
        for q in row.queries:
            after = torch.tensor(q.after_ids, device=device)
            generated = generate_prefix_answer(reader, before, memory, after, max_new_tokens=8)
            records.append({**_score_answer(reader, before, memory, after, q, generated),
                            'history_id': row.history_id, 'code': row.code, 'condition': 'full_text'})
    return records


def summarize_reference(records, rows):
    expected = {q.case_id: (row, q) for row in rows for q in row.queries}
    if len(records) != len(expected) or {r['case_id'] for r in records} != set(expected):
        raise ValueError('reference prediction coverage differs')
    correct = {}
    for record in records:
        row, q = expected[record['case_id']]
        if (record['history_id'] != row.history_id or record['code'] != row.code
                or record['category'] != q.category or record['answer'] != q.answer or record['condition'] != 'full_text'):
            raise ValueError('reference labels differ from fixed source')
        ok = normalized_answer(record['prediction']) == normalized_answer(q.answer)
        if record['correct'] != ok:
            raise ValueError('stored reference score differs from prediction')
        correct[q.case_id] = ok
    groups = {}
    for group, selected in [('all', rows), *[(f'parity_{p}', [r for r in rows if r.code.bit_count() % 2 == p]) for p in (0, 1)]]:
        scores = {}
        for label, category in (('known', 'update_known'), ('absent', 'update_missing')):
            queries = [q for row in selected for q in row.queries if q.category == category]
            count = sum(correct[q.case_id] for q in queries)
            scores[label] = {'correct': count, 'total': len(queries), 'accuracy': count / len(queries),
                             'reliability_target_met': count / len(queries) >= .95}
        scores['all_six_correct_worlds'] = sum(all(correct[q.case_id] for q in row.queries) for row in selected)
        groups[group] = scores
    return {'scores': groups, 'scientific_decision': 'interpret_reference_errors_before_training',
            'reliability_target': .95, 'hard_scientific_cutoff': False}
