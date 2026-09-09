from dataclasses import replace

import pytest
import torch

from test_memory_updates import episode
from test_readout_runner import tiny_reader
from tinymem.research.readout_runner import encode_before
from tinymem.research.readout_evaluation import evaluate_full_text


def test_full_text_keeps_all_queries_and_exact_native_tokens(tiny_reader):
    rows = [encode_before(tiny_reader, episode(i)) for i in range(2)]
    inputs = []

    def capture(module, args, kwargs):
        assert kwargs['use_cache'] is False
        inputs.append(kwargs['inputs_embeds'].detach().clone())

    handle = tiny_reader.model.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        results = evaluate_full_text(tiny_reader, rows, max_new_tokens=1)
    finally:
        handle.remove()
    assert len(results) == 20
    for i, (row, query) in enumerate((r, q) for r in rows for q in r.queries):
        native = torch.tensor(row.before_ids + row.history_ids + query.after_ids)
        torch.testing.assert_close(inputs[4 * i], tiny_reader.model.get_input_embeddings()(native).unsqueeze(0))
        record = results[i]
        assert record['condition'] == 'full_text'
        assert record['persistent_bytes'] is None
        assert record['memory_positions'] == 0
        assert record['history_tokens'] == len(row.history_ids)
        assert record['native_envelope_tokens'] == len(row.before_ids) + len(query.after_ids)
        assert record['input_positions'] == native.numel()
    other = evaluate_full_text(tiny_reader, [replace(r, queries=tuple(reversed(r.queries))) for r in rows], max_new_tokens=1)
    key = lambda r: r['case_id']
    assert sorted(results, key=key) == sorted(other, key=key)
    assert all(p.grad is None and not p.requires_grad for p in tiny_reader.model.parameters())


@pytest.mark.parametrize('defect', ['context', 'token', 'empty', 'duplicate', 'budget', 'unfrozen'])
def test_full_text_preflights_all_inputs_before_forward(tiny_reader, defect):
    rows = [encode_before(tiny_reader, episode(i)) for i in range(2)]
    budget = 1
    if defect == 'context':
        rows[-1] = replace(rows[-1], history_ids=(1,) * 2048)
    elif defect == 'token':
        q = replace(rows[-1].queries[-1], answer_ids=(True, 0))
        rows[-1] = replace(rows[-1], queries=(*rows[-1].queries[:-1], q))
    elif defect == 'empty':
        rows[-1] = replace(rows[-1], queries=())
    elif defect == 'duplicate':
        rows[-1] = rows[0]
    elif defect == 'budget':
        budget = True
    else:
        next(tiny_reader.model.parameters()).requires_grad_(True)

    def forbidden(*args, **kwargs):
        pytest.fail('invalid input reached the reader')

    handle = tiny_reader.model.register_forward_pre_hook(forbidden)
    try:
        with pytest.raises(ValueError):
            evaluate_full_text(tiny_reader, rows, max_new_tokens=budget)
    finally:
        handle.remove()
