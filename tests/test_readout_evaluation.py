from dataclasses import replace

import pytest
import torch

from test_memory_updates import episode
from test_readout_runner import tiny_reader
from tinymem.memory.readout_interface import OneShotEncoder, ReadoutBridge
from tinymem.research.readout_runner import encode_before
from tinymem.research.readout_evaluation import evaluate_readout


@pytest.mark.parametrize("kind", ["affine", "gelu"])
def test_evaluation_writes_once_and_runs_all_controls(tiny_reader, kind, monkeypatch):
    from tinymem.research import readout_evaluation
    rows = [encode_before(tiny_reader, episode(i)) for i in range(2)]
    encoder, bridge = OneShotEncoder(16), ReadoutBridge(16, kind)
    original = readout_evaluation.encode_readout_history
    writes = []

    def observe(reader, writer, ids):
        writes.append(tuple(ids.tolist()))
        return original(reader, writer, ids)

    monkeypatch.setattr(readout_evaluation, "encode_readout_history", observe)
    results = evaluate_readout(tiny_reader, encoder, bridge, rows, max_new_tokens=1)
    assert writes == [row.history_ids for row in rows]
    assert len(results) == 80
    assert {r["condition"] for r in results} == {"normal", "zero", "no_memory", "shuffled"}
    for result in results:
        assert result["persistent_bytes"] == 66
        assert result["memory_positions"] == (0 if result["condition"] == "no_memory" else 2)
        assert result["answer_tokens"] == 2
        assert result["answer_ce"] == pytest.approx((result["first_answer_ce"] + result["stopping_ce"]) / 2, abs=1e-5)
        assert result["donor_history_id"] == (rows[1 if result["history_id"] == rows[0].history_id else 0].history_id
                                                  if result["condition"] == "shuffled" else result["history_id"])
    assert all(p.grad is None for p in tiny_reader.model.parameters())
    # Evaluation cannot depend on the order in which queries are requested.
    reordered = [replace(row, queries=tuple(reversed(row.queries))) for row in rows]
    other = evaluate_readout(tiny_reader, encoder, bridge, reordered, max_new_tokens=1)
    key = lambda r: (r["history_id"], r["condition"], r["case_id"])
    assert sorted(results, key=key) == sorted(other, key=key)


def test_evaluation_rejects_duplicate_histories_before_writing(tiny_reader):
    row = encode_before(tiny_reader, episode())
    with pytest.raises(ValueError, match="unique"):
        evaluate_readout(tiny_reader, OneShotEncoder(16), ReadoutBridge(16, "affine"), [row, row])
