from copy import deepcopy
from dataclasses import asdict
import io
import json

import pytest
import torch

from test_readout_runner import tiny_reader
from test_memory_updates import episode
from tinymem.memory.readout_interface import OneShotEncoder, ReadoutBridge
from tinymem.research.readout_evaluation import evaluate_readout
from tinymem.research.readout_runner import encode_before
from tinymem.research.readout_experiment import run_arm, verify_run
from scripts.diagnose_readout_ranking import (
    CANDIDATES, candidate_scores, evaluate_ranking, load_inputs, ranking_metrics, select_histories,
)


def test_complete_candidate_scores_match_explicit_causal_log_probabilities(tiny_reader):
    before, after = torch.tensor([1, 2]), torch.tensor([3, 4])
    memory = torch.randn(2, 16)
    candidates = {"short": (5, 0), "long": (6, 7, 0)}
    weights = deepcopy(tiny_reader.model.state_dict())
    actual = candidate_scores(tiny_reader, before, memory, after, candidates)
    embedding = tiny_reader.model.get_input_embeddings()
    for name, tokens in candidates.items():
        prompt = torch.cat((embedding(before), memory, embedding(after)))
        sequence = torch.cat((prompt, embedding(torch.tensor(tokens[:-1]))))
        with torch.no_grad():
            logits = tiny_reader.model(inputs_embeds=sequence.unsqueeze(0), use_cache=False).logits[0]
        start = len(prompt) - 1
        expected = sum(float(torch.log_softmax(logits[start + i].float(), -1)[token])
                       for i, token in enumerate(tokens))
        assert actual[name] == pytest.approx(expected, abs=2e-6)
    assert all(torch.equal(v, weights[k]) for k, v in tiny_reader.model.state_dict().items())
    assert all(p.grad is None for p in tiny_reader.model.parameters())


def test_ranking_includes_abstention_and_does_not_award_ties():
    result = ranking_metrics({"office": -3.0, "garden": -4.0, "unknown": -2.0}, "office")
    assert result["top1_correct"] is False
    assert result["gold_rank_min"] == result["gold_rank_max"] == 2
    assert result["gold_minus_unknown"] == -1.0
    assert result["gold_margin"] == -1.0
    assert result["known_only_top1_correct"] is True
    tied = ranking_metrics({"office": -2.0, "garden": -4.0, "unknown": -2.0}, "office")
    assert tied["top1_correct"] is False
    assert (tied["gold_rank_min"], tied["gold_rank_max"]) == (1, 2)
    assert ranking_metrics({"office": -3.0, "unknown": -2.0}, "unknown")["known_only_top1_correct"] is None
    with pytest.raises(ValueError, match="finite"):
        ranking_metrics({"office": float("nan"), "unknown": -2.0}, "office")


def test_selection_is_order_independent_and_only_uses_history_identity():
    rows = [{"history_id": f"history-{i}", "answer": "office"} for i in range(10)]
    first = select_histories(rows, 4)
    changed = [{**row, "answer": "unknown"} for row in reversed(rows)]
    assert first == select_histories(changed, 4)
    assert len(set(first)) == 4
    with pytest.raises(ValueError, match="unique"):
        select_histories(rows + rows, 4)
    with pytest.raises(ValueError, match="count"):
        select_histories(rows, 11)
    with pytest.raises(ValueError, match="exactly 32"):
        load_inputs(None, 31)


def test_scoring_rejects_a_training_child_module(tiny_reader):
    next(module for module in tiny_reader.model.modules() if module is not tiny_reader.model).train()
    assert not tiny_reader.model.training
    with pytest.raises(ValueError, match="evaluation mode"):
        candidate_scores(tiny_reader, torch.tensor([1]), torch.zeros(2, 16), torch.tensor([2]), {"office": (3, 0)})


@pytest.mark.parametrize("kind", ["affine", "gelu"])
def test_real_checkpoint_read_path_replays_gold_and_preserves_full_donor_cycle(tiny_reader, kind):
    encoded = [encode_before(tiny_reader, episode(i)) for i in range(3)]
    encoder, bridge = OneShotEncoder(16), ReadoutBridge(16, kind)
    original = evaluate_readout(tiny_reader, encoder, bridge, encoded, max_new_tokens=1)
    saved = {(r["history_id"], r["case_id"], r["condition"]): r for r in original}
    rows = [asdict(row) for row in encoded]
    selected = [rows[0]["history_id"], rows[2]["history_id"]]
    candidates = {name: (*tiny_reader.tokenizer.encode(name, add_special_tokens=False), 0) for name in CANDIDATES}
    output = io.StringIO()
    results = evaluate_ranking(tiny_reader, encoder, bridge, rows, saved, selected, candidates, output)
    assert len(results) == 60
    assert [json.loads(line) for line in output.getvalue().splitlines()] == results
    assert max(r["gold_ce_replay_error"] for r in results) < 3e-6
    for result in results:
        assert result["donor_history_id"] == saved[result["history_id"], result["case_id"], result["condition"]]["donor_history_id"]
    # The donor for the first history is deliberately outside the selected subset.
    assert next(r for r in results if r["history_id"] == selected[0] and r["condition"] == "shuffled")["donor_history_id"] == rows[1]["history_id"]
    first = selected[0], rows[0]["queries"][0]["case_id"], "normal"
    saved[first] = {**saved[first], "answer_ce": saved[first]["answer_ce"] + 0.1}
    with pytest.raises(ValueError, match="gold CE replay"):
        evaluate_ranking(tiny_reader, encoder, bridge, rows, saved, selected, candidates, io.StringIO())


def test_tiny_sealed_checkpoint_diagnostic_end_to_end(tiny_reader, tmp_path, monkeypatch, request):
    from scripts import diagnose_readout_ranking as diagnostic
    from tinymem.research.update_protocol import file_sha256
    previous = torch.are_deterministic_algorithms_enabled()
    request.addfinalizer(lambda: torch.use_deterministic_algorithms(previous))
    diagnostic.prepare_device("cpu")
    train = [encode_before(tiny_reader, episode(i)) for i in range(2)]
    development = [encode_before(tiny_reader, episode(i)) for i in range(2, 4)]
    source = tmp_path / "source"
    run_arm(tiny_reader, {"train": train, "development": development}, source,
            kind="affine", seed=17, steps=1, learning_rate=0.001, weight_decay=0.01,
            max_new_tokens=1, input_identity={"reader": {}, "fixture": True})
    seal = verify_run(source)
    protocol = json.loads((source / "protocol.json").read_text())
    records = [json.loads(line) for line in (source / "predictions.jsonl").read_text().splitlines()]
    saved = {(r["history_id"], r["case_id"], r["condition"]): r for r in records
             if r["split"] == "train" and r["phase"] == "final" and r["condition"] in diagnostic.CONDITIONS}
    # Only replace the production dataset-size and model-loading boundaries.
    # Training, serialization, candidate scoring, replay, and seals remain real.
    monkeypatch.setattr(diagnostic, "load_inputs", lambda directory, count: (
        seal, protocol, [asdict(row) for row in train], saved, [row.history_id for row in train]))
    monkeypatch.setattr(diagnostic, "load_shared_reader", lambda identity, device: tiny_reader)
    checkpoint_loader = diagnostic.load_checkpoint
    loaded_modules = []

    def record_checkpoint(*args, **kwargs):
        modules = checkpoint_loader(*args, **kwargs)
        loaded_modules.extend(modules)
        return modules

    monkeypatch.setattr(diagnostic, "load_checkpoint", record_checkpoint)
    output = tmp_path / "diagnostic"
    diagnostic.run(source, output, 32, "cpu")
    assert all(p.requires_grad and p.grad is None for module in loaded_modules for p in module.parameters())
    assert all(not child.training for module in loaded_modules for child in module.modules())
    complete = json.loads((output / "complete.json").read_text())
    assert complete["records"] == 60
    assert complete["max_gold_ce_replay_error"] < 3e-6
    for name, digest in complete["files"].items():
        assert file_sha256(output / name) == digest
    assert verify_run(source) == seal
    with pytest.raises(FileExistsError):
        diagnostic.run(source, output, 32, "cpu")
    with pytest.raises(ValueError, match="outside the sealed"):
        diagnostic.run(source, source / "bad-output", 32, "cpu")


def test_ranking_preserves_full_original_state_construction_order(tiny_reader, monkeypatch):
    from scripts import diagnose_readout_ranking as diagnostic

    encoded = [encode_before(tiny_reader, episode(i)) for i in (3, 1, 0, 2)]
    encoder, bridge = OneShotEncoder(16), ReadoutBridge(16, "affine")
    original = evaluate_readout(tiny_reader, encoder, bridge, encoded, max_new_tokens=1)
    saved = {(r["history_id"], r["case_id"], r["condition"]): r for r in original}
    observed = []
    encode = diagnostic.encode_readout_history

    def record_history(reader, encoder, history_ids):
        observed.append(tuple(history_ids.tolist()))
        assert torch.is_inference_mode_enabled()
        assert all(p.requires_grad and p.grad is None for p in encoder.parameters())
        assert all(not child.training for child in encoder.modules())
        return encode(reader, encoder, history_ids)

    monkeypatch.setattr(diagnostic, "encode_readout_history", record_history)
    candidates = {name: (*tiny_reader.tokenizer.encode(name, add_special_tokens=False), 0)
                  for name in CANDIDATES}
    selected = [encoded[0].history_id]
    records = evaluate_ranking(tiny_reader, encoder, bridge, [asdict(row) for row in encoded],
                               saved, selected, candidates, io.StringIO())
    # Preserve the complete original state-construction sequence for numerical replay.
    assert observed == [row.history_ids for row in encoded]
    assert len(records) == 30
    assert {row["history_id"] for row in records} == set(selected)
    assert max(row["gold_ce_replay_error"] for row in records) < 3e-6
