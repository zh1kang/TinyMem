"""CPU-only checks for the one-off fixed-readout experiment runner."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import importlib.util

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from tinymem.data.reader_gate import ReaderCase
from tinymem.research.memory_prompt import NativeMemoryExample
from tinymem.research.native_memory_oracle import FixedProjectionMemoryOracle
from tinymem.research.prefix_reader import prefix_answer_loss
from tinymem.research.pretrained import PretrainedReader


spec = importlib.util.spec_from_file_location("fixed_oracle_runner", Path(__file__).with_name("fit_opaque_fixed_oracle.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixture():
    torch.manual_seed(71)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=32,
        eos_token_id=4, attention_dropout=0.0,
    )).requires_grad_(False)
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: "unknown" if ids == [4] else " ".join(map(str, ids)))
    reader = PretrainedReader(model, tokenizer)
    oracle = FixedProjectionMemoryOracle(torch.linspace(-1, 1, 64).reshape(4, 2, 8), torch.randn(16, 8))
    groups = [[(torch.tensor([1, 2]), torch.tensor([query + 5, 3]), torch.tensor([query + 14, 4]))
               for query in range(9)] for _ in range(4)]
    return reader, oracle, groups


def test_saved_state_conversion_preserves_values_exactly():
    values = torch.linspace(-1, 1, 64).reshape(4, 1, 2, 8)
    states = [{"values": value.tolist(), "valid": [[True, True]]} for value in values]
    oracle = runner.oracle_from_states(states, torch.randn(16, 8))
    assert torch.equal(oracle.codes, values[:, 0])
    assert all(oracle.state(index).nbytes == 66 for index in range(4))


@pytest.mark.parametrize("shape, valid", [
    ((2, 2, 8), [[True, True], [True, True]]),
    ((2, 8), [True, True]),
    ((1, 2, 8), [[1, 1]]),
    ((1, 2, 8), [[True, False]]),
])
def test_saved_state_conversion_rejects_extra_batches_shapes_and_non_boolean_masks(shape, valid):
    states = [{"values": torch.zeros(shape).tolist(), "valid": valid} for _ in range(4)]
    with pytest.raises(ValueError):
        runner.oracle_from_states(states, torch.randn(16, 8))


@pytest.mark.parametrize("checkpointing", [False, True])
def test_serial_update_matches_mean_world_mean_query_reference(checkpointing):
    reader, oracle, groups = fixture()
    if checkpointing:
        reader.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reader.model.train()
    reference = deepcopy(oracle)
    frozen = deepcopy(reader.model.state_dict())
    projection = oracle.projection.clone()
    optimizer = torch.optim.AdamW(oracle.parameters(), lr=0.001, weight_decay=0.01)
    expected_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.001, weight_decay=0.01)
    expected_loss = torch.stack([torch.stack([
        prefix_answer_loss(reader, before, reference(index), after, answer)
        for before, after, answer in group]).mean() for index, group in enumerate(groups)]).mean()
    expected_loss.backward()
    expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0, error_if_nonfinite=True)
    expected_optimizer.step()
    reference.project_codes_()
    actual = runner.fit_step(reader, oracle, groups, optimizer)
    assert actual["answer_ce"] == pytest.approx(float(expected_loss.detach()), abs=1e-6)
    assert actual["gradient_norm"] == pytest.approx(float(expected_norm), abs=1e-6)
    torch.testing.assert_close(oracle.codes, reference.codes, rtol=1e-6, atol=1e-7)
    assert all(value > 0 for value in actual["code_gradient_norms"])
    assert actual["queries_per_world"] == [9] * 4
    assert torch.equal(oracle.projection, projection)
    assert all(torch.equal(value, frozen[name]) for name, value in reader.model.state_dict().items())
    assert all(oracle.state(index).nbytes == 66 for index in range(4))


def test_training_rejects_trainable_reader():
    reader, oracle, groups = fixture()
    reader.model.requires_grad_(True)
    with pytest.raises(ValueError, match="reader must remain frozen"):
        runner.fit_step(reader, oracle, groups, torch.optim.AdamW(oracle.parameters()))


def test_evaluation_preserves_all_queries_and_control_semantics(tmp_path):
    reader, oracle, groups = fixture()
    worlds, examples = [], []
    for index, group in enumerate(groups):
        queries, encoded = [], []
        for query, (before, after, answer) in enumerate(group):
            category = "opaque_qa1_missing" if query == 8 else "opaque_qa1_known"
            case = ReaderCase(f"{index}:{query}", category, f"history{index}", f"facts{index}", f"question{query}",
                              "unknown" if query == 8 else "kitchen")
            queries.append(asdict(case))
            encoded.append(NativeMemoryExample(case.case_id, tuple(before.tolist()), (20, 21, 22),
                                               tuple(after.tolist()), tuple(answer.tolist())))
        worlds.append({"world_id": f"world{index}", "queries": queries})
        examples.append(encoded)
    rows = runner.evaluate(reader, oracle, worlds, examples, "initial", tmp_path / "predictions.jsonl")
    assert len(rows) == 180
    assert len({(row["case_id"], row["condition"]) for row in rows}) == 180
    for row in rows:
        expected = "unknown" if row["condition"] in ("drop", "zero", "cyclic_donor") else row["answer"]
        assert row["expected_answer"] == expected
        expected_positions = {"drop": 0, "zero": 2, "normal": 2, "cyclic_donor": 2, "full_context": 3}
        assert row["memory_positions"] == expected_positions[row["condition"]]
        if row["condition"] == "cyclic_donor":
            assert row["donor_world_id"] == f"world{(int(row['world_id'][-1]) + 1) % 4}"
        else:
            assert row["donor_world_id"] is None
    summary = runner.summarize(rows)
    assert len(summary) == 10
    for condition in runner.CONDITIONS:
        assert summary[f"{condition}:opaque_qa1_known"]["count"] == 32
        assert summary[f"{condition}:opaque_qa1_missing"]["count"] == 4
