from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from tinymem.research.memory_prompt import NativeMemoryExample
from tinymem.research.native_training import native_history_answer_loss
from tinymem.research.pretrained import PretrainedReader
from tinymem.research.recurrent_memory import NativeRecurrentMemory


@pytest.fixture
def reader():
    transformers = pytest.importorskip("transformers")
    model = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(
        vocab_size=16, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=32,
    )).requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return PretrainedReader(model, None)


def queries():
    first = NativeMemoryExample("first", (1, 3), (6, 7, 8, 9), (4, 5), (10, 2))
    return [first, replace(first, case_id="second", after_ids=(4, 11), answer_ids=(12, 2))]


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("writer_kind", ["narrow", "query_pool", "mean_pool"])
def test_multiquery_loss_matches_independent_query_gradients_and_writes_once(reader, device, writer_kind, monkeypatch):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    reader.model.to(device)
    torch.manual_seed(37)
    writer = NativeRecurrentMemory(16, memory_width=4, slots=2, segment_length=2, writer_kind=writer_kind).to(device)
    expected_writer = deepcopy(writer)
    original = writer.write
    writes, states = [], []

    def observe(shared_reader, state, ids):
        writes.append(ids.tolist())
        updated = original(shared_reader, state, ids)
        updated.values.retain_grad()
        states.append(updated)
        return updated

    monkeypatch.setattr(writer, "write", observe)
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        grouped = native_history_answer_loss(reader, writer, queries())
        grouped.backward()
        separate = torch.stack([native_history_answer_loss(reader, expected_writer, [query]) for query in queries()]).mean()
        separate.backward()
        torch.testing.assert_close(grouped, separate)
        for actual, expected in zip(writer.parameters(), expected_writer.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=2e-4)
        assert writes == [[6, 7], [8, 9]]
        assert states[0].values.grad.abs().sum() > 0
        assert all(parameter.grad is None for parameter in reader.model.parameters())
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def test_group_validation_prevents_mixed_histories_and_duplicate_queries(reader):
    writer = NativeRecurrentMemory(16, memory_width=4, slots=2, segment_length=2)
    first, second = queries()
    with pytest.raises(ValueError, match="nonempty"):
        native_history_answer_loss(reader, writer, [])
    with pytest.raises(ValueError, match="nonempty"):
        native_history_answer_loss(reader, writer, [replace(first, history_ids=())])
    for changed in (replace(second, history_ids=(6, 7)), replace(second, before_ids=(1,))):
        with pytest.raises(ValueError, match="share one history"):
            native_history_answer_loss(reader, writer, [first, changed])
    with pytest.raises(ValueError, match="distinct"):
        native_history_answer_loss(reader, writer, [first, replace(second, after_ids=first.after_ids)])


def test_explicit_unequal_chunks_preserve_write_boundaries_and_gradients(reader, monkeypatch):
    writer = NativeRecurrentMemory(16, memory_width=4, slots=2, segment_length=3)
    writes, states = [], []
    original = writer.write

    def observe(shared_reader, state, ids):
        writes.append(ids.tolist())
        updated = original(shared_reader, state, ids)
        updated.values.retain_grad()
        states.append(updated)
        return updated

    monkeypatch.setattr(writer, "write", observe)
    native_history_answer_loss(reader, writer, queries(), history_chunks=((6,), (7, 8, 9))).backward()
    assert writes == [[6], [7, 8, 9]]
    assert states[0].values.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in reader.model.parameters())


@pytest.mark.parametrize("chunks", [(), ((), (6, 7, 8, 9)), ((6, 7, 8), (9,)), ((6, 7), (9, 8)), ((6, 7),)])
def test_invalid_explicit_chunks(reader, chunks):
    writer = NativeRecurrentMemory(16, memory_width=4, slots=2, segment_length=2)
    with pytest.raises(ValueError):
        native_history_answer_loss(reader, writer, queries(), history_chunks=chunks)


@pytest.mark.parametrize("token", [True, 6.0])
def test_explicit_chunk_token_type(reader, token):
    writer = NativeRecurrentMemory(16, memory_width=4, slots=2, segment_length=2)
    with pytest.raises(TypeError):
        native_history_answer_loss(reader, writer, queries(), history_chunks=((token, 7), (8, 9)))


@pytest.mark.parametrize("token", [-1, 16])
def test_later_invalid_chunk_is_rejected_before_state_creation(reader, monkeypatch, token):
    writer = NativeRecurrentMemory(16, memory_width=4, slots=2, segment_length=2)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid history must fail before state creation or writes")

    monkeypatch.setattr(writer.writer, "empty", forbidden)
    monkeypatch.setattr(writer, "write", forbidden)
    examples = [replace(case, history_ids=(6, 7, 8, token)) for case in queries()]
    with pytest.raises(ValueError, match="out-of-vocabulary"):
        native_history_answer_loss(reader, writer, examples, history_chunks=((6, 7), (8, token)))
