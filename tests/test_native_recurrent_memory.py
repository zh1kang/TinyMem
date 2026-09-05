from copy import deepcopy

import pytest
import torch

from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.prefix_reader import prefix_answer_loss
from tinymem.research.pretrained import PretrainedReader
from tinymem.research.recurrent_memory import NativeRecurrentMemory


@pytest.fixture
def reader():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(27)
    config = transformers.Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
                                     num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=32)
    model = transformers.Qwen3ForCausalLM(config).eval().requires_grad_(False)
    return PretrainedReader(model, None)


def memory(writer_kind="narrow"):
    return NativeRecurrentMemory(16, memory_width=4, slots=2, segment_length=4, writer_kind=writer_kind)


@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("writer_kind", ["narrow", "query_pool"])
def test_future_answer_trains_earlier_native_memory_with_frozen_reader(reader, lora, writer_kind):
    if lora:
        pytest.importorskip("peft")
        from tinymem.research.reader_adaptation import attach_reader_lora

        attach_reader_lora(reader, rank=2, checkpointing=True)
        with torch.no_grad():
            for name, parameter in reader.model.named_parameters():
                if "lora_B" in name:
                    parameter.normal_(std=0.01)
        reader.model.requires_grad_(False)
    reader.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reader.model.train()
    frozen = deepcopy(reader.model.state_dict())
    compressor = memory(writer_kind)
    before, after, answer = (torch.tensor(ids) for ids in ([3], [5, 7], [8, 2]))
    first = compressor.write(reader, compressor.writer.empty(1), torch.tensor([11, 12]))
    first.values.retain_grad()
    final = compressor.write(reader, first, torch.tensor([13, 14, 15]))
    prefix_answer_loss(reader, before, compressor.memory_vectors(final), after, answer).backward()
    assert first.values.grad is not None and torch.isfinite(first.values.grad).all()
    assert first.values.grad.abs().sum() > 0
    for parameter in (compressor.writer.input_projection.weight, compressor.read_projection.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in reader.model.parameters())
    for name, parameter in reader.model.state_dict().items():
        assert torch.equal(parameter, frozen[name])
    compressor.zero_grad(set_to_none=True)
    first = compressor.write(reader, compressor.writer.empty(1), torch.tensor([11, 12]))
    first.values.retain_grad()
    final = compressor.write(reader, first.detached(), torch.tensor([13, 14, 15]))
    prefix_answer_loss(reader, before, compressor.memory_vectors(final), after, answer).backward()
    assert first.values.grad is None


@pytest.mark.parametrize("writer_kind", ["narrow", "query_pool"])
def test_writes_use_only_current_chunk_and_valid_memory_without_cache(reader, writer_kind, monkeypatch):
    compressor = memory(writer_kind).eval()
    calls = []

    def inspect(module, args, kwargs, output):
        assert not kwargs["use_cache"] and output.past_key_values is None
        assert "past_key_values" not in kwargs
        inputs = kwargs["inputs_embeds"]
        assert kwargs["position_ids"].tolist() == [list(range(inputs.shape[1]))]
        calls.append(inputs.detach().clone())

    monkeypatch.setattr(reader.model.lm_head, "forward", lambda *args: pytest.fail("writes must not compute vocabulary logits"))
    hook = reader.model.get_decoder().register_forward_hook(inspect, with_kwargs=True)
    initial = compressor.writer.empty(1)
    try:
        with torch.no_grad():
            first = compressor.write(reader, initial, torch.tensor([11, 12]))
            old = first.values.clone()
            state = compressor.write(reader, first, torch.tensor([13, 14, 15]))
    finally:
        hook.remove()
    assert [call.shape for call in calls] == [(1, 2, 16), (1, 5, 16)]
    assert torch.equal(calls[0][0], reader.model.get_input_embeddings()(torch.tensor([11, 12])))
    torch.testing.assert_close(calls[1][0, :2], compressor.memory_vectors(first))
    assert torch.equal(first.values, old)
    assert initial.values.count_nonzero() == 0 and not initial.valid.any()
    assert state.values.grad_fn is None
    assert state.nbytes == initial.nbytes == 34
    assert set(vars(state)) == {"values", "valid"}
    assert all("reader" not in name for name in compressor.state_dict())
    assert not any(isinstance(value, torch.Tensor) for value in vars(compressor).values())


def test_empty_and_invalid_slots_never_enter_read_projection_gradients(reader):
    compressor = memory()
    state = compressor.writer.empty(1)
    assert compressor.memory_vectors(state).shape == (0, 16)
    assert not compressor.memory_vectors(state).requires_grad
    values = torch.randn(1, 2, 4)
    values[:, 1] = float("nan")
    values.requires_grad_()
    masked = LatentSlotState(values, torch.tensor([[True, False]]))
    vectors = compressor.memory_vectors(masked)
    assert vectors.shape == (1, 16) and torch.isfinite(vectors).all()
    vectors.sum().backward()
    assert torch.isfinite(compressor.read_projection.weight.grad).all()
    assert values.grad[0, 1].count_nonzero() == 0


def test_empty_bank_skips_useless_reader_graph_but_populated_bank_keeps_it(reader):
    compressor = memory()
    observed = []
    hook = reader.model.get_decoder().register_forward_hook(
        lambda module, args, kwargs, output: observed.append(
            (kwargs["inputs_embeds"].requires_grad, output.last_hidden_state.requires_grad)
        ), with_kwargs=True,
    )
    try:
        first = compressor.write(reader, compressor.writer.empty(1), torch.tensor([11, 12]))
        compressor.write(reader, first, torch.tensor([13, 14]))
    finally:
        hook.remove()
    assert observed == [(False, False), (True, True)]


def test_native_writer_uses_the_active_lora_decoder_features(reader):
    pytest.importorskip("peft")
    from tinymem.research.reader_adaptation import attach_reader_lora

    attach_reader_lora(reader, rank=2, checkpointing=False)
    with torch.no_grad():
        for name, parameter in reader.model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.2)
    reader.model.requires_grad_(False).eval()
    ids = torch.tensor([11, 12, 13])
    expected = reader.model(input_ids=ids.unsqueeze(0), use_cache=False, output_hidden_states=True).hidden_states[-1]
    compressor = memory()
    inputs = []
    hook = compressor.writer.register_forward_pre_hook(lambda module, args: inputs.append(args[1].detach().clone()))
    try:
        state = compressor.write(reader, compressor.writer.empty(1), ids)
    finally:
        hook.remove()
    torch.testing.assert_close(inputs[0], expected)
    with reader.model.disable_adapter():
        changed = compressor.write(reader, compressor.writer.empty(1), ids)
    assert not torch.allclose(state.values, changed.values)


@pytest.mark.parametrize("writer_kind", ["narrow", "query_pool"])
def test_queries_do_not_update_memory_and_reset_has_no_hidden_history(reader, writer_kind):
    compressor = memory(writer_kind).eval()
    with torch.no_grad():
        first = compressor.write(reader, compressor.writer.empty(1), torch.tensor([10, 11, 12, 13]))
        saved = first.values.clone()
        vectors = compressor.memory_vectors(first)
        before, after, answer = (torch.tensor(ids) for ids in ([3], [5, 7], [8, 2]))
        initial_loss = prefix_answer_loss(reader, before, vectors, after, answer)
        prefix_answer_loss(reader, before, vectors, after + 1, answer + 1)
        final_loss = prefix_answer_loss(reader, before, compressor.memory_vectors(first), after, answer)
        repeated = compressor.write(reader, compressor.writer.empty(1), torch.tensor([10, 11, 12, 13]))
    assert torch.equal(initial_loss, final_loss)
    assert torch.equal(first.values, saved)
    assert torch.equal(repeated.values, first.values)
    assert compressor.memory_vectors(compressor.writer.empty(1)).numel() == 0


@pytest.mark.parametrize("writer_kind", ["narrow", "query_pool"])
def test_native_memory_optimizes_answer_loss_without_updating_reader(reader, writer_kind):
    compressor = memory(writer_kind)
    optimizer = torch.optim.AdamW(compressor.parameters(), lr=0.01)
    history = torch.tensor([10, 11, 12, 13, 14, 15])
    before, after, answer = (torch.tensor(ids) for ids in ([3], [5, 7], [8, 2]))
    losses = []
    initial = deepcopy(compressor.state_dict())
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        state = compressor.writer.empty(1)
        for chunk in history.split(3):
            state = compressor.write(reader, state, chunk)
        loss = prefix_answer_loss(reader, before, compressor.memory_vectors(state), after, answer)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(compressor.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
    assert not torch.equal(initial["writer.input_projection.weight"], compressor.writer.input_projection.weight)
    assert not torch.equal(initial["read_projection.weight"], compressor.read_projection.weight)
    assert all(parameter.grad is None for parameter in reader.model.parameters())


def test_native_memory_rejects_wrong_inputs_and_trainable_reader(reader, monkeypatch):
    compressor = memory()
    state = compressor.writer.empty(1)
    for ids, exception, message in ((torch.tensor([], dtype=torch.long), ValueError, "segment_length"),
                                    (torch.arange(5), ValueError, "segment_length"),
                                    (torch.tensor([1.0]), TypeError, "int32"),
                                    (torch.tensor([32]), ValueError, "vocabulary")):
        with pytest.raises(exception, match=message):
            compressor.write(reader, state, ids)
    with pytest.raises(ValueError, match="shape"):
        compressor.memory_vectors(compressor.writer.empty(2))
    reader.model.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        compressor.write(reader, state, torch.tensor([1]))
    reader.model.requires_grad_(False)
    with pytest.raises(ValueError, match="finite"):
        compressor.memory_vectors(LatentSlotState(state.values + float("nan"), torch.ones_like(state.valid)))
    reader.model.config.max_position_embeddings = 2
    monkeypatch.setattr(reader.model.get_input_embeddings(), "forward", lambda *args: pytest.fail("fail before embedding"))
    populated = LatentSlotState(state.values, torch.ones_like(state.valid))
    with pytest.raises(ValueError, match="truncation"):
        compressor.write(reader, populated, torch.tensor([1]))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
@pytest.mark.parametrize("writer_kind", ["narrow", "query_pool"])
def test_native_memory_backward_is_deterministic_on_mps_with_partial_slots(reader, writer_kind):
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        reader.model.to("mps")
        compressor = memory(writer_kind).to("mps")
        values = torch.randn(1, 2, 4, device="mps", requires_grad=True)
        partial = LatentSlotState(values, torch.tensor([[False, True]], device="mps"))
        compressor.memory_vectors(partial).square().sum().backward()
        assert values.grad[0, 0].count_nonzero() == 0
        assert values.grad[0, 1].abs().sum() > 0
        compressor.zero_grad(set_to_none=True)
        first = compressor.write(reader, compressor.writer.empty(1), torch.tensor([11, 12], device="mps"))
        first.values.retain_grad()
        final = compressor.write(reader, first, torch.tensor([13, 14, 15], device="mps"))
        before, after, answer = (torch.tensor(ids, device="mps") for ids in ([3], [5, 7], [8, 2]))
        loss = prefix_answer_loss(reader, before, compressor.memory_vectors(final), after, answer)
        loss.backward()
        torch.mps.synchronize()
        assert torch.isfinite(first.values.grad).all() and first.values.grad.abs().sum() > 0
        assert all(parameter.grad is None for parameter in reader.model.parameters())
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
