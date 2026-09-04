import torch
import pytest
from dataclasses import replace

from tinymem.memory.recurrent_slots import LatentSlotState, RecurrentSlotWriter
from tinymem.model.config import ModelConfig
from tinymem.model.recurrent_slot_decoder import RecurrentSlotDecoder
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.data.replacement_qa import generate_replacement_qa_examples
from tinymem.tokenization.byte_tokenizer import ByteTokenizer
from tinymem.training.recurrent_slots import write_replacement_histories, recurrent_slot_answer_loss


def decoder():
    torch.manual_seed(9)
    return RecurrentSlotDecoder(
        DecoderOnlyTransformer(ModelConfig(vocab_size=260, d_model=8, n_layers=1, n_heads=2, d_ff=16, max_local_tokens=64, dropout=0.0)),
        memory_width=4, slots=2, segment_length=32,
    )


def test_writer_is_bounded_functional_and_preserves_empty_rows():
    writer = RecurrentSlotWriter(8, 4, 2)
    state = writer.empty(2)
    initial = state
    original = state.values.clone()
    for _ in range(20):
        next_state = writer(state, torch.randn(2, 5, 8), torch.tensor([[True] * 5, [False] * 5]))
        torch.testing.assert_close(next_state.values[1], state.values[1])
        assert next_state.nbytes == 2 * 2 * (4 * 4 + 1)
        assert torch.isfinite(next_state.values).all()
        state = next_state
    torch.testing.assert_close(initial.values, original)
    assert state.valid.tolist() == [[True, True], [False, False]]


def test_padded_values_cannot_change_written_state():
    writer = RecurrentSlotWriter(8, 4, 2)
    state = writer.empty(1)
    hidden = torch.randn(1, 4, 8)
    valid = torch.tensor([[True, True, False, False]])
    changed = hidden.clone()
    changed[:, 2:] = float("nan")
    torch.testing.assert_close(writer(state, hidden, valid).values, writer(state, changed, valid).values)


def test_delayed_loss_reaches_earlier_state_with_frozen_reader():
    model = decoder()
    model.reader.requires_grad_(False)
    ids = torch.tensor([[12, 13, 14]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    first = model.write(ids, valid, model.writer.empty(1))
    first.values.retain_grad()
    second = model.write(ids + 1, valid, first)
    model(ids + 2, second).square().mean().backward()
    assert first.values.grad is not None and first.values.grad.abs().sum() > 0
    assert model.writer.input_projection.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.reader.parameters())

    model.zero_grad(set_to_none=True)
    first = model.write(ids, valid, model.writer.empty(1))
    first.values.retain_grad()
    second = model.write(ids + 1, valid, first.detached())
    model(ids + 2, second).sum().backward()
    assert first.values.grad is None


def test_reads_do_not_change_memory_or_retain_a_query_cache():
    model = decoder().eval()
    ids = torch.tensor([[11, 12, 13]])
    with torch.no_grad():
        state = model.write(ids, torch.ones_like(ids, dtype=torch.bool), model.writer.empty(1))
        saved = state.values.clone()
        before = model(ids, state)
        model(ids + 20, state)
        after = model(ids, state)
        torch.testing.assert_close(before, after)
        torch.testing.assert_close(saved, state.values)
        assert set(vars(state)) == {"values", "valid"}
        assert state.values.grad_fn is None
    assert not any("cache" in name for name in vars(model))


def test_invalid_padding_and_positions_fail_explicitly():
    model = decoder()
    with pytest.raises(ValueError, match="right padded"):
        model.write(torch.tensor([[1, 2, 3]]), torch.tensor([[True, False, True]]), model.writer.empty(1))
    with pytest.raises(ValueError, match="segment_length"):
        model(torch.zeros(1, 33, dtype=torch.long), model.writer.empty(1))


def test_history_rollout_ignores_future_query_and_labels_and_trains_end_to_end():
    model = RecurrentSlotDecoder(
        DecoderOnlyTransformer(ModelConfig(vocab_size=260, d_model=8, n_layers=1, n_heads=2, d_ff=16, max_local_tokens=128, dropout=0.0)),
        memory_width=4, slots=2, segment_length=64,
    )
    rows = generate_replacement_qa_examples(ByteTokenizer(), split="train", count=4, memory_capacity=2)
    altered = [replace(row, query_ids=(1, 2), answer_ids=(3,), correction_slot=1-row.correction_slot) for row in rows]
    first = write_replacement_histories(model, rows)
    second = write_replacement_histories(model, altered)
    torch.testing.assert_close(first.values, second.values)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    before = model.writer.input_projection.weight.detach().clone()
    loss = recurrent_slot_answer_loss(model, rows)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert not torch.equal(before, model.writer.input_projection.weight)
