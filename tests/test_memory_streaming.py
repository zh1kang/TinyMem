import pytest
import torch

from tinymem.memory.heavy_hitter import HeavyHitterMemory
from tinymem.model.config import ModelConfig
from tinymem.model.streaming import StreamingDecoder
from tinymem.model.transformer import DecoderOnlyTransformer


def make_model() -> DecoderOnlyTransformer:
    return DecoderOnlyTransformer(
        ModelConfig(
            vocab_size=16,
            d_model=8,
            n_layers=2,
            n_heads=2,
            d_ff=16,
            max_local_tokens=4,
        )
    ).eval()


def make_memory_stream(model: DecoderOnlyTransformer) -> StreamingDecoder:
    return StreamingDecoder(
        model,
        segment_length=2,
        memory_policy=HeavyHitterMemory(capacity=2, recent_slots=1),
    )


def test_memory_stream_preserves_local_only_logits_before_read_integration() -> None:
    torch.manual_seed(41)
    model = make_model()
    local_stream = StreamingDecoder(model, segment_length=2)
    memory_stream = make_memory_stream(model)
    input_ids = torch.tensor([[1, 2, 3, 4]])

    local_logits = local_stream.process_segment(input_ids[:, :2])
    local_logits = torch.cat(
        (local_logits, local_stream.process_segment(input_ids[:, 2:])),
        dim=1,
    )
    memory_logits = memory_stream.process_segment(input_ids[:, :2])
    memory_logits = torch.cat(
        (memory_logits, memory_stream.process_segment(input_ids[:, 2:])),
        dim=1,
    )

    torch.testing.assert_close(memory_logits, local_logits)
    assert not memory_stream.memory_state.valid.any()


def test_memory_stream_stores_expired_tokens_with_cumulative_scores() -> None:
    torch.manual_seed(43)
    model = make_model()
    stream = make_memory_stream(model)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])

    for start in range(0, input_ids.shape[1], 2):
        stream.process_segment(input_ids[:, start : start + 2])

    state = stream.memory_state
    assert torch.equal(state.valid, torch.tensor([[True, True]]))
    assert torch.equal(state.positions, torch.tensor([[0, 1]]))
    assert torch.equal(state.token_ids, torch.tensor([[1, 2]]))
    assert state.scores is not None
    assert torch.isfinite(state.scores[state.valid]).all()
    assert (state.scores[state.valid] > 0).all()
    torch.testing.assert_close(
        state.values,
        model.token_embedding(torch.tensor([[1, 2]])).detach(),
    )


def test_memory_stream_returns_a_defensive_memory_copy() -> None:
    stream = make_memory_stream(make_model())
    stream.process_segment(torch.tensor([[1, 2]]))

    copied_state = stream.memory_state
    copied_state.valid.fill_(True)
    copied_state.positions.fill_(9)

    assert not stream.memory_state.valid.any()
    assert (stream.memory_state.positions == -1).all()


def test_memory_stream_reset_clears_all_owned_state() -> None:
    stream = make_memory_stream(make_model())
    stream.process_segment(torch.tensor([[1, 2]]))

    stream.reset()

    assert stream.position == 0
    assert stream.cache_bytes == 0
    assert stream.memory_bytes == 0
    with pytest.raises(RuntimeError, match="not been initialized"):
        _ = stream.memory_state


def test_memory_stream_rejects_invalid_policy() -> None:
    with pytest.raises(TypeError, match="memory_policy"):
        StreamingDecoder(
            make_model(),
            segment_length=2,
            memory_policy=object(),
        )
