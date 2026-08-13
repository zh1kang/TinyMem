import pytest
import torch

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


def test_streaming_segments_match_full_context_within_window() -> None:
    torch.manual_seed(31)
    model = make_model()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    stream = StreamingDecoder(model, segment_length=2)

    full_logits = model(input_ids)
    streamed_logits = torch.cat(
        [
            stream.process_segment(input_ids[:, :2]),
            stream.process_segment(input_ids[:, 2:]),
        ],
        dim=1,
    )

    torch.testing.assert_close(streamed_logits, full_logits)
    assert stream.position == 4
    assert stream.caches[0].sequence_length == 4


def test_streaming_slides_cache_and_preserves_absolute_position() -> None:
    model = make_model()
    stream = StreamingDecoder(model, segment_length=2)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])

    logits = stream.process_segment(input_ids[:, :2])
    logits = torch.cat((logits, stream.process_segment(input_ids[:, 2:4])), dim=1)
    logits = torch.cat((logits, stream.process_segment(input_ids[:, 4:])), dim=1)

    assert logits.shape == (1, 6, 16)
    assert stream.position == 6
    assert stream.caches[0].start_position == 2
    assert stream.caches[0].end_position == 6
    assert stream.caches[0].sequence_length == 4
    assert stream.cache_bytes > 0


def test_streaming_reset_clears_state() -> None:
    stream = StreamingDecoder(make_model(), segment_length=2)
    stream.process_segment(torch.tensor([[1, 2]]))

    stream.reset()

    assert stream.position == 0
    assert stream.cache_bytes == 0
    assert all(cache.sequence_length == 0 for cache in stream.caches)


@pytest.mark.parametrize("segment_length", [0, -1, True, 5, 2.0])
def test_streaming_rejects_invalid_segment_length(segment_length: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        StreamingDecoder(make_model(), segment_length=segment_length)


def test_streaming_rejects_inconsistent_segment_batches() -> None:
    stream = StreamingDecoder(make_model(), segment_length=2)
    stream.process_segment(torch.tensor([[1, 2]]))

    with pytest.raises(ValueError, match="batch size"):
        stream.process_segment(torch.tensor([[1, 2], [3, 4]]))
