import math

import torch

from tinymem.evaluation.continuous_memory import drop_memory
from tinymem.memory.continuous import MeanPoolMemoryCompressor
from tinymem.memory.recurrent_memory import RecurrentMemoryBank
from tinymem.model.config import ModelConfig
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.training.language_model import (
    evaluate_segmented_language_model,
    sample_language_model_batch,
    train_segmented_language_model,
)


def make_decoder() -> SegmentedContinuousDecoder:
    config = ModelConfig(
        vocab_size=32,
        d_model=8,
        n_layers=1,
        n_heads=2,
        d_ff=16,
        max_local_tokens=4,
        dropout=0.0,
    )
    return SegmentedContinuousDecoder(
        DecoderOnlyTransformer(config),
        MeanPoolMemoryCompressor(8),
        RecurrentMemoryBank(capacity=2, model_width=8),
        segment_length=4,
    )


def test_language_model_batch_sampling_is_seeded() -> None:
    stream = torch.arange(24)
    first = sample_language_model_batch(
        stream,
        sequence_length=6,
        batch_size=3,
        generator=torch.Generator().manual_seed(7),
        device="cpu",
        vocab_size=32,
    )
    second = sample_language_model_batch(
        stream,
        sequence_length=6,
        batch_size=3,
        generator=torch.Generator().manual_seed(7),
        device="cpu",
        vocab_size=32,
    )

    assert first.shape == (3, 6)
    assert torch.equal(first, second)
    assert torch.equal(first[:, 1:] - first[:, :-1], torch.ones(3, 5))


def test_segmented_language_model_training_updates_parameters() -> None:
    decoder = make_decoder()
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=0.01)
    before = decoder.model.lm_head.weight.detach().clone()

    losses = train_segmented_language_model(
        decoder,
        optimizer,
        torch.arange(128) % 32,
        steps=2,
        batch_size=2,
        sequence_length=12,
        gradient_clip_norm=1.0,
        device="cpu",
        seed=3,
    )

    assert len(losses) == 2
    assert all(math.isfinite(loss) for loss in losses)
    assert not torch.equal(before, decoder.model.lm_head.weight)


def test_language_model_evaluation_reports_weighted_perplexity() -> None:
    decoder = make_decoder()
    stream = torch.arange(30) % 32

    normal = evaluate_segmented_language_model(
        decoder,
        stream,
        sequence_length=10,
        batch_size=2,
        pad_id=31,
        device="cpu",
    )
    ablated = evaluate_segmented_language_model(
        decoder,
        stream,
        sequence_length=10,
        batch_size=2,
        pad_id=31,
        device="cpu",
        final_segment_only=True,
        memory_intervention=drop_memory,
    )

    assert normal.predicted_tokens == 29
    assert normal.perplexity == math.exp(normal.loss)
    assert 0 < ablated.predicted_tokens < normal.predicted_tokens
    assert math.isfinite(ablated.loss)
