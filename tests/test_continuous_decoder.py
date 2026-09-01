import pytest
import torch

from tinymem.memory.continuous import (
    AttentionPoolMemoryCompressor,
    ContinuousMemoryCompressor,
    MeanPoolMemoryCompressor,
)
from tinymem.memory.recurrent_memory import RecurrentMemoryBank
from tinymem.model.config import ModelConfig
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.kv_cache import KVCache
from tinymem.model.transformer import DecoderOnlyTransformer


def make_decoder(
    *,
    segment_length: int = 2,
    capacity: int = 2,
    vocab_size: int = 16,
    compressor_type: type[ContinuousMemoryCompressor] = MeanPoolMemoryCompressor,
) -> SegmentedContinuousDecoder:
    config = ModelConfig(
        vocab_size=vocab_size,
        d_model=8,
        n_layers=1,
        n_heads=2,
        d_ff=16,
        max_local_tokens=4,
        dropout=0.0,
    )
    model = DecoderOnlyTransformer(config)
    compressor = compressor_type(config.d_model)
    bank = RecurrentMemoryBank(
        capacity=capacity,
        model_width=config.d_model,
    )
    return SegmentedContinuousDecoder(
        model,
        compressor,
        bank,
        segment_length=segment_length,
    )


def test_segmented_decoder_returns_logits_and_fixed_memory() -> None:
    decoder = make_decoder(segment_length=2, capacity=2)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    token_valid = torch.ones_like(input_ids, dtype=torch.bool)

    output = decoder(input_ids, token_valid)

    assert output.logits.shape == (1, 4, 16)
    assert output.memory.shape == (1, 2, 8)
    assert output.memory_valid.shape == (1, 2)
    assert output.memory_positions.shape == (1, 2)
    assert torch.equal(output.memory_valid, torch.tensor([[True, True]]))
    assert torch.equal(output.memory_positions, torch.tensor([[1, 3]]))


def test_segmented_decoder_accepts_attention_pooling() -> None:
    decoder = make_decoder(
        compressor_type=AttentionPoolMemoryCompressor,
    )
    input_ids = torch.tensor([[1, 2, 3, 4]])
    token_valid = torch.ones_like(input_ids, dtype=torch.bool)

    output = decoder(input_ids, token_valid)

    assert output.logits.shape == (1, 4, 16)
    assert output.memory_valid.all()


def test_first_segment_logits_use_only_initial_empty_memory() -> None:
    torch.manual_seed(7)
    decoder = make_decoder(segment_length=2)
    decoder.eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    token_valid = torch.ones_like(input_ids, dtype=torch.bool)
    caches = [
        KVCache(max_length=2, start_position=0)
        for _ in range(decoder.model.config.n_layers)
    ]
    expected_hidden = decoder.model.forward_hidden(
        input_ids[:, :2],
        caches=caches,
    )
    expected_logits = decoder.model.lm_head(expected_hidden)

    output = decoder(input_ids, token_valid)

    torch.testing.assert_close(output.logits[:, :2], expected_logits)


def test_invalid_segment_does_not_replace_existing_memory() -> None:
    decoder = make_decoder(segment_length=2, capacity=2)
    input_ids = torch.tensor([[1, 2, 0, 0]])
    token_valid = torch.tensor([[True, True, False, False]])

    output = decoder(input_ids, token_valid)

    assert torch.equal(output.memory_valid, torch.tensor([[False, True]]))
    assert torch.equal(output.memory_positions, torch.tensor([[-1, 1]]))


def test_later_segment_loss_reaches_the_compressor() -> None:
    torch.manual_seed(11)
    decoder = make_decoder(segment_length=2)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    token_valid = torch.ones_like(input_ids, dtype=torch.bool)

    output = decoder(input_ids, token_valid)
    loss = output.logits[:, 2:].square().mean()
    loss.backward()

    gradient = decoder.compressor.projection.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_later_tokens_cannot_change_earlier_segment_logits() -> None:
    torch.manual_seed(13)
    decoder = make_decoder(segment_length=2)
    decoder.eval()
    valid = torch.ones(1, 4, dtype=torch.bool)

    original = decoder(torch.tensor([[1, 2, 3, 4]]), valid)
    changed = decoder(torch.tensor([[1, 2, 9, 10]]), valid)

    torch.testing.assert_close(original.logits[:, :2], changed.logits[:, :2])


def test_segmented_decoder_applies_memory_intervention_to_every_segment() -> None:
    decoder = make_decoder(segment_length=2)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    token_valid = torch.ones_like(input_ids, dtype=torch.bool)
    observed_valid = []

    def observe(memory: object) -> object:
        observed_valid.append(memory.valid.clone())
        return memory

    decoder(
        input_ids,
        token_valid,
        memory_intervention=observe,
    )

    assert len(observed_valid) == 2
    assert not observed_valid[0].any()
    assert observed_valid[1].any()


def test_segmented_decoder_rejects_invalid_memory_intervention() -> None:
    decoder = make_decoder()
    input_ids = torch.tensor([[1, 2]])
    token_valid = torch.ones_like(input_ids, dtype=torch.bool)

    with pytest.raises(TypeError, match="memory_intervention"):
        decoder(input_ids, token_valid, memory_intervention="zero")

    with pytest.raises(TypeError, match="return AttentionMemory"):
        decoder(
            input_ids,
            token_valid,
            memory_intervention=lambda memory: memory.values,
        )


@pytest.mark.parametrize(
    ("input_ids", "token_valid", "error"),
    [
        ([[1, 2]], torch.tensor([[True, True]]), TypeError),
        (
            torch.ones(1, 2, 1, dtype=torch.long),
            torch.ones(1, 2, 1, dtype=torch.bool),
            ValueError,
        ),
        (
            torch.empty(1, 0, dtype=torch.long),
            torch.empty(1, 0, dtype=torch.bool),
            ValueError,
        ),
        (torch.ones(1, 2), torch.ones(1, 2, dtype=torch.bool), TypeError),
        (torch.ones(1, 2, dtype=torch.long), [[True, True]], TypeError),
        (
            torch.ones(1, 2, dtype=torch.long),
            torch.ones(1, 3, dtype=torch.bool),
            ValueError,
        ),
        (torch.ones(1, 2, dtype=torch.long), torch.ones(1, 2), TypeError),
        (
            torch.ones(1, 3, dtype=torch.long),
            torch.tensor([[True, False, True]]),
            ValueError,
        ),
    ],
)
def test_segmented_decoder_rejects_invalid_inputs(
    input_ids: object,
    token_valid: object,
    error: type[Exception],
) -> None:
    decoder = make_decoder()

    with pytest.raises(error):
        decoder(input_ids, token_valid)


@pytest.mark.parametrize("segment_length", [True, 0, 5, 2.5])
def test_segmented_decoder_rejects_invalid_segment_length(
    segment_length: object,
) -> None:
    error = (
        ValueError
        if isinstance(segment_length, int) and not isinstance(segment_length, bool)
        else TypeError
    )

    with pytest.raises(error):
        make_decoder(segment_length=segment_length)
