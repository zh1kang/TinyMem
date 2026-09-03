import pytest
import torch

from tinymem.memory.continuous import MeanPoolMemoryCompressor
from tinymem.memory.recurrent_memory import RecurrentMemoryBank
from tinymem.model.answerability import AnswerabilityHead
from tinymem.model.config import ModelConfig
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.memory_input import AttentionMemory
from tinymem.model.transformer import DecoderOnlyTransformer


def test_answerability_head_returns_one_logit_per_token() -> None:
    head = AnswerabilityHead(8, hidden_width=4)
    local_hidden = torch.randn(2, 3, 8)
    memory = AttentionMemory(
        values=torch.randn(2, 4, 8),
        valid=torch.tensor(
            [[False, False, True, True], [False, False, False, False]]
        ),
        positions=torch.tensor([[-1, -1, 2, 5], [-1, -1, -1, -1]]),
    )

    logits = head(local_hidden, memory)

    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()


def test_answerability_head_validates_feature_width() -> None:
    head = AnswerabilityHead(8)
    memory = AttentionMemory(
        values=torch.zeros(1, 2, 8),
        valid=torch.zeros(1, 2, dtype=torch.bool),
        positions=torch.full((1, 2), -1, dtype=torch.long),
    )

    with pytest.raises(ValueError, match="width"):
        head(torch.randn(1, 3, 4), memory)


def test_segmented_decoder_exposes_causal_answerability_logits() -> None:
    config = ModelConfig(
        vocab_size=16,
        d_model=8,
        n_layers=1,
        n_heads=2,
        d_ff=16,
        max_local_tokens=4,
        dropout=0.0,
    )
    decoder = SegmentedContinuousDecoder(
        DecoderOnlyTransformer(config),
        MeanPoolMemoryCompressor(8),
        RecurrentMemoryBank(capacity=2, model_width=8),
        segment_length=2,
        answerability_head=AnswerabilityHead(8),
    )
    input_ids = torch.tensor([[1, 2, 3, 4]])

    output = decoder(input_ids, torch.ones_like(input_ids, dtype=torch.bool))

    assert output.answerability_logits is not None
    assert output.answerability_logits.shape == (1, 4)
    output.answerability_logits[:, 2:].sum().backward()
    first_layer = decoder.answerability_head.projection[0]
    assert isinstance(first_layer, torch.nn.Linear)
    assert first_layer.weight.grad is not None
