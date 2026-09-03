import torch

from tinymem.data.correction_deletion import generate_update_examples
from tinymem.memory.continuous import MeanPoolMemoryCompressor
from tinymem.memory.recurrent_memory import RecurrentMemoryBank
from tinymem.model.answerability import AnswerabilityHead
from tinymem.model.config import ModelConfig
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.training.continuous import (
    encode_update_with_write_targets,
    train_continuous_answer_supervision,
)
from tinymem.training.controlled_qa import build_qa_vocabulary


def test_update_training_optimizes_answerability_head() -> None:
    updates = generate_update_examples(
        split="train",
        count=4,
        deletion_rate=0.5,
        correction_counts=(0, 1),
        query_delay=1,
        distractor_count=1,
    )
    examples = [item.example for item in updates]
    vocabulary = build_qa_vocabulary(examples)
    encoded = [
        encode_update_with_write_targets(
            example,
            vocabulary,
            segment_length=8,
        )
        for example in examples
    ]
    config = ModelConfig(
        vocab_size=len(vocabulary),
        d_model=8,
        n_layers=1,
        n_heads=2,
        d_ff=16,
        max_local_tokens=8,
        dropout=0.0,
    )
    decoder = SegmentedContinuousDecoder(
        DecoderOnlyTransformer(config),
        MeanPoolMemoryCompressor(8),
        RecurrentMemoryBank(capacity=2, model_width=8),
        segment_length=8,
        answerability_head=AnswerabilityHead(8),
    )
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=0.01)
    before = decoder.answerability_head.projection[-1].weight.detach().clone()

    losses = train_continuous_answer_supervision(
        decoder,
        optimizer,
        encoded,
        steps=2,
        batch_size=4,
        gradient_clip_norm=1.0,
        pad_id=vocabulary.token_to_id["<pad>"],
        device="cpu",
        seed=7,
        answerability_loss_weight=1.0,
    )

    assert len(losses) == 2
    assert all(torch.isfinite(torch.tensor(losses)))
    assert not torch.equal(
        before,
        decoder.answerability_head.projection[-1].weight,
    )
