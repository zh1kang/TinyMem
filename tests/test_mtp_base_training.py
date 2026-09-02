import pytest
import torch

from tinymem.data.babi import parse_babi_lines
from tinymem.model.config import ModelConfig
from tinymem.model.multi_token_prediction import MultiTokenPredictionHeads
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.training.controlled_qa import (
    EncodedQAExample,
    build_qa_vocabulary,
    encode_qa_examples,
    train_answer_supervision,
)


def test_base_answer_training_updates_mtp_heads() -> None:
    examples = parse_babi_lines(
        [
            "1 Mary moved to the kitchen.\n",
            "2 John went to the office.\n",
            "3 Where is Mary?\tkitchen\t1\n",
            "1 John went to the office.\n",
            "2 Mary moved to the garden.\n",
            "3 Where is John?\toffice\t1\n",
        ],
        task_id="qa1",
        split="train",
        source_name="fixture.txt",
    )
    vocabulary = build_qa_vocabulary(examples)
    encoded, skipped = encode_qa_examples(examples, vocabulary, max_tokens=64)
    config = ModelConfig(
        vocab_size=len(vocabulary),
        d_model=8,
        n_layers=1,
        n_heads=2,
        d_ff=16,
        max_local_tokens=64,
    )
    model = DecoderOnlyTransformer(config)
    heads = MultiTokenPredictionHeads(8, len(vocabulary), (2, 3, 4))
    optimizer = torch.optim.AdamW(
        (*model.parameters(), *heads.parameters()),
        lr=0.01,
    )
    before = heads.heads[0].weight.detach().clone()

    losses = train_answer_supervision(
        model,
        optimizer,
        encoded,
        steps=3,
        batch_size=2,
        gradient_clip_norm=1.0,
        pad_id=vocabulary.token_to_id["<pad>"],
        device="cpu",
        seed=9,
        mtp_heads=heads,
        mtp_loss_weight=0.2,
    )

    assert skipped == 0
    assert len(losses) == 3
    assert not torch.equal(heads.heads[0].weight, before)


def test_base_mtp_training_rejects_short_examples_before_sampling() -> None:
    config = ModelConfig(
        vocab_size=8,
        d_model=8,
        n_layers=1,
        n_heads=2,
        d_ff=16,
        max_local_tokens=8,
    )
    model = DecoderOnlyTransformer(config)
    heads = MultiTokenPredictionHeads(8, 8, (2, 3, 4))
    optimizer = torch.optim.AdamW((*model.parameters(), *heads.parameters()))
    example = EncodedQAExample(
        input_ids=(1, 2, 3, 4),
        answer_id=4,
        source_example_id="short",
    )

    with pytest.raises(ValueError, match="longer than the maximum horizon"):
        train_answer_supervision(
            model,
            optimizer,
            [example],
            steps=1,
            batch_size=1,
            gradient_clip_norm=1.0,
            pad_id=0,
            device="cpu",
            seed=1,
            mtp_heads=heads,
            mtp_loss_weight=0.2,
        )
