import torch

from tinymem.data.babi import parse_babi_lines
from tinymem.model.config import ModelConfig
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.training.controlled_qa import (
    answer_cross_entropy,
    build_qa_vocabulary,
    collate_answer_supervision,
    encode_qa_example,
)


def test_answer_batch_supervises_only_the_answer_token() -> None:
    examples = parse_babi_lines(
        [
            "1 Mary moved to the kitchen.\n",
            "2 Where is Mary?\tkitchen\t1\n",
            "1 John went to the office.\n",
            "2 Where is John?\toffice\t1\n",
        ],
        task_id="qa1",
        split="train",
        source_name="fixture.txt",
    )
    vocabulary = build_qa_vocabulary(examples)
    encoded = [encode_qa_example(example, vocabulary) for example in examples]

    input_ids, target_ids = collate_answer_supervision(
        encoded,
        pad_id=vocabulary.token_to_id["<pad>"],
        device="cpu",
    )

    assert input_ids.shape == target_ids.shape
    assert (target_ids != -100).sum().item() == len(encoded)
    for row, example in enumerate(encoded):
        answer_position = len(example.input_ids) - 1
        assert target_ids[row, answer_position].item() == example.answer_id
        assert torch.equal(
            target_ids[row, :answer_position],
            torch.full((answer_position,), -100),
        )


def test_answer_cross_entropy_returns_finite_mean_and_restores_mode() -> None:
    examples = parse_babi_lines(
        [
            "1 Mary moved to the kitchen.\n",
            "2 Where is Mary?\tkitchen\t1\n",
            "1 John went to the office.\n",
            "2 Where is John?\toffice\t1\n",
        ],
        task_id="qa1",
        split="validation",
        source_name="fixture.txt",
    )
    vocabulary = build_qa_vocabulary(examples)
    encoded = [encode_qa_example(example, vocabulary) for example in examples]
    model = DecoderOnlyTransformer(
        ModelConfig(
            vocab_size=len(vocabulary),
            d_model=8,
            n_layers=1,
            n_heads=2,
            d_ff=16,
            max_local_tokens=32,
        )
    ).train()

    loss = answer_cross_entropy(
        model,
        encoded,
        batch_size=1,
        pad_id=vocabulary.token_to_id["<pad>"],
        device="cpu",
    )

    assert loss > 0
    assert torch.isfinite(torch.tensor(loss))
    assert model.training
