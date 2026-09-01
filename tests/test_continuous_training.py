import torch

from tinymem.data.babi import parse_babi_lines
from tinymem.data.vocabulary import ControlledVocabulary
from tinymem.memory.continuous import MeanPoolMemoryCompressor
from tinymem.memory.recurrent_memory import RecurrentMemoryBank
from tinymem.model.config import ModelConfig
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.training.continuous import (
    collate_segmented_answer_supervision,
    train_continuous_answer_supervision,
)
from tinymem.training.controlled_qa import (
    EncodedQAExample,
    build_qa_vocabulary,
    encode_qa_example,
)


def make_examples() -> tuple[ControlledVocabulary, list[EncodedQAExample]]:
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
    return vocabulary, [
        encode_qa_example(example, vocabulary)
        for example in examples
    ]


def make_decoder(vocab_size: int) -> SegmentedContinuousDecoder:
    config = ModelConfig(
        vocab_size=vocab_size,
        d_model=8,
        n_layers=1,
        n_heads=2,
        d_ff=16,
        max_local_tokens=4,
    )
    return SegmentedContinuousDecoder(
        DecoderOnlyTransformer(config),
        MeanPoolMemoryCompressor(config.d_model),
        RecurrentMemoryBank(capacity=2, model_width=config.d_model),
        segment_length=2,
    )


def test_segmented_collation_returns_right_padding_mask() -> None:
    vocabulary, examples = make_examples()

    input_ids, target_ids, token_valid = collate_segmented_answer_supervision(
        examples,
        pad_id=vocabulary.token_to_id["<pad>"],
        device="cpu",
    )

    assert input_ids.shape == target_ids.shape == token_valid.shape
    assert token_valid.dtype == torch.bool
    for row, example in enumerate(examples):
        length = len(example.input_ids)
        assert token_valid[row, :length].all()
        assert not token_valid[row, length:].any()


def test_continuous_training_updates_the_compressor() -> None:
    torch.manual_seed(23)
    vocabulary, examples = make_examples()
    decoder = make_decoder(len(vocabulary))
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=0.01)
    before = decoder.compressor.projection.weight.detach().clone()

    losses = train_continuous_answer_supervision(
        decoder,
        optimizer,
        examples,
        steps=3,
        batch_size=2,
        gradient_clip_norm=1.0,
        pad_id=vocabulary.token_to_id["<pad>"],
        device="cpu",
        seed=5,
    )

    assert len(losses) == 3
    assert all(torch.isfinite(torch.tensor(losses)))
    assert not torch.equal(decoder.compressor.projection.weight, before)
