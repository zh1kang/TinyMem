import torch

from tinymem.data.babi import parse_babi_lines
from tinymem.evaluation.multi_token_prediction import (
    evaluate_base_mtp_loss,
    evaluate_continuous_mtp_loss,
)
from tinymem.memory.continuous import MeanPoolMemoryCompressor
from tinymem.memory.recurrent_memory import RecurrentMemoryBank
from tinymem.model.config import ModelConfig
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.multi_token_prediction import MultiTokenPredictionHeads
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.training.controlled_qa import build_qa_vocabulary, encode_qa_examples


def test_mtp_diagnostics_cover_base_and_segmented_paths() -> None:
    examples = parse_babi_lines(
        [
            "1 Mary moved to the kitchen.\n",
            "2 John went to the office.\n",
            "3 Where is Mary?\tkitchen\t1\n",
        ],
        task_id="qa1",
        split="validation",
        source_name="fixture.txt",
    )
    vocabulary = build_qa_vocabulary(examples)
    encoded, _ = encode_qa_examples(examples, vocabulary, max_tokens=64)
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
    decoder = SegmentedContinuousDecoder(
        model,
        MeanPoolMemoryCompressor(8),
        RecurrentMemoryBank(capacity=2, model_width=8),
        segment_length=4,
        mtp_heads=heads,
    )

    base = evaluate_base_mtp_loss(
        model,
        heads,
        encoded,
        batch_size=1,
        pad_id=vocabulary.token_to_id["<pad>"],
        device="cpu",
    )
    segmented = evaluate_continuous_mtp_loss(
        decoder,
        encoded,
        batch_size=1,
        pad_id=vocabulary.token_to_id["<pad>"],
        device="cpu",
    )

    assert base.keys() == segmented.keys() == {2, 3, 4}
    assert all(torch.isfinite(torch.tensor(value)) for value in base.values())
    assert all(torch.isfinite(torch.tensor(value)) for value in segmented.values())
