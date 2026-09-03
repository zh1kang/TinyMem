import torch

from tinymem.data.correction_deletion import generate_update_examples
from tinymem.evaluation.answerability import evaluate_decoder_answerability
from tinymem.evaluation.continuous_memory import drop_memory
from tinymem.memory.continuous import MeanPoolMemoryCompressor
from tinymem.memory.recurrent_memory import RecurrentMemoryBank
from tinymem.model.answerability import AnswerabilityHead
from tinymem.model.config import ModelConfig
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.training.continuous import encode_update_with_write_targets
from tinymem.training.controlled_qa import build_qa_vocabulary


def test_decoder_answerability_evaluation_preserves_example_order() -> None:
    updates = generate_update_examples(
        split="validation",
        count=4,
        deletion_rate=0.5,
        correction_counts=(1,),
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
    )
    decoder = SegmentedContinuousDecoder(
        DecoderOnlyTransformer(config),
        MeanPoolMemoryCompressor(8),
        RecurrentMemoryBank(capacity=2, model_width=8),
        segment_length=8,
        answerability_head=AnswerabilityHead(8),
    )

    result = evaluate_decoder_answerability(
        decoder,
        encoded,
        batch_size=2,
        pad_id=vocabulary.token_to_id["<pad>"],
        device="cpu",
        intervention_name="drop",
        memory_intervention=drop_memory,
        corrupted_memory=True,
    )

    assert len(result.probabilities) == 4
    assert len(result.correctness) == 4
    assert result.answerable == (False, False, False, False)
    assert result.intervention == "drop"
    assert result.evaluation.points[1].total == 4
