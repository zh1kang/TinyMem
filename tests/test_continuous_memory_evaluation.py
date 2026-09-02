from dataclasses import replace

import pytest
import torch

from tinymem.data.babilong import parse_babilong_records
from tinymem.evaluation.continuous_memory import (
    calibrate_write_threshold,
    drop_memory,
    evaluate_continuous_answers,
    evaluate_continuous_qa1,
    paired_accuracy_test,
    shuffle_memory,
    zero_memory,
)
from tinymem.memory.continuous import (
    MeanPoolMemoryCompressor,
    MultiSlotAttentionMemoryCompressor,
)
from tinymem.memory.controller import AdaptiveWriteController
from tinymem.memory.recurrent_memory import (
    GatedRecurrentMemoryBank,
    RecurrentMemoryBank,
)
from tinymem.memory.write_gate import TokenSegmentWriteGate
from tinymem.model.config import ModelConfig
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.memory_input import AttentionMemory
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.training.controlled_qa import build_qa_vocabulary, encode_qa_example


def make_memory() -> AttentionMemory:
    return AttentionMemory(
        values=torch.arange(24, dtype=torch.float32).reshape(2, 3, 4),
        valid=torch.tensor([[False, True, True], [True, True, True]]),
        positions=torch.tensor([[-1, 2, 3], [0, 1, 3]]),
    )


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


def make_token_gated_decoder(vocab_size: int) -> SegmentedContinuousDecoder:
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
        MultiSlotAttentionMemoryCompressor(8, summary_slots=1),
        GatedRecurrentMemoryBank(capacity=2, model_width=8),
        segment_length=2,
        write_gate=TokenSegmentWriteGate(8),
    )


def make_adaptive_decoder(vocab_size: int) -> SegmentedContinuousDecoder:
    config = ModelConfig(
        vocab_size=vocab_size,
        d_model=8,
        n_layers=1,
        n_heads=2,
        d_ff=16,
        max_local_tokens=4,
    )
    decoder = SegmentedContinuousDecoder(
        DecoderOnlyTransformer(config),
        MultiSlotAttentionMemoryCompressor(8, summary_slots=1),
        GatedRecurrentMemoryBank(capacity=2, model_width=8),
        segment_length=2,
        write_controller=AdaptiveWriteController(8),
    )
    decoder.eval()
    return decoder


def test_memory_interventions_preserve_the_expected_controls() -> None:
    memory = make_memory()

    dropped = drop_memory(memory)
    zeroed = zero_memory(memory)
    shuffled = shuffle_memory(memory)

    assert not dropped.valid.any()
    assert (dropped.positions == -1).all()
    assert torch.equal(zeroed.values, torch.zeros_like(memory.values))
    assert torch.equal(zeroed.valid, memory.valid)
    assert torch.equal(zeroed.positions, memory.positions)
    assert torch.equal(shuffled.values[0], memory.values[1])
    assert torch.equal(shuffled.valid[0], memory.valid[1])
    assert torch.equal(shuffled.positions[0], memory.positions[1])


def test_shuffle_memory_rejects_one_batch_row() -> None:
    memory = make_memory()
    one_row = AttentionMemory(
        values=memory.values[:1],
        valid=memory.valid[:1],
        positions=memory.positions[:1],
    )

    with pytest.raises(ValueError, match="batch size"):
        shuffle_memory(one_row)


def test_continuous_evaluation_returns_exact_counts_and_curve() -> None:
    examples = parse_babilong_records(
        [
            {
                "input": "Mary moved to the kitchen.",
                "question": "Where is Mary? ",
                "target": "kitchen",
            },
            {
                "input": "John went to the office.",
                "question": "Where is John? ",
                "target": "office",
            },
        ],
        task_id="qa1",
        split="test",
        source_name="fixture.txt",
    )
    vocabulary = build_qa_vocabulary(examples)
    decoder = make_decoder(len(vocabulary))
    result = evaluate_continuous_qa1(
        decoder,
        vocabulary,
        examples,
        batch_size=2,
        device="cpu",
    )

    assert result.intervention == "normal"
    assert result.count == 2
    assert result.correct <= result.count
    assert sum(bucket.count for bucket in result.curve) == result.count
    assert result.to_dict()["count"] == 2
    assert result.writes.true_positive > 0
    assert result.writes.false_negative == 0


def test_paired_accuracy_test_requires_more_than_a_one_answer_margin() -> None:
    result = paired_accuracy_test(
        (True, False),
        (False, True),
        alpha=0.05,
    )

    assert result.accuracy_difference == 0.0
    assert result.one_sided_p_value == 0.75
    assert not result.significant


def test_paired_accuracy_test_accepts_consistent_normal_memory_wins() -> None:
    result = paired_accuracy_test(
        (True,) * 5,
        (False,) * 5,
        alpha=0.05,
    )

    assert result.accuracy_difference == 1.0
    assert result.one_sided_p_value == 0.03125
    assert result.significant


def test_encoded_validation_evaluation_handles_shuffled_singleton_remainder() -> None:
    examples = parse_babilong_records(
        [
            {
                "input": f"{person} moved to the {place}.",
                "question": f"Where is {person}? ",
                "target": place,
            }
            for person, place in (
                ("Mary", "kitchen"),
                ("John", "office"),
                ("Sandra", "garden"),
            )
        ],
        task_id="qa1",
        split="validation",
        source_name="fixture.txt",
    )
    vocabulary = build_qa_vocabulary(examples)
    decoder = make_decoder(len(vocabulary))

    result = evaluate_continuous_answers(
        decoder,
        [encode_qa_example(example, vocabulary) for example in examples],
        batch_size=2,
        pad_id=vocabulary.token_to_id["<pad>"],
        device="cpu",
        intervention_name="shuffle",
        memory_intervention=shuffle_memory,
    )

    assert result.intervention == "shuffle"
    assert result.count == 3
    assert result.correct <= result.count


def test_encoded_evaluation_applies_forced_write_policy() -> None:
    examples = parse_babilong_records(
        [
            {
                "input": "Mary moved to the kitchen.",
                "question": "Where is Mary? ",
                "target": "kitchen",
            },
            {
                "input": "John went to the office.",
                "question": "Where is John? ",
                "target": "office",
            },
        ],
        task_id="qa1",
        split="validation",
        source_name="fixture.txt",
    )
    vocabulary = build_qa_vocabulary(examples)
    encoded = [encode_qa_example(example, vocabulary) for example in examples]
    decoder = make_adaptive_decoder(len(vocabulary))
    max_segments = max(
        (len(example.input_ids) + decoder.segment_length - 1)
        // decoder.segment_length
        for example in encoded
    )
    forced = torch.zeros(len(encoded), max_segments, dtype=torch.bool)

    result = evaluate_continuous_answers(
        decoder,
        encoded,
        batch_size=2,
        pad_id=vocabulary.token_to_id["<pad>"],
        device="cpu",
        intervention_name="never_write",
        forced_writes=forced,
    )

    assert result.count == 2


def test_encoded_validation_evaluation_rejects_single_item_shuffle_batches() -> None:
    examples = parse_babilong_records(
        [
            {
                "input": "Mary moved to the kitchen.",
                "question": "Where is Mary? ",
                "target": "kitchen",
            },
            {
                "input": "John moved to the office.",
                "question": "Where is John? ",
                "target": "office",
            },
        ],
        task_id="qa1",
        split="validation",
        source_name="fixture.txt",
    )
    vocabulary = build_qa_vocabulary(examples)
    decoder = make_decoder(len(vocabulary))

    with pytest.raises(ValueError, match="batch size above one"):
        evaluate_continuous_answers(
            decoder,
            [encode_qa_example(example, vocabulary) for example in examples],
            batch_size=1,
            pad_id=vocabulary.token_to_id["<pad>"],
            device="cpu",
            memory_intervention=shuffle_memory,
        )


def test_encoded_validation_evaluation_counts_write_decisions() -> None:
    examples = parse_babilong_records(
        [
            {
                "input": "Mary moved to the kitchen.",
                "question": "Where is Mary? ",
                "target": "kitchen",
            },
            {
                "input": "John moved to the office.",
                "question": "Where is John? ",
                "target": "office",
            },
        ],
        task_id="qa1",
        split="validation",
        source_name="fixture.txt",
    )
    vocabulary = build_qa_vocabulary(examples)
    encoded = []
    expected_segments = 0
    for example in examples:
        item = encode_qa_example(example, vocabulary)
        segment_count = (len(item.input_ids) + 1) // 2
        targets = (True, *(False for _ in range(segment_count - 1)))
        encoded.append(replace(item, segment_write_targets=targets))
        expected_segments += segment_count

    result = evaluate_continuous_answers(
        make_decoder(len(vocabulary)),
        encoded,
        batch_size=2,
        pad_id=vocabulary.token_to_id["<pad>"],
        device="cpu",
    )

    assert result.writes is not None
    assert result.writes.true_positive == 2
    assert result.writes.false_positive == expected_segments - 2
    assert result.writes.false_negative == 0


def test_write_threshold_calibration_enforces_zero_false_positives() -> None:
    examples = parse_babilong_records(
        [
            {
                "input": "Mary moved to the kitchen.",
                "question": "Where is Mary? ",
                "target": "kitchen",
            },
            {
                "input": "John moved to the office.",
                "question": "Where is John? ",
                "target": "office",
            },
        ],
        task_id="qa1",
        split="validation",
        source_name="fixture.txt",
    )
    vocabulary = build_qa_vocabulary(examples)
    encoded = []
    for example in examples:
        item = encode_qa_example(example, vocabulary)
        segment_count = (len(item.input_ids) + 1) // 2
        encoded.append(
            replace(
                item,
                segment_write_targets=(True,)
                + (False,) * (segment_count - 1),
            )
        )

    decoder = make_token_gated_decoder(len(vocabulary))
    result = calibrate_write_threshold(
        decoder,
        encoded,
        batch_size=2,
        pad_id=vocabulary.token_to_id["<pad>"],
        device="cpu",
        max_false_positive_rate=0.0,
    )
    decoder.bank.set_write_threshold(result.threshold)
    evaluation = evaluate_continuous_answers(
        decoder,
        encoded,
        batch_size=2,
        pad_id=vocabulary.token_to_id["<pad>"],
        device="cpu",
    )

    assert 0.5 <= result.threshold <= 1.0
    assert result.allowed_false_positives == 0
    assert result.writes.false_positive == 0
    assert evaluation.writes is not None
    assert evaluation.writes.false_positive == 0
