import torch

from tinymem.data.babi import parse_babi_lines
from tinymem.data.memory_required_qa import build_memory_required_qa_examples
from tinymem.evaluation.memory_required_qa import (
    evaluate_memory_required_qa,
    evaluate_memory_required_write_gate,
)
from tinymem.memory.continuous import MeanPoolMemoryCompressor
from tinymem.memory.recurrent_memory import (
    GatedRecurrentMemoryBank,
    RecurrentMemoryBank,
)
from tinymem.memory.write_gate import TokenSegmentWriteGate
from tinymem.model.config import ModelConfig
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.tokenization.byte_tokenizer import ByteTokenizer
from tinymem.training.memory_required_qa import (
    audit_cross_segment_gradients,
    collate_memory_required_distractor,
    collate_memory_required_prefix_segment,
    collate_memory_required_query,
    collate_memory_required_support,
    memory_required_write_gate_loss,
    precompute_oracle_memories,
    train_memory_required_qa,
    unroll_memory_required_prefix,
)


def make_examples(
    distractor_segments: int = 0,
    distractor_variant: str = "trained",
    support_position_mode: str = "first",
):
    source = parse_babi_lines(
        [
            "1 Mary moved to the kitchen.\n",
            "2 Where is Mary?\tkitchen\t1\n",
            "1 John went to the office.\n",
            "2 Where is John?\toffice\t1\n",
        ],
        task_id="qa1",
        split="train",
        source_name="memory-required.txt",
    )
    return build_memory_required_qa_examples(
        source,
        ByteTokenizer(),
        segment_length=64,
        distractor_segments=distractor_segments,
        distractor_variant=distractor_variant,
        support_position_mode=support_position_mode,
    )


def make_decoder(
    memory_capacity: int = 1,
    *,
    gated: bool = False,
) -> SegmentedContinuousDecoder:
    config = ModelConfig(
        vocab_size=ByteTokenizer.vocab_size,
        d_model=8,
        n_layers=1,
        n_heads=2,
        d_ff=16,
        max_local_tokens=64,
        dropout=0.0,
    )
    bank = (
        GatedRecurrentMemoryBank(
            capacity=memory_capacity,
            model_width=config.d_model,
        )
        if gated
        else RecurrentMemoryBank(
            capacity=memory_capacity,
            model_width=config.d_model,
        )
    )
    return SegmentedContinuousDecoder(
        DecoderOnlyTransformer(config),
        MeanPoolMemoryCompressor(config.d_model),
        bank,
        segment_length=64,
        write_gate=(TokenSegmentWriteGate(config.d_model) if gated else None),
        memory_position_mode="virtual",
    )


def test_memory_required_examples_place_only_support_in_first_segment() -> None:
    examples = make_examples()

    for example in examples:
        assert example.segment_length == 64
        assert len(example.support_ids) < example.segment_length
        assert example.query_ids[0] == ord("Q")
        assert len(example.query_ids) + len(example.answer_ids) + 1 <= 64


def test_memory_required_examples_place_distractors_before_query() -> None:
    examples = make_examples(distractor_segments=2)

    for example in examples:
        assert len(example.distractor_ids) == 2
        assert all(len(distractor) <= 64 for distractor in example.distractor_ids)
        assert example.query_position_offset == 3 * 64


def test_heldout_distractors_change_only_intervening_content() -> None:
    trained = make_examples(distractor_segments=2, distractor_variant="trained")
    heldout = make_examples(distractor_segments=2, distractor_variant="heldout")

    for trained_example, heldout_example in zip(trained, heldout, strict=True):
        assert trained_example.support_ids == heldout_example.support_ids
        assert trained_example.query_ids == heldout_example.query_ids
        assert trained_example.answer_ids == heldout_example.answer_ids
        assert trained_example.distractor_ids != heldout_example.distractor_ids
        assert heldout_example.distractor_variant == "heldout"


def test_matched_distractors_use_other_babi_facts_and_cycle_support() -> None:
    examples = make_examples(
        distractor_segments=2,
        distractor_variant="matched",
        support_position_mode="cycled",
    )
    tokenizer = ByteTokenizer()

    assert [example.support_segment_index for example in examples] == [0, 1]
    for example in examples:
        support_subject = tokenizer.decode(example.support_ids).split()[1]
        assert example.prefix_ids[example.support_segment_index] == example.support_ids
        for distractor in example.distractor_ids:
            distractor_text = tokenizer.decode(distractor)
            assert distractor_text.startswith("Fact: ")
            assert distractor_text.split()[1] != support_subject


def test_cycled_prefix_unroll_preserves_physical_slot_positions() -> None:
    torch.manual_seed(43)
    examples = make_examples(
        distractor_segments=2,
        distractor_variant="matched",
        support_position_mode="cycled",
    )
    decoder = make_decoder(memory_capacity=3).eval()

    memory = unroll_memory_required_prefix(
        decoder,
        examples,
        pad_id=ByteTokenizer.special_tokens["<pad>"],
        device="cpu",
    )

    assert memory.valid.all()
    for row, example in enumerate(examples):
        assert memory.positions[row].tolist() == [
            index * 64 + len(segment) - 1
            for index, segment in enumerate(example.prefix_ids)
        ]
    segment_ids, segment_valid = collate_memory_required_prefix_segment(
        examples,
        1,
        pad_id=ByteTokenizer.special_tokens["<pad>"],
        device="cpu",
    )
    assert segment_ids.shape == segment_valid.shape
    assert segment_valid.sum(dim=1).tolist() == [
        len(example.prefix_ids[1]) for example in examples
    ]


def test_memory_required_collation_supervises_only_the_answer() -> None:
    examples = make_examples()
    input_ids, target_ids, valid = collate_memory_required_query(
        examples,
        pad_id=ByteTokenizer.special_tokens["<pad>"],
        device="cpu",
    )

    answer_start = len(examples[0].query_ids)
    assert (target_ids[0, :answer_start] == -100).all()
    assert target_ids[0, answer_start : answer_start + 7].tolist() == list(
        examples[0].answer_ids
    )
    assert valid[0, : answer_start + 8].all()
    assert input_ids.shape == target_ids.shape == valid.shape


def test_support_padding_is_invalid_and_does_not_change_memory() -> None:
    torch.manual_seed(2)
    examples = make_examples()
    decoder = make_decoder().eval()
    support_ids, support_valid = collate_memory_required_support(
        examples,
        pad_id=ByteTokenizer.special_tokens["<pad>"],
        device="cpu",
    )

    batched = decoder(support_ids, support_valid)

    assert support_valid.sum(dim=1).tolist() == [
        len(example.support_ids) for example in examples
    ]
    for row, example in enumerate(examples):
        individual_ids = torch.tensor(example.support_ids).unsqueeze(0)
        individual = decoder(
            individual_ids,
            torch.ones_like(individual_ids, dtype=torch.bool),
        )
        torch.testing.assert_close(
            batched.memory[row : row + 1],
            individual.memory,
        )


def test_distractors_unroll_into_separate_valid_memory_slots() -> None:
    torch.manual_seed(17)
    examples = make_examples(distractor_segments=2)
    decoder = make_decoder(memory_capacity=3).eval()

    memory = unroll_memory_required_prefix(
        decoder,
        examples,
        pad_id=ByteTokenizer.special_tokens["<pad>"],
        device="cpu",
    )

    assert memory.valid.all()
    for row, example in enumerate(examples):
        assert memory.positions[row].tolist() == [
            len(example.support_ids) - 1,
            64 + len(example.distractor_ids[0]) - 1,
            128 + len(example.distractor_ids[1]) - 1,
        ]
    distractor_ids, distractor_valid = collate_memory_required_distractor(
        examples,
        1,
        pad_id=ByteTokenizer.special_tokens["<pad>"],
        device="cpu",
    )
    assert distractor_ids.shape == distractor_valid.shape
    assert distractor_valid.sum(dim=1).tolist() == [
        len(example.distractor_ids[1]) for example in examples
    ]


def test_capacity_pressure_evicts_the_oldest_support_summary() -> None:
    torch.manual_seed(19)
    examples = make_examples(distractor_segments=2)
    decoder = make_decoder(memory_capacity=2).eval()

    memory = unroll_memory_required_prefix(
        decoder,
        examples,
        pad_id=ByteTokenizer.special_tokens["<pad>"],
        device="cpu",
    )

    assert memory.valid.all()
    for row, example in enumerate(examples):
        assert memory.positions[row].tolist() == [
            64 + len(example.distractor_ids[0]) - 1,
            128 + len(example.distractor_ids[1]) - 1,
        ]


def test_skipping_distractor_writes_preserves_one_support_slot() -> None:
    torch.manual_seed(23)
    examples = make_examples(distractor_segments=2)
    decoder = make_decoder(memory_capacity=1).eval()

    memory = unroll_memory_required_prefix(
        decoder,
        examples,
        pad_id=ByteTokenizer.special_tokens["<pad>"],
        device="cpu",
        write_distractors=False,
    )

    assert memory.valid.all()
    assert memory.positions[:, 0].tolist() == [
        len(example.support_ids) - 1 for example in examples
    ]


def test_oracle_reader_training_uses_precomputed_support_slots() -> None:
    torch.manual_seed(3)
    examples = make_examples()
    decoder = make_decoder()
    oracle_values = precompute_oracle_memories(
        decoder,
        examples,
        device="cpu",
    )
    optimizer = torch.optim.AdamW(decoder.model.parameters(), lr=0.01)

    history = train_memory_required_qa(
        decoder,
        optimizer,
        examples,
        mode="oracle",
        steps=2,
        batch_size=2,
        gradient_clip_norm=1.0,
        pad_id=ByteTokenizer.special_tokens["<pad>"],
        device="cpu",
        seed=11,
        oracle_values=oracle_values,
    )

    assert oracle_values.shape == (2, 1, 8)
    assert len(history.losses) == 2
    assert all(torch.isfinite(torch.tensor(history.losses)))


def test_write_gate_loss_supervises_support_and_distractor_decisions() -> None:
    torch.manual_seed(29)
    examples = make_examples(distractor_segments=2)
    decoder = make_decoder(gated=True)

    loss = memory_required_write_gate_loss(
        decoder,
        examples,
        pad_id=ByteTokenizer.special_tokens["<pad>"],
        device="cpu",
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert decoder.write_gate is not None
    assert decoder.write_gate.score.weight.grad is not None
    assert decoder.write_gate.score.weight.grad.norm() > 0


def test_write_gate_evaluation_reports_hard_decision_quality() -> None:
    torch.manual_seed(31)
    examples = make_examples(distractor_segments=2)
    decoder = make_decoder(gated=True)

    result = evaluate_memory_required_write_gate(
        decoder,
        examples,
        device="cpu",
    )

    assert result.segment_count == 6
    assert result.true_positive == 2
    assert result.false_positive == 4
    assert result.false_negative == 0
    assert result.true_negative == 0
    assert result.recall == 1.0
    assert result.false_positive_rate == 1.0


def test_gated_training_records_answer_and_write_losses() -> None:
    torch.manual_seed(37)
    examples = make_examples(distractor_segments=2)
    decoder = make_decoder(gated=True)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=0.01)

    history = train_memory_required_qa(
        decoder,
        optimizer,
        examples,
        mode="gated",
        steps=2,
        batch_size=2,
        gradient_clip_norm=1.0,
        pad_id=ByteTokenizer.special_tokens["<pad>"],
        device="cpu",
        seed=41,
        write_loss_weight=1.0,
    )

    assert len(history.losses) == 2
    assert len(history.answer_losses) == 2
    assert len(history.write_losses) == 2
    assert all(torch.isfinite(torch.tensor(history.losses)))


def test_task_gradient_reaches_writer_only_when_memory_is_read() -> None:
    torch.manual_seed(5)
    decoder = make_decoder(memory_capacity=3)
    example = make_examples(distractor_segments=2)[0]

    normal = audit_cross_segment_gradients(
        decoder,
        example,
        condition="normal",
        pad_id=ByteTokenizer.special_tokens["<pad>"],
        device="cpu",
    )
    dropped = audit_cross_segment_gradients(
        decoder,
        example,
        condition="drop",
        pad_id=ByteTokenizer.special_tokens["<pad>"],
        device="cpu",
    )

    assert normal.support_hidden_grad_norm > 0
    assert normal.summary_grad_norm > 0
    assert normal.writer_grad_norm > 0
    assert dropped.support_hidden_grad_norm == 0
    assert dropped.summary_grad_norm == 0
    assert dropped.writer_grad_norm == 0


def test_memory_required_evaluation_runs_all_causal_conditions() -> None:
    torch.manual_seed(13)
    decoder = make_decoder(memory_capacity=3)
    examples = make_examples(distractor_segments=2)
    oracle_values = precompute_oracle_memories(
        decoder,
        examples,
        device="cpu",
    )

    oracle = evaluate_memory_required_qa(
        decoder,
        examples,
        mode="oracle",
        device="cpu",
        max_new_tokens=2,
        oracle_values=oracle_values,
    )
    learned = evaluate_memory_required_qa(
        decoder,
        examples,
        mode="learned",
        device="cpu",
        max_new_tokens=2,
    )

    assert tuple(oracle) == (
        "normal",
        "drop",
        "zero",
        "shuffle",
        "drop_slot_0",
        "drop_slot_1",
        "drop_slot_2",
    )
    assert tuple(learned) == (
        "normal",
        "drop",
        "zero",
        "shuffle",
        "no_writes",
        "drop_slot_0",
        "drop_slot_1",
        "drop_slot_2",
    )
    assert oracle["normal"].count == learned["normal"].count == 2
    assert oracle["normal"].mean_first_byte_logit_linf_from_normal == 0
    assert learned["normal"].mean_first_byte_logit_linf_from_normal == 0
    assert tuple(learned["normal"].support_position_exact_accuracy) == (0,)
