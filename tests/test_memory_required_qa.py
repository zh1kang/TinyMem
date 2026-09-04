import torch

from tinymem.data.babi import parse_babi_lines
from tinymem.data.memory_required_qa import build_memory_required_qa_examples
from tinymem.evaluation.memory_required_qa import evaluate_memory_required_qa
from tinymem.memory.continuous import MeanPoolMemoryCompressor
from tinymem.memory.recurrent_memory import RecurrentMemoryBank
from tinymem.model.config import ModelConfig
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.tokenization.byte_tokenizer import ByteTokenizer
from tinymem.training.memory_required_qa import (
    audit_cross_segment_gradients,
    collate_memory_required_distractor,
    collate_memory_required_query,
    collate_memory_required_support,
    precompute_oracle_memories,
    train_memory_required_qa,
    unroll_memory_required_prefix,
)


def make_examples(distractor_segments: int = 0):
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
    )


def make_decoder(memory_capacity: int = 1) -> SegmentedContinuousDecoder:
    config = ModelConfig(
        vocab_size=ByteTokenizer.vocab_size,
        d_model=8,
        n_layers=1,
        n_heads=2,
        d_ff=16,
        max_local_tokens=64,
        dropout=0.0,
    )
    return SegmentedContinuousDecoder(
        DecoderOnlyTransformer(config),
        MeanPoolMemoryCompressor(config.d_model),
        RecurrentMemoryBank(capacity=memory_capacity, model_width=config.d_model),
        segment_length=64,
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

    assert tuple(oracle) == ("normal", "drop", "zero", "shuffle")
    assert tuple(learned) == (
        "normal",
        "drop",
        "zero",
        "shuffle",
        "no_writes",
    )
    assert oracle["normal"].count == learned["normal"].count == 2
    assert oracle["normal"].mean_first_byte_logit_linf_from_normal == 0
    assert learned["normal"].mean_first_byte_logit_linf_from_normal == 0
