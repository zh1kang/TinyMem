import torch

from tinymem.data.babi import parse_babi_lines
from tinymem.data.babilong import parse_babilong_records
from tinymem.data.correction_deletion import generate_update_examples
from tinymem.data.vocabulary import ControlledVocabulary
from tinymem.memory.continuous import (
    MeanPoolMemoryCompressor,
    MultiSlotAttentionMemoryCompressor,
)
from tinymem.memory.controller import AdaptiveWriteController
from tinymem.memory.discrete_compressor import DiscreteMemoryCompressor
from tinymem.memory.recurrent_memory import (
    GatedRecurrentMemoryBank,
    RecurrentMemoryBank,
)
from tinymem.model.config import ModelConfig
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.training.continuous import (
    collate_segmented_answer_supervision,
    encode_qa_with_distributed_facts,
    encode_qa_with_evidence_write_targets,
    encode_qa_with_token_distractor,
    encode_qa_with_token_distractors,
    encode_update_with_write_targets,
    qa1_requires_cross_segment_memory,
    train_continuous_answer_supervision,
)
from tinymem.training.controlled_qa import (
    EncodedQAExample,
    build_qa_vocabulary,
    encode_qa_example,
)
from tinymem.training.discrete import GumbelTemperatureSchedule


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


def test_encode_update_labels_fact_correction_and_deletion_segments() -> None:
    update = generate_update_examples(
        split="test",
        count=1,
        deletion_rate=1.0,
        correction_counts=(1,),
        query_delay=2,
        distractor_count=2,
    )[0].example
    vocabulary = build_qa_vocabulary([update])

    encoded = encode_update_with_write_targets(
        update,
        vocabulary,
        segment_length=4,
    )

    assert encoded.segment_write_targets is not None
    assert encoded.segment_event_types is not None
    assert len(encoded.segment_write_targets) == len(encoded.segment_event_types)
    assert any("set" in labels for labels in encoded.segment_event_types)
    assert any("correction" in labels for labels in encoded.segment_event_types)
    assert any("delete" in labels for labels in encoded.segment_event_types)
    assert any("background" in labels for labels in encoded.segment_event_types)
    assert all(
        encoded.segment_write_targets[index]
        for index, labels in enumerate(encoded.segment_event_types)
        if {"set", "correction", "delete"}.intersection(labels)
    )


def test_distributed_qa1_labels_repeated_movements_as_corrections() -> None:
    examples = parse_babi_lines(
        [
            "1 Mary moved to the kitchen.\n",
            "2 John went to the office.\n",
            "3 Mary travelled to the garden.\n",
            "4 Where is Mary?\tgarden\t3\n",
        ],
        task_id="qa1",
        split="validation",
        source_name="updates.txt",
    )
    vocabulary = build_qa_vocabulary(examples)

    encoded = encode_qa_with_distributed_facts(
        examples[0],
        vocabulary,
        distractor_ids=[vocabulary.token_to_id["<unk>"]] * 24,
        segment_length=4,
    )

    assert encoded.segment_event_types is not None
    labels = {
        label
        for segment_labels in encoded.segment_event_types
        for label in segment_labels
    }
    assert {"set", "correction", "background"} <= labels


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


def make_gated_decoder(vocab_size: int) -> SegmentedContinuousDecoder:
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
    )


def make_discrete_decoder(vocab_size: int) -> SegmentedContinuousDecoder:
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
        DiscreteMemoryCompressor(
            8,
            codebook_size=6,
            summary_slots=1,
            temperature=2.0,
        ),
        RecurrentMemoryBank(capacity=2, model_width=8),
        segment_length=2,
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
    return SegmentedContinuousDecoder(
        DecoderOnlyTransformer(config),
        MultiSlotAttentionMemoryCompressor(8, summary_slots=1),
        GatedRecurrentMemoryBank(capacity=2, model_width=8),
        segment_length=2,
        write_controller=AdaptiveWriteController(8, temperature=2.0),
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


def test_memory_curriculum_requires_evidence_before_the_query_segment() -> None:
    examples = parse_babi_lines(
        [
            "1 Mary moved to the kitchen.\n",
            "2 John went to the office.\n",
            "3 Sandra went to the garden.\n",
            "4 Where is Mary?\tkitchen\t1\n",
        ],
        task_id="qa1",
        split="train",
        source_name="fixture.txt",
    )
    vocabulary = build_qa_vocabulary(examples)

    assert qa1_requires_cross_segment_memory(
        examples[0],
        vocabulary,
        segment_length=8,
    )
    assert not qa1_requires_cross_segment_memory(
        examples[0],
        vocabulary,
        segment_length=128,
    )


def test_encoded_distractor_is_inserted_before_the_question() -> None:
    vocabulary, _ = make_examples()
    raw_example = parse_babi_lines(
        [
            "1 Mary moved to the kitchen.\n",
            "2 Where is Mary?\tkitchen\t1\n",
        ],
        task_id="qa1",
        split="train",
        source_name="fixture.txt",
    )[0]
    distractor_ids = [vocabulary.token_to_id["<unk>"]] * 5

    delayed = encode_qa_with_token_distractor(
        raw_example,
        vocabulary,
        distractor_ids,
    )

    assert delayed.input_ids[-1] == delayed.answer_id
    assert delayed.input_ids.count(vocabulary.token_to_id["<unk>"]) >= 5
    assert len(delayed.input_ids) > len(
        encode_qa_example(raw_example, vocabulary).input_ids
    )


def test_position_randomized_distractors_label_only_context_segments() -> None:
    vocabulary, _ = make_examples()
    raw_example = parse_babi_lines(
        [
            "1 Mary moved to the kitchen.\n",
            "2 Where is Mary?\tkitchen\t1\n",
        ],
        task_id="qa1",
        split="train",
        source_name="fixture.txt",
    )[0]
    unknown = vocabulary.token_to_id["<unk>"]

    delayed = encode_qa_with_token_distractors(
        raw_example,
        vocabulary,
        prefix_distractor_ids=[unknown] * 4,
        suffix_distractor_ids=[unknown] * 4,
        segment_length=4,
    )

    assert delayed.segment_write_targets is not None
    assert not delayed.segment_write_targets[0]
    assert any(delayed.segment_write_targets[1:-1])
    assert not delayed.segment_write_targets[-1]


def test_distributed_fact_encoding_separates_fact_write_targets() -> None:
    vocabulary, _ = make_examples()
    raw_example = parse_babi_lines(
        [
            "1 Mary moved to the kitchen.\n",
            "2 John went to the office.\n",
            "3 Where is Mary?\tkitchen\t1\n",
        ],
        task_id="qa1",
        split="train",
        source_name="fixture.txt",
    )[0]
    unknown = vocabulary.token_to_id["<unk>"]

    distributed = encode_qa_with_distributed_facts(
        raw_example,
        vocabulary,
        distractor_ids=[unknown] * 24,
        segment_length=4,
        gap_rotation=1,
    )

    targets = distributed.segment_write_targets
    assert targets is not None
    positive_segments = [index for index, target in enumerate(targets) if target]
    assert positive_segments
    assert any(
        right - left > 1
        for left, right in zip(
            positive_segments,
            positive_segments[1:],
            strict=False,
        )
    )
    assert not targets[0]
    assert not targets[-1]


def test_evidence_write_encoding_labels_only_the_supporting_fact() -> None:
    vocabulary, _ = make_examples()
    raw_example = parse_babilong_records(
        [
            {
                "input": (
                    "Mary moved to the kitchen.\n"
                    "John went to the office."
                ),
                "question": "Where is Mary? ",
                "target": "kitchen",
            }
        ],
        task_id="qa1",
        split="train",
        source_name="fixture.txt",
    )[0]

    encoded = encode_qa_with_evidence_write_targets(
        raw_example,
        vocabulary,
        segment_length=4,
    )

    assert encoded.input_ids[-1] == encoded.answer_id
    assert encoded.segment_write_targets is not None
    assert any(encoded.segment_write_targets)
    assert not encoded.segment_write_targets[-1]


def test_symbolic_write_loss_updates_the_gated_classifier() -> None:
    torch.manual_seed(29)
    vocabulary, _ = make_examples()
    raw_examples = parse_babi_lines(
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
    unknown = vocabulary.token_to_id["<unk>"]
    examples = [
        encode_qa_with_token_distractors(
            example,
            vocabulary,
            prefix_distractor_ids=[unknown] * prefix,
            suffix_distractor_ids=[unknown] * (4 - prefix),
            segment_length=2,
        )
        for example, prefix in zip(raw_examples, (0, 4), strict=True)
    ]
    decoder = make_gated_decoder(len(vocabulary))
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=0.01)
    before = decoder.bank.write_score.weight.detach().clone()

    losses = train_continuous_answer_supervision(
        decoder,
        optimizer,
        examples,
        steps=2,
        batch_size=2,
        gradient_clip_norm=1.0,
        pad_id=vocabulary.token_to_id["<pad>"],
        device="cpu",
        seed=7,
        write_loss_weight=1.0,
    )

    assert all(torch.isfinite(torch.tensor(losses)))
    assert not torch.equal(decoder.bank.write_score.weight, before)


def test_symbolic_write_loss_accepts_a_single_class_batch() -> None:
    torch.manual_seed(31)
    vocabulary, examples = make_examples()
    example = examples[0]
    segment_count = (len(example.input_ids) + 1) // 2
    labeled = EncodedQAExample(
        input_ids=example.input_ids,
        answer_id=example.answer_id,
        source_example_id=example.source_example_id,
        segment_write_targets=(True,) * segment_count,
    )
    decoder = make_gated_decoder(len(vocabulary))
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=0.01)

    losses = train_continuous_answer_supervision(
        decoder,
        optimizer,
        [labeled],
        steps=1,
        batch_size=1,
        gradient_clip_norm=1.0,
        pad_id=vocabulary.token_to_id["<pad>"],
        device="cpu",
        seed=9,
        write_loss_weight=1.0,
    )

    assert len(losses) == 1
    assert torch.isfinite(torch.tensor(losses[0]))


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


def test_discrete_training_anneals_temperature_and_uses_codes() -> None:
    torch.manual_seed(37)
    vocabulary, examples = make_examples()
    decoder = make_discrete_decoder(len(vocabulary))
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=0.01)
    assert isinstance(decoder.compressor, DiscreteMemoryCompressor)
    before = decoder.compressor.logit_projection.weight.detach().clone()

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
        codebook_usage_loss_weight=0.01,
        temperature_schedule=GumbelTemperatureSchedule(
            start=2.0,
            end=0.5,
            anneal_steps=2,
        ),
    )

    assert len(losses) == 3
    assert all(torch.isfinite(torch.tensor(losses)))
    assert decoder.compressor.temperature == 0.5
    assert not torch.equal(
        decoder.compressor.logit_projection.weight,
        before,
    )


def test_adaptive_training_anneals_temperature_and_charges_writes() -> None:
    torch.manual_seed(41)
    vocabulary, examples = make_examples()
    decoder = make_adaptive_decoder(len(vocabulary))
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=0.01)
    assert isinstance(decoder.write_controller, AdaptiveWriteController)
    before = decoder.write_controller.action_projection.weight.detach().clone()

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
        write_cost_weight=0.01,
        controller_temperature_schedule=GumbelTemperatureSchedule(
            start=2.0,
            end=0.5,
            anneal_steps=2,
        ),
    )

    assert len(losses) == 3
    assert all(torch.isfinite(torch.tensor(losses)))
    assert decoder.write_controller.temperature == 0.5
    assert not torch.equal(
        decoder.write_controller.action_projection.weight,
        before,
    )
