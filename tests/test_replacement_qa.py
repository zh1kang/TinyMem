import torch

from tinymem.data.replacement_qa import generate_replacement_qa_examples
from tinymem.evaluation.replacement_qa import (
    REPLACEMENT_CONDITIONS,
    evaluate_replacement_qa,
)
from tinymem.memory.continuous import MeanPoolMemoryCompressor
from tinymem.memory.recurrent_memory import RecurrentMemoryBank
from tinymem.memory.replacement import (
    SlotReplacementController,
    replace_memory_slot,
)
from tinymem.model.config import ModelConfig
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.memory_input import AttentionMemory
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.tokenization.byte_tokenizer import ByteTokenizer
from tinymem.training.losses import next_token_cross_entropy
from tinymem.training.replacement_qa import (
    build_replacement_memory,
    collate_replacement_query,
    replacement_answer_loss,
    train_replacement_qa,
)


def make_decoder(capacity: int = 3) -> SegmentedContinuousDecoder:
    config = ModelConfig(
        vocab_size=ByteTokenizer.vocab_size,
        d_model=8,
        n_layers=1,
        n_heads=2,
        d_ff=16,
        max_local_tokens=512,
        dropout=0.0,
    )
    return SegmentedContinuousDecoder(
        DecoderOnlyTransformer(config),
        MeanPoolMemoryCompressor(config.d_model),
        RecurrentMemoryBank(capacity=capacity, model_width=config.d_model),
        segment_length=64,
        memory_position_mode="virtual",
    )


def make_examples(count: int = 9, capacity: int = 3):
    return generate_replacement_qa_examples(
        ByteTokenizer(),
        split="train",
        count=count,
        memory_capacity=capacity,
        segment_length=64,
        base_seed=17,
    )


def test_replacement_generation_is_reproducible_and_balanced() -> None:
    first = make_examples()
    second = make_examples()

    assert first == second
    assert {example.correction_slot for example in first} == {0, 1, 2}
    assert {example.query_slot for example in first} == {0, 1, 2}
    assert any(example.query_requires_correction for example in first)
    assert any(not example.query_requires_correction for example in first)


def test_replacement_generation_uses_the_correct_final_answer() -> None:
    tokenizer = ByteTokenizer()

    for example in make_examples():
        facts = [tokenizer.decode(ids) for ids in example.initial_fact_ids]
        correction = tokenizer.decode(example.correction_ids)
        answer = tokenizer.decode(example.answer_ids)
        if example.query_requires_correction:
            assert answer in correction
        else:
            assert answer in facts[example.query_slot]


def test_slot_controller_returns_hard_choices_with_gradient() -> None:
    torch.manual_seed(3)
    controller = SlotReplacementController(4)
    correction = torch.randn(2, 4, requires_grad=True)
    memory = torch.randn(2, 3, 4, requires_grad=True)
    valid = torch.tensor([[True, True, True], [True, False, True]])

    output = controller(correction, memory, valid)
    output.assignments[:, 0].sum().backward()

    assert output.logits.shape == (2, 3)
    assert output.probabilities.shape == (2, 3)
    assert output.assignments.shape == (2, 3)
    assert output.slots.shape == (2,)
    assert torch.equal(output.assignments.sum(dim=1), torch.ones(2))
    assert output.probabilities[1, 1] == 0
    assert correction.grad is not None
    assert memory.grad is not None
    assert controller.pair_projection.weight.grad is not None


def test_replacement_is_out_of_place_and_updates_only_selected_slots() -> None:
    values = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    original = values.clone()
    memory = AttentionMemory(
        values=values,
        valid=torch.ones(2, 3, dtype=torch.bool),
        positions=torch.tensor([[3, 7, 11], [3, 7, 11]]),
    )
    replacement = torch.tensor([[30.0] * 4, [40.0] * 4], requires_grad=True)
    assignments = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        requires_grad=True,
    )

    updated = replace_memory_slot(
        memory,
        replacement,
        torch.ones(2, dtype=torch.bool),
        torch.tensor([15, 16]),
        assignments,
    )
    updated.values.sum().backward()

    torch.testing.assert_close(values, original)
    torch.testing.assert_close(updated.values[0, 1], replacement[0])
    torch.testing.assert_close(updated.values[1, 2], replacement[1])
    torch.testing.assert_close(updated.values[0, 0], values[0, 0])
    assert updated.positions.tolist() == [[3, 15, 11], [3, 7, 16]]
    assert replacement.grad is not None
    assert assignments.grad is not None


def test_answer_loss_reaches_correction_summary_and_replacement_controller() -> None:
    torch.manual_seed(5)
    decoder = make_decoder()
    controller = SlotReplacementController(8)
    examples = make_examples(count=3)
    built = build_replacement_memory(
        decoder,
        controller,
        examples,
        pad_id=ByteTokenizer.special_tokens["<pad>"],
        device="cpu",
    )
    built.correction_summary.retain_grad()
    input_ids, targets, valid = collate_replacement_query(
        examples,
        pad_id=ByteTokenizer.special_tokens["<pad>"],
        device="cpu",
    )

    output = decoder(
        input_ids,
        valid,
        initial_memory=built.corrected_memory,
        position_offset=examples[0].query_position_offset,
        update_memory=False,
    )
    next_token_cross_entropy(output.logits, targets).backward()

    assert built.correction_summary.grad is not None
    assert torch.count_nonzero(built.correction_summary.grad) > 0
    assert controller.pair_projection.weight.grad is not None
    assert torch.count_nonzero(controller.pair_projection.weight.grad) > 0


def test_replacement_answer_loss_emphasizes_semantic_prefix_bytes() -> None:
    examples = make_examples(count=2)
    input_ids, targets, _ = collate_replacement_query(
        examples,
        pad_id=ByteTokenizer.special_tokens["<pad>"],
        device="cpu",
    )
    logits = torch.zeros(
        input_ids.shape[0],
        input_ids.shape[1],
        ByteTokenizer.vocab_size,
        requires_grad=True,
    )

    ordinary = replacement_answer_loss(
        logits,
        targets,
        examples,
        semantic_prefix_bytes=2,
        semantic_prefix_weight=1.0,
    )
    emphasized = replacement_answer_loss(
        logits,
        targets,
        examples,
        semantic_prefix_bytes=2,
        semantic_prefix_weight=8.0,
    )
    emphasized.backward()

    torch.testing.assert_close(ordinary, emphasized)
    first_prediction_index = len(examples[0].query_ids) - 1
    final_prediction_index = (
        len(examples[0].query_ids) + len(examples[0].answer_ids) - 1
    )
    first_gradient = logits.grad[0, first_prediction_index].abs().sum()
    final_gradient = logits.grad[0, final_prediction_index].abs().sum()
    assert first_gradient > final_gradient


def test_replacement_training_and_causal_evaluation_smoke() -> None:
    torch.manual_seed(7)
    decoder = make_decoder(capacity=2)
    controller = SlotReplacementController(8)
    examples = make_examples(count=4, capacity=2)
    optimizer = torch.optim.AdamW(
        (*decoder.parameters(), *controller.parameters()),
        lr=0.01,
    )

    history = train_replacement_qa(
        decoder,
        controller,
        optimizer,
        examples,
        steps=2,
        slot_pretrain_steps=1,
        batch_size=2,
        replacement_loss_weight=1.0,
        semantic_prefix_bytes=2,
        semantic_prefix_weight=8.0,
        gradient_clip_norm=1.0,
        pad_id=ByteTokenizer.special_tokens["<pad>"],
        device="cpu",
        seed=11,
    )
    result = evaluate_replacement_qa(
        decoder,
        controller,
        examples,
        device="cpu",
        max_new_tokens=2,
    )

    assert len(history.losses) == 2
    assert len(history.answer_losses) == 2
    assert len(history.replacement_losses) == 2
    assert tuple(result.conditions) == REPLACEMENT_CONDITIONS
    assert len(result.predicted_slots) == len(examples)
