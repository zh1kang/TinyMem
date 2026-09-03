import pytest
import torch

from tinymem.memory.continuous import (
    AttentionPoolMemoryCompressor,
    ContinuousMemoryCompressor,
    MeanPoolMemoryCompressor,
    MultiSlotAttentionMemoryCompressor,
)
from tinymem.memory.controller import AdaptiveWriteController
from tinymem.memory.discrete_compressor import DiscreteMemoryCompressor
from tinymem.memory.recurrent_memory import (
    GatedRecurrentMemoryBank,
    RecurrentMemoryBank,
)
from tinymem.memory.write_gate import TokenSegmentWriteGate
from tinymem.model.config import ModelConfig
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.kv_cache import KVCache
from tinymem.model.memory_input import AttentionMemory
from tinymem.model.multi_token_prediction import MultiTokenPredictionHeads
from tinymem.model.transformer import DecoderOnlyTransformer


def make_decoder(
    *,
    segment_length: int = 2,
    capacity: int = 2,
    vocab_size: int = 16,
    compressor_type: type[ContinuousMemoryCompressor] = MeanPoolMemoryCompressor,
    memory_position_mode: str = "absolute",
) -> SegmentedContinuousDecoder:
    config = ModelConfig(
        vocab_size=vocab_size,
        d_model=8,
        n_layers=1,
        n_heads=2,
        d_ff=16,
        max_local_tokens=4,
        dropout=0.0,
    )
    model = DecoderOnlyTransformer(config)
    compressor = compressor_type(config.d_model)
    bank = RecurrentMemoryBank(
        capacity=capacity,
        model_width=config.d_model,
    )
    return SegmentedContinuousDecoder(
        model,
        compressor,
        bank,
        segment_length=segment_length,
        memory_position_mode=memory_position_mode,
    )


def test_segmented_decoder_returns_logits_and_fixed_memory() -> None:
    decoder = make_decoder(segment_length=2, capacity=2)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    token_valid = torch.ones_like(input_ids, dtype=torch.bool)

    output = decoder(input_ids, token_valid)

    assert output.logits.shape == (1, 4, 16)
    assert output.memory.shape == (1, 2, 8)
    assert output.memory_valid.shape == (1, 2)
    assert output.memory_positions.shape == (1, 2)
    assert torch.equal(output.memory_valid, torch.tensor([[True, True]]))
    assert torch.equal(output.memory_positions, torch.tensor([[1, 3]]))
    assert output.memory_codes is None
    assert output.proposed_code_indices is None
    assert output.code_probabilities is None
    assert output.code_assignments is None
    assert output.proposed_code_valid is None
    assert output.prequantized_codes is None
    assert output.mtp_logits is None


def test_segmented_decoder_continues_from_explicit_memory_state() -> None:
    torch.manual_seed(31)
    decoder = make_decoder(segment_length=2, capacity=2)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    token_valid = torch.ones_like(input_ids, dtype=torch.bool)

    complete = decoder(input_ids, token_valid)
    first = decoder(input_ids[:, :2], token_valid[:, :2])
    continued = decoder(
        input_ids[:, 2:],
        token_valid[:, 2:],
        initial_memory=AttentionMemory(
            values=first.memory,
            valid=first.memory_valid,
            positions=first.memory_positions,
        ),
        position_offset=2,
    )

    torch.testing.assert_close(
        continued.logits,
        complete.logits[:, 2:],
    )
    torch.testing.assert_close(continued.memory, complete.memory)
    assert torch.equal(continued.memory_valid, complete.memory_valid)
    assert torch.equal(continued.memory_positions, complete.memory_positions)


def test_segmented_decoder_validates_initial_memory_shape() -> None:
    decoder = make_decoder(segment_length=2, capacity=2)
    input_ids = torch.tensor([[1, 2]])
    initial_memory = AttentionMemory(
        values=torch.zeros(1, 1, 8),
        valid=torch.zeros(1, 1, dtype=torch.bool),
        positions=torch.full((1, 1), -1, dtype=torch.long),
    )

    with pytest.raises(ValueError, match="initial_memory values"):
        decoder(
            input_ids,
            torch.ones_like(input_ids, dtype=torch.bool),
            initial_memory=initial_memory,
            position_offset=2,
        )


def test_segmented_decoder_concatenates_mtp_logits_across_segments() -> None:
    decoder = make_decoder(segment_length=2, capacity=2)
    decoder.mtp_heads = MultiTokenPredictionHeads(8, 16, (2, 3, 4))
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])

    output = decoder(input_ids, torch.ones_like(input_ids, dtype=torch.bool))

    assert output.mtp_logits is not None
    assert tuple(output.mtp_logits) == (2, 3, 4)
    assert all(logits.shape == (1, 5, 16) for logits in output.mtp_logits.values())


def test_segmented_decoder_stores_discrete_codes_at_evaluation() -> None:
    config = ModelConfig(
        vocab_size=16,
        d_model=8,
        n_layers=1,
        n_heads=2,
        d_ff=16,
        max_local_tokens=4,
    )
    compressor = DiscreteMemoryCompressor(
        8,
        codebook_size=6,
        summary_slots=1,
    )
    decoder = SegmentedContinuousDecoder(
        DecoderOnlyTransformer(config),
        compressor,
        RecurrentMemoryBank(capacity=2, model_width=8),
        segment_length=2,
    )
    decoder.eval()

    output = decoder(
        torch.tensor([[1, 2, 3, 4]]),
        torch.ones(1, 4, dtype=torch.bool),
    )

    assert output.memory_codes is not None
    assert output.memory_codes.shape == (1, 2)
    assert (output.memory_codes >= 0).all()
    torch.testing.assert_close(
        output.memory,
        compressor.codebook.embedding(output.memory_codes),
    )
    assert output.proposed_code_indices is not None
    assert output.proposed_code_indices.shape == (1, 2)
    assert output.code_probabilities is not None
    assert output.code_probabilities.shape == (1, 2, 6)
    assert output.code_assignments is not None
    assert output.code_assignments.shape == (1, 2, 6)
    assert output.proposed_code_valid is not None
    assert output.proposed_code_valid.all()
    assert output.prequantized_codes is not None
    assert output.prequantized_codes.shape == (1, 2, 8)


def test_segmented_decoder_can_ablate_with_soft_evaluation_codes() -> None:
    config = ModelConfig(
        vocab_size=16,
        d_model=8,
        n_layers=1,
        n_heads=2,
        d_ff=16,
        max_local_tokens=4,
    )
    compressor = DiscreteMemoryCompressor(
        8,
        codebook_size=6,
        summary_slots=1,
        evaluation_mode="soft",
    )
    decoder = SegmentedContinuousDecoder(
        DecoderOnlyTransformer(config),
        compressor,
        RecurrentMemoryBank(capacity=2, model_width=8),
        segment_length=2,
    )
    decoder.eval()

    output = decoder(
        torch.tensor([[1, 2, 3, 4]]),
        torch.ones(1, 4, dtype=torch.bool),
    )

    assert output.code_assignments is not None
    expected_memory = (
        output.code_assignments @ compressor.codebook.embedding.weight
    )
    torch.testing.assert_close(output.memory, expected_memory)
    assert torch.all((output.code_assignments > 0).sum(dim=-1) > 1)


def test_later_segment_loss_reaches_discrete_compressor() -> None:
    torch.manual_seed(19)
    config = ModelConfig(
        vocab_size=16,
        d_model=8,
        n_layers=1,
        n_heads=2,
        d_ff=16,
        max_local_tokens=4,
    )
    compressor = DiscreteMemoryCompressor(
        8,
        codebook_size=6,
        summary_slots=1,
    )
    decoder = SegmentedContinuousDecoder(
        DecoderOnlyTransformer(config),
        compressor,
        RecurrentMemoryBank(capacity=2, model_width=8),
        segment_length=2,
    )

    output = decoder(
        torch.tensor([[1, 2, 3, 4]]),
        torch.ones(1, 4, dtype=torch.bool),
    )
    output.logits[:, 2:].square().mean().backward()

    assert compressor.logit_projection.weight.grad is not None
    assert torch.count_nonzero(compressor.logit_projection.weight.grad) > 0
    assert compressor.codebook.embedding.weight.grad is not None
    assert torch.count_nonzero(compressor.codebook.embedding.weight.grad) > 0


def test_invalid_segment_does_not_replace_discrete_codes() -> None:
    config = ModelConfig(
        vocab_size=16,
        d_model=8,
        n_layers=1,
        n_heads=2,
        d_ff=16,
        max_local_tokens=4,
    )
    decoder = SegmentedContinuousDecoder(
        DecoderOnlyTransformer(config),
        DiscreteMemoryCompressor(
            8,
            codebook_size=6,
            summary_slots=1,
        ),
        RecurrentMemoryBank(capacity=2, model_width=8),
        segment_length=2,
    )
    decoder.eval()

    output = decoder(
        torch.tensor([[1, 2, 0, 0]]),
        torch.tensor([[True, True, False, False]]),
    )

    assert output.memory_codes is not None
    assert output.memory_codes[0, 0] == -1
    assert output.memory_codes[0, 1] >= 0


def test_virtual_memory_positions_form_a_bounded_prefix() -> None:
    decoder = make_decoder(
        capacity=4,
        memory_position_mode="virtual",
    )
    memory_positions = torch.tensor(
        [[-1, -1, 7, 15], [1, 5, 9, 13]]
    )
    memory_valid = torch.tensor(
        [[False, False, True, True], [True, True, True, True]]
    )

    positions = decoder._attention_memory_positions(
        memory_positions,
        memory_valid,
        position_offset=64,
    )

    assert torch.equal(
        positions,
        torch.tensor([[-1, -1, 62, 63], [60, 61, 62, 63]]),
    )


def test_segmented_decoder_accepts_attention_pooling() -> None:
    decoder = make_decoder(
        compressor_type=AttentionPoolMemoryCompressor,
    )
    input_ids = torch.tensor([[1, 2, 3, 4]])
    token_valid = torch.ones_like(input_ids, dtype=torch.bool)

    output = decoder(input_ids, token_valid)

    assert output.logits.shape == (1, 4, 16)
    assert output.memory_valid.all()


def test_segmented_decoder_appends_multiple_summaries_and_positions() -> None:
    config = ModelConfig(
        vocab_size=16,
        d_model=8,
        n_layers=1,
        n_heads=2,
        d_ff=16,
        max_local_tokens=4,
        dropout=0.0,
    )
    decoder = SegmentedContinuousDecoder(
        DecoderOnlyTransformer(config),
        MultiSlotAttentionMemoryCompressor(
            config.d_model,
            summary_slots=2,
        ),
        RecurrentMemoryBank(capacity=4, model_width=config.d_model),
        segment_length=2,
    )

    output = decoder(
        torch.tensor([[1, 2, 3, 4]]),
        torch.ones(1, 4, dtype=torch.bool),
    )

    assert output.memory_valid.all()
    assert torch.equal(output.memory_positions, torch.tensor([[1, 1, 3, 3]]))
    assert output.writes_applied.shape == (1, 2)
    assert output.write_logits is None


def test_segmented_decoder_exposes_gated_write_logits() -> None:
    config = ModelConfig(
        vocab_size=16,
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
    )

    output = decoder(
        torch.tensor([[1, 2, 3, 4]]),
        torch.ones(1, 4, dtype=torch.bool),
    )

    assert output.writes_applied.shape == (1, 2)
    assert output.write_logits is not None
    assert output.write_logits.shape == (1, 2)


def test_segmented_decoder_exposes_adaptive_controller_state() -> None:
    config = ModelConfig(
        vocab_size=16,
        d_model=8,
        n_layers=1,
        n_heads=2,
        d_ff=16,
        max_local_tokens=4,
    )
    controller = AdaptiveWriteController(8)
    decoder = SegmentedContinuousDecoder(
        DecoderOnlyTransformer(config),
        MultiSlotAttentionMemoryCompressor(8, summary_slots=1),
        GatedRecurrentMemoryBank(capacity=2, model_width=8),
        segment_length=2,
        write_controller=controller,
    )
    decoder.eval()

    output = decoder(
        torch.tensor([[1, 2, 3, 4]]),
        torch.ones(1, 4, dtype=torch.bool),
    )

    assert output.controller_action_logits is not None
    assert output.controller_action_logits.shape == (1, 2, 2)
    assert output.controller_probabilities is not None
    assert output.controller_probabilities.shape == (1, 2, 2)
    assert output.write_logits is not None
    torch.testing.assert_close(
        output.write_logits.sigmoid(),
        output.controller_probabilities[:, :, 1],
    )
    assert output.controller_assignments is not None
    assert output.controller_assignments.shape == (1, 2, 2)
    assert output.controller_surprise is not None
    assert output.controller_surprise.shape == (1, 2)
    assert output.controller_valid is not None
    assert output.controller_valid.shape == (1, 2)
    assert output.controller_valid.all()
    assert output.writes_applied.all()


def test_later_loss_reaches_adaptive_write_controller() -> None:
    torch.manual_seed(23)
    config = ModelConfig(
        vocab_size=16,
        d_model=8,
        n_layers=1,
        n_heads=2,
        d_ff=16,
        max_local_tokens=4,
    )
    controller = AdaptiveWriteController(8)
    decoder = SegmentedContinuousDecoder(
        DecoderOnlyTransformer(config),
        MultiSlotAttentionMemoryCompressor(8, summary_slots=1),
        GatedRecurrentMemoryBank(capacity=2, model_width=8),
        segment_length=2,
        write_controller=controller,
    )

    output = decoder(
        torch.tensor([[1, 2, 3, 4]]),
        torch.ones(1, 4, dtype=torch.bool),
    )
    output.logits[:, 2:].square().mean().backward()

    gradient = controller.action_projection.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_forced_writes_override_adaptive_actions_during_evaluation() -> None:
    config = ModelConfig(
        vocab_size=16,
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

    output = decoder(
        torch.tensor([[1, 2, 3, 4]]),
        torch.ones(1, 4, dtype=torch.bool),
        forced_writes=torch.tensor([[False, True]]),
    )

    assert torch.equal(
        output.writes_applied,
        torch.tensor([[False, True]]),
    )
    assert output.controller_assignments is not None
    assert output.controller_assignments[:, :, 1].all()


def test_forced_writes_are_rejected_during_training() -> None:
    decoder = make_decoder(segment_length=2)

    with pytest.raises(ValueError, match="only during evaluation"):
        decoder(
            torch.tensor([[1, 2]]),
            torch.ones(1, 2, dtype=torch.bool),
            forced_writes=torch.ones(1, 1, dtype=torch.bool),
        )


def test_decoder_rejects_two_external_write_deciders() -> None:
    config = ModelConfig(
        vocab_size=16,
        d_model=8,
        n_layers=1,
        n_heads=2,
        d_ff=16,
        max_local_tokens=4,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        SegmentedContinuousDecoder(
            DecoderOnlyTransformer(config),
            MultiSlotAttentionMemoryCompressor(8, summary_slots=1),
            GatedRecurrentMemoryBank(capacity=2, model_width=8),
            segment_length=2,
            write_gate=TokenSegmentWriteGate(8),
            write_controller=AdaptiveWriteController(8),
        )


def test_token_write_gate_is_independent_of_memory_interventions() -> None:
    config = ModelConfig(
        vocab_size=16,
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
        write_gate=TokenSegmentWriteGate(8),
    )
    input_ids = torch.tensor([[1, 2, 3, 4]])
    token_valid = torch.ones(1, 4, dtype=torch.bool)

    normal = decoder(input_ids, token_valid)
    zeroed = decoder(
        input_ids,
        token_valid,
        memory_intervention=lambda memory: AttentionMemory(
            values=torch.zeros_like(memory.values),
            valid=memory.valid,
            positions=memory.positions,
        ),
    )

    assert normal.write_logits is not None
    assert zeroed.write_logits is not None
    torch.testing.assert_close(normal.write_logits, zeroed.write_logits)


def test_write_gate_gradient_does_not_modify_shared_token_embeddings() -> None:
    config = ModelConfig(
        vocab_size=16,
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
        write_gate=TokenSegmentWriteGate(8),
    )

    output = decoder(
        torch.tensor([[1, 2, 3, 4]]),
        torch.ones(1, 4, dtype=torch.bool),
    )
    assert output.write_logits is not None
    output.write_logits.sum().backward()

    assert decoder.model.token_embedding.weight.grad is None
    assert decoder.write_gate.score.weight.grad is not None


def test_first_segment_logits_use_only_initial_empty_memory() -> None:
    torch.manual_seed(7)
    decoder = make_decoder(segment_length=2)
    decoder.eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    token_valid = torch.ones_like(input_ids, dtype=torch.bool)
    caches = [
        KVCache(max_length=2, start_position=0)
        for _ in range(decoder.model.config.n_layers)
    ]
    expected_hidden = decoder.model.forward_hidden(
        input_ids[:, :2],
        caches=caches,
    )
    expected_logits = decoder.model.lm_head(expected_hidden)

    output = decoder(input_ids, token_valid)

    torch.testing.assert_close(output.logits[:, :2], expected_logits)


def test_invalid_segment_does_not_replace_existing_memory() -> None:
    decoder = make_decoder(segment_length=2, capacity=2)
    input_ids = torch.tensor([[1, 2, 0, 0]])
    token_valid = torch.tensor([[True, True, False, False]])

    output = decoder(input_ids, token_valid)

    assert torch.equal(output.memory_valid, torch.tensor([[False, True]]))
    assert torch.equal(output.memory_positions, torch.tensor([[-1, 1]]))


def test_later_segment_loss_reaches_the_compressor() -> None:
    torch.manual_seed(11)
    decoder = make_decoder(segment_length=2)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    token_valid = torch.ones_like(input_ids, dtype=torch.bool)

    output = decoder(input_ids, token_valid)
    loss = output.logits[:, 2:].square().mean()
    loss.backward()

    gradient = decoder.compressor.projection.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_later_tokens_cannot_change_earlier_segment_logits() -> None:
    torch.manual_seed(13)
    decoder = make_decoder(segment_length=2)
    decoder.eval()
    valid = torch.ones(1, 4, dtype=torch.bool)

    original = decoder(torch.tensor([[1, 2, 3, 4]]), valid)
    changed = decoder(torch.tensor([[1, 2, 9, 10]]), valid)

    torch.testing.assert_close(original.logits[:, :2], changed.logits[:, :2])


def test_segmented_decoder_applies_memory_intervention_to_every_segment() -> None:
    decoder = make_decoder(segment_length=2)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    token_valid = torch.ones_like(input_ids, dtype=torch.bool)
    observed_valid = []

    def observe(memory: object) -> object:
        observed_valid.append(memory.valid.clone())
        return memory

    decoder(
        input_ids,
        token_valid,
        memory_intervention=observe,
    )

    assert len(observed_valid) == 2
    assert not observed_valid[0].any()
    assert observed_valid[1].any()


def test_segmented_decoder_rejects_invalid_memory_intervention() -> None:
    decoder = make_decoder()
    input_ids = torch.tensor([[1, 2]])
    token_valid = torch.ones_like(input_ids, dtype=torch.bool)

    with pytest.raises(TypeError, match="memory_intervention"):
        decoder(input_ids, token_valid, memory_intervention="zero")

    with pytest.raises(TypeError, match="return AttentionMemory"):
        decoder(
            input_ids,
            token_valid,
            memory_intervention=lambda memory: memory.values,
        )


@pytest.mark.parametrize(
    ("input_ids", "token_valid", "error"),
    [
        ([[1, 2]], torch.tensor([[True, True]]), TypeError),
        (
            torch.ones(1, 2, 1, dtype=torch.long),
            torch.ones(1, 2, 1, dtype=torch.bool),
            ValueError,
        ),
        (
            torch.empty(1, 0, dtype=torch.long),
            torch.empty(1, 0, dtype=torch.bool),
            ValueError,
        ),
        (torch.ones(1, 2), torch.ones(1, 2, dtype=torch.bool), TypeError),
        (torch.ones(1, 2, dtype=torch.long), [[True, True]], TypeError),
        (
            torch.ones(1, 2, dtype=torch.long),
            torch.ones(1, 3, dtype=torch.bool),
            ValueError,
        ),
        (torch.ones(1, 2, dtype=torch.long), torch.ones(1, 2), TypeError),
        (
            torch.ones(1, 3, dtype=torch.long),
            torch.tensor([[True, False, True]]),
            ValueError,
        ),
    ],
)
def test_segmented_decoder_rejects_invalid_inputs(
    input_ids: object,
    token_valid: object,
    error: type[Exception],
) -> None:
    decoder = make_decoder()

    with pytest.raises(error):
        decoder(input_ids, token_valid)


@pytest.mark.parametrize("segment_length", [True, 0, 5, 2.5])
def test_segmented_decoder_rejects_invalid_segment_length(
    segment_length: object,
) -> None:
    error = (
        ValueError
        if isinstance(segment_length, int) and not isinstance(segment_length, bool)
        else TypeError
    )

    with pytest.raises(error):
        make_decoder(segment_length=segment_length)


@pytest.mark.parametrize("mode", [None, "relative", 1])
def test_segmented_decoder_rejects_invalid_memory_position_mode(
    mode: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        make_decoder(memory_position_mode=mode)  # type: ignore[arg-type]
