import pytest
import torch

from tinymem.data.babi import parse_babi_lines
from tinymem.data.correction_deletion import generate_update_examples
from tinymem.evaluation.controller import (
    evaluate_controller_policies,
    matched_random_write_mask,
    matched_surprise_write_mask,
    periodic_write_mask,
)
from tinymem.memory.continuous import MultiSlotAttentionMemoryCompressor
from tinymem.memory.controller import AdaptiveWriteController
from tinymem.memory.recurrent_memory import GatedRecurrentMemoryBank
from tinymem.model.config import ModelConfig
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.training.controlled_qa import (
    EncodedQAExample,
    build_qa_vocabulary,
    encode_qa_example,
)
from tinymem.training.continuous import encode_update_with_write_targets


def make_fixture() -> tuple[SegmentedContinuousDecoder, list, int]:
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
        split="validation",
        source_name="fixture.txt",
    )
    vocabulary = build_qa_vocabulary(examples)
    encoded = []
    for example in examples:
        item = encode_qa_example(example, vocabulary)
        segment_count = (len(item.input_ids) + 3) // 4
        encoded.append(
            EncodedQAExample(
                input_ids=item.input_ids,
                answer_id=item.answer_id,
                source_example_id=item.source_example_id,
                segment_write_targets=(True,)
                + (False,) * (segment_count - 1),
            )
        )
    config = ModelConfig(
        vocab_size=len(vocabulary),
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
        segment_length=4,
        write_controller=AdaptiveWriteController(8),
    )
    return decoder, encoded, vocabulary.token_to_id["<pad>"]


def test_policy_masks_enforce_frequency_contracts() -> None:
    valid = torch.tensor(
        [[True, True, True, False], [True, True, False, False]]
    )
    surprise = torch.tensor(
        [[0.1, 0.9, 0.2, 4.0], [0.8, 0.3, 5.0, 6.0]]
    )

    periodic = periodic_write_mask(valid, interval=2)
    random = matched_random_write_mask(valid, writes=2, seed=7)
    surprise_mask = matched_surprise_write_mask(
        surprise,
        valid,
        writes=2,
    )

    assert torch.equal(
        periodic,
        torch.tensor(
            [[False, True, False, False], [False, True, False, False]]
        ),
    )
    assert int(random.sum()) == 2
    assert not (random & ~valid).any()
    assert torch.equal(
        surprise_mask,
        torch.tensor(
            [[False, True, False, False], [True, False, False, False]]
        ),
    )


def test_controller_comparison_covers_all_required_policies() -> None:
    decoder, examples, pad_id = make_fixture()

    comparison = evaluate_controller_policies(
        decoder,
        examples,
        batch_size=2,
        pad_id=pad_id,
        device="cpu",
        periodic_interval=2,
        random_seed=11,
    )

    indexed = {result.policy: result for result in comparison.policies}
    assert set(indexed) == {
        "periodic",
        "random_matched",
        "surprise_threshold",
        "learned",
        "oracle",
    }
    assert indexed["random_matched"].writes == indexed["learned"].writes
    assert indexed["surprise_threshold"].writes == indexed["learned"].writes
    assert comparison.surprise_threshold is not None
    assert len(comparison.traces) == len(examples)
    assert all(trace.surprises for trace in comparison.traces)
    assert all(trace.event_types for trace in comparison.traces)
    assert all(result.tokens > 0 for result in comparison.policies)
    document = comparison.to_dict()
    assert "exit_criteria_met" in document
    assert not comparison.responds_to_corrections


def test_controller_comparison_reports_update_event_write_rates() -> None:
    updates = [
        item.example
        for item in generate_update_examples(
            split="validation",
            count=2,
            deletion_rate=1.0,
            correction_counts=(1,),
            query_delay=1,
            distractor_count=1,
        )
    ]
    vocabulary = build_qa_vocabulary(updates)
    encoded = [
        encode_update_with_write_targets(
            example,
            vocabulary,
            segment_length=4,
        )
        for example in updates
    ]
    config = ModelConfig(
        vocab_size=len(vocabulary),
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
        segment_length=4,
        write_controller=AdaptiveWriteController(8),
    )

    comparison = evaluate_controller_policies(
        decoder,
        encoded,
        batch_size=2,
        pad_id=vocabulary.token_to_id["<pad>"],
        device="cpu",
        periodic_interval=2,
        random_seed=7,
    )

    assert comparison.event_counts["set"] > 0
    assert comparison.event_counts["correction"] > 0
    assert comparison.event_counts["delete"] > 0
    assert comparison.event_counts["background"] > 0
    assert set(comparison.event_write_rates) == set(comparison.event_counts)
    assert "event_types" in comparison.traces[0].to_dict()["events"][0]


@pytest.mark.parametrize("writes", [-1, 6])
def test_matched_random_rejects_impossible_write_counts(writes: int) -> None:
    with pytest.raises(ValueError):
        matched_random_write_mask(
            torch.ones(1, 5, dtype=torch.bool),
            writes=writes,
            seed=0,
        )
