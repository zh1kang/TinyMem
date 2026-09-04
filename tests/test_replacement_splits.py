import hashlib
import json
from collections import Counter
from dataclasses import asdict, replace

import pytest

from tinymem.data.replacement_qa import (
    generate_replacement_qa_examples,
    replacement_history_id,
    validate_replacement_example,
)
from tinymem.tokenization.byte_tokenizer import ByteTokenizer
from tinymem.data.replacement_manifest import replacement_manifest
from tinymem.evaluation.replacement_qa import mismatched_history_indices


def examples(split="train", count=100, seed=0, capacity=2, **kwargs):
    return generate_replacement_qa_examples(
        ByteTokenizer(),
        split=split,
        count=count,
        base_seed=seed,
        memory_capacity=capacity,
        **kwargs,
    )


@pytest.mark.parametrize("capacity", [2, 3, 4])
def test_split_histories_are_disjoint_across_queries_orders_and_seeds(capacity):
    partitions = {}
    for split in ("train", "validation", "test"):
        histories = set()
        for seed in (0, 1337, 2027):
            rows = examples(split, capacity**2 * 20, seed, capacity)
            histories.update(map(replacement_history_id, rows))
            assert len({row.source_example_id for row in rows}) == len(rows)
        partitions[split] = histories

    assert partitions["train"].isdisjoint(partitions["validation"])
    assert partitions["train"].isdisjoint(partitions["test"])
    assert partitions["validation"].isdisjoint(partitions["test"])


def test_replacement_data_is_reproducible_prefix_stable_and_balanced():
    rows = examples(count=1000)

    assert rows == examples(count=1000)
    assert rows[:12] == examples(count=12)
    assert Counter((row.correction_slot, row.query_slot) for row in rows) == {
        (0, 0): 250, (0, 1): 250, (1, 0): 250, (1, 1): 250
    }
    for start in range(0, len(rows), 2):
        first, second = rows[start:start + 2]
        assert replacement_history_id(first) == replacement_history_id(second)
        assert first.query_ids != second.query_ids
        validate_replacement_example(first)
        validate_replacement_example(second)


def test_legacy_generator_replays_pre_repair_content_exactly():
    rows = examples(count=12, seed=1337, protocol="legacy_seed_v1")
    encoded = json.dumps(
        [asdict(row) for row in rows], sort_keys=True, separators=(",", ":")
    ).encode()

    assert hashlib.sha256(encoded).hexdigest() == (
        "034cf90f22ba8e651a3e9cb2c8ba8d021bc3c279d17a74eac24f4b742c495ac6"
    )


def test_history_identity_ignores_query_and_initial_fact_order():
    row = examples(count=1)[0]
    reordered = replace(
        row,
        initial_fact_ids=tuple(reversed(row.initial_fact_ids)),
        correction_slot=1 - row.correction_slot,
        query_slot=1 - row.query_slot,
    )

    assert replacement_history_id(row) == replacement_history_id(reordered)
    validate_replacement_example(reordered)


def test_symbolic_replay_rejects_wrong_answer_and_slot_metadata():
    row = examples(count=1)[0]
    tokenizer = ByteTokenizer()

    with pytest.raises(ValueError, match="answer"):
        validate_replacement_example(
            replace(row, answer_ids=tuple(tokenizer.encode("nowhere")))
        )
    with pytest.raises(ValueError, match="correction_slot"):
        validate_replacement_example(replace(row, correction_slot=1))
    with pytest.raises(ValueError, match="query_slot"):
        validate_replacement_example(replace(row, query_slot=1))


def test_finite_partition_exhaustion_is_explicit():
    with pytest.raises(ValueError, match="distinct histories"):
        examples("validation", count=100_000)


def test_unknown_protocol_is_rejected():
    with pytest.raises(ValueError, match="protocol"):
        examples(protocol="not-a-protocol")


def test_memory_shuffle_changes_every_history_in_grouped_query_data():
    rows = examples(count=12)
    for ordered in (rows, list(reversed(rows)), rows[::2] + rows[1::2]):
        indices = mismatched_history_indices(ordered)
        assert sorted(indices) == list(range(len(ordered)))
        assert all(
            replacement_history_id(row) != replacement_history_id(ordered[other])
            for row, other in zip(ordered, indices, strict=True)
        )
    with pytest.raises(ValueError, match="distinct histories"):
        mismatched_history_indices(rows[:2])
    with pytest.raises(ValueError, match="imbalanced"):
        mismatched_history_indices(rows[:3])


def test_manifest_records_exact_inputs_and_rejects_leaked_histories():
    train = examples(count=12)
    validation = examples("validation", count=12)
    manifest = replacement_manifest(
        train, validation, protocol="history_disjoint_v2",
        data_seed=0, validation_seed=0,
    )
    assert manifest["train_validation_history_overlap"] == 0
    assert manifest["partitions"]["train"]["examples"] == [asdict(row) for row in train]
    leaked = [replace(row, split="validation") for row in train]
    with pytest.raises(ValueError, match="overlap"):
        replacement_manifest(
            train, leaked, protocol="history_disjoint_v2",
            data_seed=0, validation_seed=0,
        )
    legacy = replacement_manifest(
        train, leaked, protocol="legacy_seed_v1", data_seed=0, validation_seed=0,
    )
    assert legacy["train_validation_history_overlap"] == 6
