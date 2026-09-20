"""Behavioral tests for the bounded phase-two fact data protocol."""

from dataclasses import replace

import pytest

from tinymem.research.delta_fact_data import (
    ENTITIES,
    FAMILIAR_FORMS,
    HELDOUT_FORMS,
    ROOM_PAIRS,
    build_dataset,
    parse_statement,
    replay,
    validate_dataset,
)


def _prefix_groups(dataset):
    groups = {}
    for split, episodes in (
        ("train", dataset.train),
        ("validation", dataset.validation),
        ("test", dataset.test),
    ):
        for episode in episodes:
            groups.setdefault((split, episode.prefix_id, episode.wording), []).append(episode)
    return groups


def test_dataset_is_deterministic_and_validates_all_branch_families():
    first = build_dataset(seed=17, train_prefixes=32, validation_prefixes=16, test_prefixes=16)
    second = build_dataset(seed=17, train_prefixes=32, validation_prefixes=16, test_prefixes=16)

    assert first == second
    validate_dataset(first)
    groups = _prefix_groups(first)
    assert {episode.condition for rows in groups.values() for episode in rows} == {
        "no_write",
        "repeat",
        "correction",
        "balanced",
    }
    assert len(first.train) == 320
    assert len(first.validation) == 160
    assert len(first.test) == 320
    assert all(episode.wording == "familiar" for episode in first.train + first.validation)
    assert {episode.wording for episode in first.test} == {"familiar", "heldout"}


def test_initial_truth_codes_are_balanced_and_prefixes_are_split_disjoint():
    dataset = build_dataset(seed=23, train_prefixes=32, validation_prefixes=16, test_prefixes=16)
    codes = {}
    signatures = {}
    for split, episodes in (
        ("train", dataset.train),
        ("validation", dataset.validation),
        ("test", dataset.test),
    ):
        for episode in episodes:
            key = (split, episode.prefix_id)
            signature = tuple((statement.entity, statement.value) for statement in episode.prefix)
            signatures[key] = signature
            initial = tuple(statement.value for statement in sorted(episode.prefix[:4], key=lambda item: item.entity))
            codes.setdefault(split, set()).add(initial)
    assert all(len(values) == 16 for values in codes.values())
    assert len(set(signatures.values())) == len(signatures)


def test_paired_test_wordings_share_the_same_logical_program():
    dataset = build_dataset(seed=29, train_prefixes=16, validation_prefixes=16, test_prefixes=16)
    by_program = {}
    for episode in dataset.test:
        by_program.setdefault((episode.prefix_id, episode.condition, episode.target), {})[episode.wording] = episode
    assert by_program
    for pair in by_program.values():
        assert set(pair) == {"familiar", "heldout"}
        assert tuple((s.entity, s.value) for s in pair["familiar"].prefix) == tuple(
            (s.entity, s.value) for s in pair["heldout"].prefix
        )
        assert tuple((s.entity, s.value) for s in pair["familiar"].tail) == tuple(
            (s.entity, s.value) for s in pair["heldout"].tail
        )


def test_repeat_correction_balanced_and_no_write_semantics():
    dataset = build_dataset(seed=31, train_prefixes=16, validation_prefixes=16, test_prefixes=16)
    rows = [episode for episode in dataset.train if episode.prefix_id.endswith("p0000")]
    by_condition = {episode.condition: episode for episode in rows}
    before = replay(by_condition["no_write"].prefix)
    assert by_condition["no_write"].tail == ()
    assert by_condition["no_write"].target is None
    assert replay(by_condition["no_write"].prefix + by_condition["no_write"].tail) == before

    for target in range(4):
        repeat = next(row for row in rows if row.condition == "repeat" and row.target == target)
        assert len(repeat.tail) == 8
        assert len({statement.text for statement in repeat.tail}) == 1
        assert {(statement.entity, statement.value) for statement in repeat.tail} == {
            (target, before[target])
        }
        assert replay(repeat.prefix + repeat.tail) == before

        correction = next(row for row in rows if row.condition == "correction" and row.target == target)
        assert len(correction.tail) == 8
        assert len({statement.text for statement in correction.tail}) == 1
        assert correction.tail[0].entity == target
        assert correction.tail[0].value == 1 - before[target]
        assert all(statement.entity == target and statement.value == correction.tail[0].value
                   for statement in correction.tail)
        after = replay(correction.prefix + correction.tail)
        assert after[target] == 1 - before[target]
        assert after[:target] + after[target + 1:] == before[:target] + before[target + 1:]

    balanced = by_condition["balanced"]
    assert len(balanced.tail) == 8
    logical_tail = [(statement.entity, statement.value) for statement in balanced.tail]
    assert all(logical_tail.count(pair) == 2 for pair in logical_tail[:4])
    assert replay(balanced.prefix + balanced.tail) == before


def test_parser_and_replay_are_independent_and_handle_missing_facts():
    assert ENTITIES == ("Alice", "Bob", "Clara", "David")
    assert ROOM_PAIRS == (("bathroom", "hallway"), ("bedroom", "kitchen"), ("garden", "office"), ("bathroom", "hallway"))
    assert len(FAMILIAR_FORMS) == 3
    assert len(HELDOUT_FORMS) == 2
    assert replay(()) == (None, None, None, None)
    parsed = parse_statement("Alice moved to the hallway.")
    assert (parsed.entity, parsed.value, parsed.text) == (0, 1, "Alice moved to the hallway.")
    assert parse_statement("The kitchen is where Bob is.").entity == 1
    assert replay(("Alice is in the bathroom.", "Alice moved to the hallway.")) == (1, None, None, None)
    with pytest.raises(ValueError):
        parse_statement("Alice is in the attic.")
    with pytest.raises(ValueError):
        parse_statement("Eve is in the bathroom.")
    with pytest.raises(ValueError):
        parse_statement("Alice is in the bathroom")


def test_validation_rejects_metadata_or_rendering_tampering():
    dataset = build_dataset(seed=37, train_prefixes=16, validation_prefixes=16, test_prefixes=16)
    original = dataset.train[0]
    bad_statement = replace(original.prefix[0], value=1 - original.prefix[0].value)
    bad_episode = replace(original, prefix=(bad_statement, *original.prefix[1:]))
    tampered = replace(dataset, train=(bad_episode, *dataset.train[1:]))
    with pytest.raises(ValueError, match="metadata"):
        validate_dataset(tampered)

    bad_id = replace(original, id=original.id.replace("no_write-none", "no_write-0"))
    tampered = replace(dataset, train=(bad_id, *dataset.train[1:]))
    with pytest.raises(ValueError, match="id"):
        validate_dataset(tampered)


def test_validation_rejects_template_family_prefix_branch_and_split_tampering():
    dataset = build_dataset(seed=41, train_prefixes=16, validation_prefixes=16, test_prefixes=16)
    familiar = dataset.test[0]
    statement = familiar.prefix[0]
    heldout_text = HELDOUT_FORMS[0].format(
        entity=ENTITIES[statement.entity], room=ROOM_PAIRS[statement.entity][statement.value]
    )
    bad_wording = replace(statement, text=heldout_text)
    bad_episode = replace(familiar, prefix=(bad_wording, *familiar.prefix[1:]))
    tampered = replace(dataset, test=(bad_episode, *dataset.test[1:]))
    with pytest.raises(ValueError, match="template family"):
        validate_dataset(tampered)

    train_prefix = dataset.train[0]
    statement = train_prefix.prefix[0]
    alternate_text = next(
        form.format(entity=ENTITIES[statement.entity], room=ROOM_PAIRS[statement.entity][statement.value])
        for form in FAMILIAR_FORMS
        if form.format(entity=ENTITIES[statement.entity], room=ROOM_PAIRS[statement.entity][statement.value])
        != statement.text
    )
    changed_prefix = replace(statement, text=alternate_text)
    changed_branch = replace(train_prefix, prefix=(changed_prefix, *train_prefix.prefix[1:]))
    tampered = replace(dataset, train=(changed_branch, *dataset.train[1:]))
    with pytest.raises(ValueError, match="exact rendered prefix"):
        validate_dataset(tampered)

    heldout_first = [episode for episode in dataset.test if episode.prefix_id == "test/p0000" and episode.wording == "heldout"]
    heldout_second = [episode for episode in dataset.test if episode.prefix_id == "test/p0001" and episode.wording == "heldout"]
    replacements = iter(heldout_second)
    changed_test = []
    for episode in dataset.test:
        if episode in heldout_first:
            source = next(replacements)
            changed_test.append(replace(episode, prefix=source.prefix, tail=source.tail))
        else:
            changed_test.append(episode)
    tampered = replace(dataset, test=tuple(changed_test))
    with pytest.raises(ValueError, match="logical prefix"):
        validate_dataset(tampered)

    split_mismatch = replace(train_prefix, prefix_id="validation/p0000")
    tampered = replace(dataset, train=(split_mismatch, *dataset.train[1:]))
    with pytest.raises(ValueError, match="split"):
        validate_dataset(tampered)


def test_validation_rejects_unbalanced_or_incomplete_prefix_counts():
    dataset = build_dataset(seed=43, train_prefixes=16, validation_prefixes=16, test_prefixes=16)
    train = tuple(episode for episode in dataset.train if episode.prefix_id != "train/p0000")
    tampered = replace(dataset, train=train)
    with pytest.raises(ValueError, match="multiple of sixteen"):
        validate_dataset(tampered)


def test_invalid_sizes_fail_explicitly():
    for kwargs in (
        {"train_prefixes": 0},
        {"validation_prefixes": 15},
        {"test_prefixes": True},
    ):
        with pytest.raises(ValueError):
            build_dataset(**kwargs)
