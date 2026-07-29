from tinymem.data.correction_deletion import (
    UNKNOWN,
    generate_update_examples,
    interpret_update_example,
)


def test_generation_is_reproducible() -> None:
    first = generate_update_examples(split="train", count=5, base_seed=12)
    second = generate_update_examples(split="train", count=5, base_seed=12)
    assert first == second


def test_split_episode_seeds_and_ids_are_disjoint() -> None:
    train = generate_update_examples(split="train", count=5)
    validation = generate_update_examples(split="validation", count=5)
    test = generate_update_examples(split="test", count=5)
    seed_sets = [{item.episode_seed for item in split} for split in (train, validation, test)]
    assert seed_sets[0].isdisjoint(seed_sets[1])
    assert seed_sets[0].isdisjoint(seed_sets[2])
    assert seed_sets[1].isdisjoint(seed_sets[2])


def test_correction_changes_authoritative_answer() -> None:
    item = generate_update_examples(
        split="train", count=1, correction_counts=(1,), deletion_rate=0.0
    )[0]
    lines = item.example.context.splitlines()
    assert lines[0].startswith("SET ")
    assert lines[1].startswith("CORRECT ")
    assert item.example.answer == lines[1].removesuffix(".").split()[-1]
    assert interpret_update_example(item.example).answer == item.example.answer


def test_deletion_produces_unknown() -> None:
    item = generate_update_examples(split="test", count=1, deletion_rate=1.0)[0]
    assert item.example.answer == UNKNOWN
    assert interpret_update_example(item.example).answer == UNKNOWN


def test_delay_and_distractor_counts_are_independent() -> None:
    item = generate_update_examples(
        split="train", count=1, deletion_rate=0.0, query_delay=5, distractor_count=2
    )[0]
    assert item.query_delay == 5
    assert item.distractor_count == 2
    assert item.example.context.count("DISTRACTOR ") == 2
    assert item.example.context.count("WAIT ") == 3
