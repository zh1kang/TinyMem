import copy

import pytest
from independent_fact_fixtures import recurrent_manifest


@pytest.fixture(scope="module")
def previous_manifest():
    return recurrent_manifest()


def test_build_preserves_training_and_labels_old_continuity(previous_manifest):
    from tinymem.research.independent_fact_answer_protocol import build_answer_manifest

    manifest = build_answer_manifest(previous_manifest)
    assert manifest["training_streams"] == previous_manifest["training_streams"]
    assert len(manifest["training_streams"]) == 256
    assert len(manifest["old_continuity_streams"]) == 32
    assert all(stream["kind"] == "old_continuity" for stream in manifest["old_continuity_streams"])
    assert [stream["id"] for stream in manifest["old_continuity_streams"]] == [
        stream["id"] for stream in previous_manifest["evaluation_streams"]
    ]


def test_new_programs_have_truthful_mixed_controls_and_complement_endpoints(previous_manifest):
    from tinymem.research.independent_fact_answer_protocol import (
        build_answer_manifest,
        validate_answer_manifest,
    )

    manifest = build_answer_manifest(previous_manifest)
    validate_answer_manifest(manifest)
    streams = manifest["evaluation_streams"]
    assert len(streams) == 32
    assert {stream["program"] for stream in streams} == {"forward_interleaved", "reverse_interleaved"}
    for stream in streams:
        assert stream["id"].startswith("fresh-")
        assert stream["events"][7]["after_code"] == stream["initial_code"] ^ 15
        assert stream["events"][-1]["after_code"] == stream["initial_code"] ^ 15
        first = stream["events"][:8]
        assert {event["action"] for event in first} == {"C", "R"}
        assert {event["target_fact"] for event in first} == {0, 1, 2, 3}
        assert all(sum(event["action"] == action for event in first if event["target_fact"] == fact) == 1
                   for fact in range(4) for action in ("C", "R"))
        assert all(event["action"] == "R" for event in stream["events"][8:])


def test_new_programs_are_deterministic_distinct_and_avoid_training_windows(previous_manifest):
    from tinymem.research.independent_fact_answer_protocol import build_answer_manifest

    first = build_answer_manifest(previous_manifest)
    second = build_answer_manifest(previous_manifest)
    assert first == second
    old_signatures = {
        tuple((event["target_fact"], event["action"]) for event in stream["events"])
        for stream in previous_manifest["evaluation_streams"]
    }
    training_orders = {
        tuple(event["target_fact"] for event in stream["events"])
        for stream in previous_manifest["training_streams"]
    }
    signatures = {
        tuple((event["target_fact"], event["action"]) for event in stream["events"])
        for stream in first["evaluation_streams"]
    }
    assert len(signatures) == 2
    assert not signatures & old_signatures
    for stream in first["evaluation_streams"]:
        targets = [event["target_fact"] for event in stream["events"]]
        assert all(tuple(targets[index:index + 4]) not in training_orders for index in range(13))


def test_state_catalog_covers_initial_primitive_and_old_new_prefixes(previous_manifest):
    from tinymem.research.independent_fact_answer_protocol import (
        build_answer_manifest,
        state_catalog,
    )

    manifest = build_answer_manifest(previous_manifest)
    catalog = state_catalog(manifest)
    assert len(catalog) == 16 + 128 + (32 + 32) * 16
    assert sum(value["kind"] == "initial" for value in catalog.values()) == 16
    assert sum(value["kind"] == "one" for value in catalog.values()) == 128
    assert sum(value["kind"] == "old_continuity" for value in catalog.values()) == 32 * 16
    assert sum(value["kind"] == "new_evaluation" for value in catalog.values()) == 32 * 16
    for key, value in catalog.items():
        assert {"code", "kind", "before_code", "fact", "new_bit", "after_code"} <= set(value)
        if value["kind"] in {"old_continuity", "new_evaluation"}:
            assert key.endswith(f"step-{value['step']:02d}")


def test_validation_rejects_changed_truth_or_reused_new_ids(previous_manifest):
    from tinymem.research.independent_fact_answer_protocol import (
        build_answer_manifest,
        validate_answer_manifest,
    )

    manifest = build_answer_manifest(previous_manifest)
    changed = copy.deepcopy(manifest)
    changed["evaluation_streams"][0]["events"][0]["after_code"] ^= 2
    with pytest.raises(ValueError, match="truth|endpoint|event"):
        validate_answer_manifest(changed)
    changed = copy.deepcopy(manifest)
    changed["evaluation_streams"][1]["id"] = changed["evaluation_streams"][0]["id"]
    with pytest.raises(ValueError, match="duplicate|id"):
        validate_answer_manifest(changed)


def test_sentences_independently_replay_every_new_world(previous_manifest):
    from tinymem.research.independent_fact_answer_protocol import (
        build_answer_manifest,
        state_catalog,
    )
    from tinymem.research.independent_fact_data import (
        ENTITIES,
        ROOM_PAIRS,
        build_worlds,
    )

    manifest = build_answer_manifest(previous_manifest)
    catalog = state_catalog(manifest)
    worlds = build_worlds()
    for stream in manifest['evaluation_streams']:
        truth = {}
        for sentence in worlds[stream['initial_code']].cases[0].context.splitlines():
            entity, room = sentence.removesuffix('.').split(' moved to the ')
            truth[entity] = room
        for event in stream['events']:
            entity, room = event['text'].removesuffix('.').split(' moved to the ')
            truth[entity] = room
            actual_code = sum(ROOM_PAIRS[f].index(truth[e]) * (2 ** f) for f, e in enumerate(ENTITIES[:4]))
            assert actual_code == event['after_code']
            assert catalog[f"{stream['id']}:step-{event['step']:02d}"]['code'] == actual_code
        assert all(truth[e] != worlds[stream['initial_code']].cases[f].answer for f,e in enumerate(ENTITIES[:4]))
