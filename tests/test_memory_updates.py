from dataclasses import asdict, replace
import json

import pytest

from tinymem.data.memory_updates import (
    EVENT_KINDS, make_update_episode, replay_update_chunks, text_sha256,
    update_episode_from_dict, validate_update_episode, validate_update_splits,
)
from tinymem.data.reader_gate import ReaderCase


def sources(index=0):
    # Distinct source text as well as IDs: repeated visible moves preserve values.
    left = ("Mary moved to the kitchen.\nJohn went to the garden.\n"
            "Daniel journeyed to the bathroom.\nSandra travelled to the office.\n"
            + "Mary went back to the bedroom.\n" * (index + 1)).rstrip()
    right = ("Mary went to the hallway.\nJohn moved to the bedroom.\n"
             "Daniel went back to the office.\nSandra journeyed to the kitchen.\n"
             + "Sandra moved to the kitchen.\n" * (index + 1)).rstrip()
    return (ReaderCase(f"left-{index}:question-1", "babi_qa1", f"left-{index}", left, "Where is Mary?", "bedroom"),
            ReaderCase(f"right-{index}:question-1", "babi_qa1", f"right-{index}", right, "Where is Sandra?", "kitchen"))


def episode(index=0, seed=17):
    return make_update_episode(sources(index), (f"a-{index}", f"b-{index}"),
                               episode_id=f"update:{index}", seed=seed)


def test_paired_events_and_independent_word_based_replay():
    for seed in range(12):
        row = episode(seed=seed)
        assert row == episode(seed=seed)
        assert len(row.initial_chunks) == 4
        assert len(row.before) == 10
        assert len(set(row.entities)) == 10
        assert tuple(branch.kind for branch in row.branches) == EVENT_KINDS
        initial = {}
        for chunk in row.initial_chunks:
            for line in chunk.splitlines():
                words = line.split()
                initial[words[0]] = words[-1][:-1]
        assert [case.answer for case in row.before] == [initial.get(name, "unknown") for name in row.entities]
        for branch in row.branches:
            state = dict(initial)
            words = branch.event.split()
            state[words[0]] = words[-1][:-1]
            assert branch.target == words[0]
            assert [case.answer for case in branch.queries] == [state.get(name, "unknown") for name in row.entities]
            assert [case.question for case in branch.queries] == [case.question for case in row.before]
            assert all(case.context == "\n\n".join((*row.initial_chunks, branch.event)) for case in branch.queries)
            for name in initial.keys() - {branch.target}:
                assert state[name] == initial[name]
            if branch.kind == "addition":
                assert branch.target not in initial and len(state) == 9
            elif branch.kind == "repetition":
                assert state == initial
                assert branch.event in "\n".join(row.initial_chunks).splitlines()
            else:
                assert state[branch.target] != initial[branch.target] and len(state) == 8
            assert branch.queries[-1].answer == "unknown"
            assert row.entities[-1] not in branch.queries[0].context


def test_serialization_round_trip_and_queries_are_not_write_inputs():
    row = episode()
    assert update_episode_from_dict(json.loads(json.dumps(asdict(row)))) == row
    # Visible histories contain only facts; query wording/answer labels are separate.
    text = "\n".join((*row.initial_chunks, *(branch.event for branch in row.branches)))
    assert "Where is" not in text and "unknown" not in text and "query-" not in text
    assert row.episode_id not in text and all(group not in text for group in row.source_group_ids)
    modified = replace(row, before=tuple(replace(case, question="different", answer="different") for case in row.before))
    assert modified.initial_chunks == row.initial_chunks and modified.branches == row.branches
    with pytest.raises(ValueError, match="replay"):
        validate_update_episode(modified)


def test_chunk_separators_make_the_before_write_prefix_identical_after_every_event():
    row = episode()
    before = tuple((chunk + "\n\n").encode() for chunk in row.initial_chunks)
    for branch in row.branches:
        after = (*before, (branch.event + "\n\n").encode())
        assert b"".join(before) == (row.before[0].context + "\n\n").encode()
        assert b"".join(after) == (branch.queries[0].context + "\n\n").encode()
        assert after[:4] == before


@pytest.mark.parametrize("field,value", [
    ("answer", "wrong"), ("question", "Where is someone_else?"), ("category", "update_missing"),
    ("context", "hidden history"), ("case_id", "wrong"), ("history_id", "wrong"),
])
def test_corrupt_before_or_after_metadata_is_rejected(field, value):
    row = episode()
    for stage in ("before", "after"):
        cases = row.before if stage == "before" else row.branches[0].queries
        corrupted = (replace(cases[0], **{field: value}), *cases[1:])
        changed = (replace(row, before=corrupted) if stage == "before" else
                   replace(row, branches=(replace(row.branches[0], queries=corrupted), *row.branches[1:])))
        with pytest.raises(ValueError, match="replay"):
            validate_update_episode(changed)


def test_event_semantics_must_match_not_just_labels():
    row = episode()
    add, repeat, correction = row.branches
    for bad in (replace(add, target=row.entities[0]),
                replace(repeat, event=correction.event),
                replace(correction, event=repeat.event),
                replace(correction, event=correction.event + "\n" + add.event)):
        branches = tuple(bad if branch.kind == bad.kind else branch for branch in row.branches)
        with pytest.raises(ValueError):
            validate_update_episode(replace(row, branches=branches))
    with pytest.raises(ValueError, match="declared order"):
        validate_update_episode(replace(row, branches=(correction, add, repeat)))
    with pytest.raises(ValueError, match="ten questions"):
        validate_update_episode(replace(row, before=row.before[:-1]))


@pytest.mark.parametrize("text", ["", "Who is there?", "Mary moved to the kitchen.",
    "person00000000000000000000 moved to the attic.",
    "person00000000000000000000 moved to the kitchen.\n\n",
    "person00000000000000000000  moved to the kitchen.",
    "person00000000000000000000 moved to the kitchen. Answer: bedroom"])
def test_strict_replay_rejects_nonfacts(text):
    with pytest.raises(ValueError):
        replay_update_chunks((text,))


def split_fixture():
    splits = {name: (episode(index),) for index, name in enumerate(("train", "development", "confirmation"))}
    groups = {}
    for rows in splits.values():
        for row in rows:
            for group, case, sha in zip(row.source_group_ids, row.source_case_ids, row.source_context_sha256, strict=True):
                groups[group] = {"episodes": [case.rsplit(":question-", 1)[0]], "context_sha256": [sha], "excluded": False}
    return splits, groups


def test_disjoint_splits_and_blocked_sources():
    splits, groups = split_fixture()
    validate_update_splits(splits, source_groups=groups, blocked_groups=set())
    with pytest.raises(ValueError, match="forbidden"):
        validate_update_splits(splits, source_groups=groups, blocked_groups={"a-1"})
    groups["a-2"]["excluded"] = True
    with pytest.raises(ValueError, match="consumed"):
        validate_update_splits(splits, source_groups=groups, blocked_groups=set())


def test_source_aliases_and_reused_worlds_cannot_evade_split_checks():
    splits, groups = split_fixture()
    for field in ("source_group_ids", "source_case_ids", "source_context_sha256"):
        damaged = {**splits, "development": (replace(splits["development"][0], **{field: getattr(splits["train"][0], field)}),)}
        with pytest.raises(ValueError, match="overlap"):
            validate_update_splits(damaged, source_groups=groups, blocked_groups=set())
    with pytest.raises(ValueError, match="duplicate"):
        validate_update_splits({**splits, "development": splits["train"]}, source_groups=groups, blocked_groups=set())
    del groups["a-2"]
    with pytest.raises(ValueError, match="manifest"):
        validate_update_splits(splits, source_groups=groups, blocked_groups=set())


def test_manifest_membership_checks_and_unknown_fields():
    splits, groups = split_fixture()
    groups["a-1"]["context_sha256"] = [text_sha256("not this source")]
    with pytest.raises(ValueError, match="manifest"):
        validate_update_splits(splits, source_groups=groups, blocked_groups=set())
    for at_branch in (False, True):
        row = asdict(episode())
        target = row["branches"][0] if at_branch else row
        target["secret_answer_cache"] = ["kitchen"]
        with pytest.raises(ValueError, match="unexpected"):
            update_episode_from_dict(row)
    row = asdict(episode())
    row["before"][0]["secret_answer_cache"] = "bedroom"
    with pytest.raises(TypeError):
        update_episode_from_dict(row)


@pytest.mark.parametrize("seed", [True, -1, 1.5])
def test_invalid_seed_is_not_silently_coerced(seed):
    with pytest.raises(ValueError):
        episode(seed=seed)
