import json
from pathlib import Path

import pytest

from scripts import prepare_memory_updates as builder
from tinymem.data.memory_updates import text_sha256, update_episode_from_dict
from test_memory_updates import sources


def dump(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, sort_keys=True) + "\n")


@pytest.fixture
def input_tree(tmp_path):
    root = tmp_path / "inputs"
    source = root / builder.SOURCE
    groups, representatives, lines = [], [], []
    for index in range(12):
        case = sources(index // 2)[index % 2]
        facts = case.context.splitlines()
        episode = f"{source.name}:episode-{index + 1:06d}"
        group = f"group-{index}"
        context_sha = text_sha256(case.context)
        groups.append({"group_id": group, "episodes": [episode], "context_sha256": [context_sha], "excluded": index < 4})
        representatives.append({"group_id": group, "episode": episode,
            "source_example_id": f"{episode}:question-{len(facts) + 1}", "context_sha256": context_sha})
        lines.extend(f"{number} {fact}\n" for number, fact in enumerate(facts, 1))
        answer = "DO_NOT_PARSE" if 4 <= index < 8 else case.answer
        lines.append(f"{len(facts) + 1} {case.question}\t{answer}\t{len(facts)}\n")
    source.parent.mkdir(parents=True)
    source.write_text("".join(lines))
    selection = {"train": representatives[:2], "development": representatives[2:4], "confirmation": representatives[4:6]}
    dump(root / builder.REFERENCE / "source_selection.json", selection)
    dump(root / builder.REFERENCE / "source_groups.json", {group["group_id"]: group for group in groups[:6]})
    dump(root / builder.RESERVE / "data_manifest.json", {"all_source_groups": groups, "selected": groups[6:8]})
    dump(root / "consumed.json", {"previous_native_data": "fixture identity only"})
    dump(root / builder.RESERVE / "protocol.json", {"exclusion_sha256": {"consumed.json": builder.digest(root / "consumed.json")}})
    reference = {"selection_sha256": builder.digest(root / builder.REFERENCE / "source_selection.json"),
                 "source_groups_sha256": builder.digest(root / builder.REFERENCE / "source_groups.json"),
                 "reserve_manifest_sha256": builder.digest(root / builder.RESERVE / "data_manifest.json"),
                 "reserve_protocol_sha256": builder.digest(root / builder.RESERVE / "protocol.json"),
                 "source_sha256": {str(builder.SOURCE): builder.digest(source)}}
    dump(root / builder.REFERENCE / "protocol.json", reference)
    dump(root / builder.STUDY, {"data_protocol_sha256": builder.digest(root / builder.REFERENCE / "protocol.json")})
    spec = builder.load(builder.ROOT / builder.DESIGN)
    spec["worlds"] = {split: 1 for split in selection}
    spec["status"] = "synthetic_fixture_not_scientific_data"
    spec["old_study_protocol_sha256"] = builder.digest(root / builder.STUDY)
    dump(root / builder.DESIGN, spec)
    return root


def test_real_builder_freezes_paired_data_without_opening_old_confirmation(input_tree, tmp_path, monkeypatch):
    old_open = Path.open
    forbidden = input_tree / builder.REFERENCE / "confirmation.json"

    def guarded_open(path, *args, **kwargs):
        if path == forbidden:
            pytest.fail("the old confirmation dataset must never be opened")
        return old_open(path, *args, **kwargs)

    old_parse = builder.parse_question_payload

    def guarded_parse(payload):
        assert "DO_NOT_PARSE" not in payload, "old confirmation/holdout answer was parsed"
        return old_parse(payload)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(builder, "parse_question_payload", guarded_parse)
    out = tmp_path / "first"
    result = builder.prepare_memory_updates(out, root=input_tree)
    assert result["eligible_fresh_confirmation_groups"] == 4
    assert result["model_loaded"] is result["old_confirmation_data_read"] is False
    assert result["old_confirmation_groups_used"] == result["original_holdout_groups_used"] == 0
    assert result["exclusion_files_verified"] == 1
    for split in ("train", "development", "confirmation"):
        rows = builder.load(out / f"{split}.json")
        assert len(rows) == 1
        row = update_episode_from_dict(rows[0])
        assert len(row.before) == 10 and len(row.branches) == 3
    chosen = builder.load(out / "source_selection.json")
    assert {row["group_id"] for row in chosen["confirmation"]} <= {"group-8", "group-9", "group-10", "group-11"}
    assert chosen["train"] == builder.load(input_tree / builder.REFERENCE / "source_selection.json")["train"]
    second = builder.prepare_memory_updates(tmp_path / "second", root=input_tree)
    assert second == result  # No output-directory or wall-clock effects on frozen identity.
    assert all(builder.digest(out / name) == sha for name, sha in result["data_sha256"].items())
    with pytest.raises(FileExistsError, match="fresh"):
        builder.prepare_memory_updates(out, root=input_tree)


@pytest.mark.parametrize("relative", [builder.STUDY, builder.SOURCE, builder.RESERVE / "data_manifest.json",
    builder.REFERENCE / "source_selection.json", builder.REFERENCE / "source_groups.json", Path("consumed.json")])
def test_input_hash_mismatches_abort_before_outputs(input_tree, tmp_path, relative):
    path = input_tree / relative
    path.write_text(path.read_text() + " ")
    out = tmp_path / "failed"
    with pytest.raises(ValueError, match="identity changed"):
        builder.prepare_memory_updates(out, root=input_tree)
    assert not out.exists()


def test_not_enough_unused_sources_is_not_repaired_by_reusing_confirmation(input_tree, tmp_path):
    spec = builder.load(input_tree / builder.DESIGN)
    spec["worlds"]["confirmation"] = 3
    dump(input_tree / builder.DESIGN, spec)
    with pytest.raises(ValueError, match="not enough fresh"):
        builder.prepare_memory_updates(tmp_path / "failed", root=input_tree)
    assert not (tmp_path / "failed").exists()


@pytest.mark.parametrize("key,value", [("persistent_bytes", 258), ("events_per_branch", 2), ("chunk_separator", "\n"),
    ("initial_chunks", 3), ("query_entities", 9), ("reader_revision", "not-the-pinned-reader"), ("seed", True)])
def test_changed_design_is_not_silently_ignored(input_tree, tmp_path, key, value):
    spec = builder.load(input_tree / builder.DESIGN)
    spec[key] = value
    dump(input_tree / builder.DESIGN, spec)
    with pytest.raises(ValueError):
        builder.prepare_memory_updates(tmp_path / "failed", root=input_tree)
    assert not (tmp_path / "failed").exists()


def test_bad_selected_source_answer_fails_without_resampling(input_tree, tmp_path):
    source = input_tree / builder.SOURCE
    source.write_text(source.read_text().replace("\tbedroom\t", "\twrong\t", 1))
    reference = builder.load(input_tree / builder.REFERENCE / "protocol.json")
    reference["source_sha256"][str(builder.SOURCE)] = builder.digest(source)
    dump(input_tree / builder.REFERENCE / "protocol.json", reference)
    dump(input_tree / builder.STUDY, {"data_protocol_sha256": builder.digest(input_tree / builder.REFERENCE / "protocol.json")})
    spec = builder.load(input_tree / builder.DESIGN)
    spec["old_study_protocol_sha256"] = builder.digest(input_tree / builder.STUDY)
    dump(input_tree / builder.DESIGN, spec)
    with pytest.raises(ValueError, match="source answer"):
        builder.prepare_memory_updates(tmp_path / "failed", root=input_tree)
    assert not (tmp_path / "failed").exists()


def test_disagreeing_hash_pins_fail_instead_of_overwriting(input_tree, tmp_path):
    reserve_path = input_tree / builder.RESERVE / "protocol.json"
    reserve = builder.load(reserve_path)
    reserve["exclusion_sha256"][str(builder.SOURCE)] = "0" * 64
    dump(reserve_path, reserve)
    reference = builder.load(input_tree / builder.REFERENCE / "protocol.json")
    reference["reserve_protocol_sha256"] = builder.digest(reserve_path)
    dump(input_tree / builder.REFERENCE / "protocol.json", reference)
    dump(input_tree / builder.STUDY, {"data_protocol_sha256": builder.digest(input_tree / builder.REFERENCE / "protocol.json")})
    spec = builder.load(input_tree / builder.DESIGN)
    spec["old_study_protocol_sha256"] = builder.digest(input_tree / builder.STUDY)
    dump(input_tree / builder.DESIGN, spec)
    with pytest.raises(ValueError, match="conflicting input hashes"):
        builder.prepare_memory_updates(tmp_path / "failed", root=input_tree)
    assert not (tmp_path / "failed").exists()


def test_reused_split_counts_must_match_frozen_pairing(input_tree, tmp_path):
    spec = builder.load(input_tree / builder.DESIGN)
    spec["worlds"]["train"] = 2
    dump(input_tree / builder.DESIGN, spec)
    with pytest.raises(ValueError, match="reused source counts"):
        builder.prepare_memory_updates(tmp_path / "failed", root=input_tree)
    assert not (tmp_path / "failed").exists()
