import json
from pathlib import Path
import shutil

import pytest

from scripts import prepare_memory_updates as builder
from test_prepare_memory_updates import input_tree, dump
from tinymem.research.update_protocol import file_sha256, load_development_data, shared_reader_identity


@pytest.fixture
def data_tree(input_tree):
    output = input_tree / "new_data"
    builder.prepare_memory_updates(output, root=input_tree)
    for name in builder.CODE:
        target = input_tree / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(builder.ROOT / name, target)
    return input_tree, output


def test_development_loader_never_opens_confirmation_histories(data_tree, monkeypatch):
    root, output = data_tree
    original = Path.open

    def guarded(path, *args, **kwargs):
        if path.name == "confirmation.json":
            pytest.fail("development verification must not even hash confirmation histories")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    data = load_development_data(output, root=root)
    assert len(data.train) == len(data.development) == 1
    assert data.protocol_sha256 == file_sha256(output / "protocol.json")
    assert data.train[0].episode_id == "memory-update-v1:train:0000"


@pytest.mark.parametrize("name", ["train.json", "development.json", "source_selection.json", "source_groups.json"])
def test_changed_data_fails_before_loading(data_tree, name):
    root, output = data_tree
    path = output / name
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="identity changed"):
        load_development_data(output, root=root)


@pytest.mark.parametrize("section", ["source_sha256", "input_sha256", "data_sha256"])
def test_omitted_provenance_entry_is_not_accepted(data_tree, section):
    root, output = data_tree
    protocol = json.loads((output / "protocol.json").read_text())
    protocol[section].pop(next(iter(protocol[section])))
    dump(output / "protocol.json", protocol)
    with pytest.raises(ValueError):
        load_development_data(output, root=root)


def test_pairing_mismatch_fails_even_with_updated_file_hash(data_tree):
    root, output = data_tree
    path = output / "train.json"
    rows = json.loads(path.read_text())
    rows[0]["source_group_ids"].reverse()
    rows[0]["source_case_ids"].reverse()
    rows[0]["source_context_sha256"].reverse()
    dump(path, rows)
    protocol = json.loads((output / "protocol.json").read_text())
    protocol["data_sha256"][path.name] = file_sha256(path)
    dump(output / "protocol.json", protocol)
    with pytest.raises(ValueError, match="pairing"):
        load_development_data(output, root=root)


def test_changed_generator_and_path_escape_fail(data_tree):
    root, output = data_tree
    with pytest.raises(ValueError, match="outside"):
        load_development_data(root.parent / "foreign", root=root)
    source = root / builder.CODE[0]
    source.write_text(source.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="identity changed"):
        load_development_data(output, root=root)


def test_shared_reader_identity_is_pinned_without_loading_weights(input_tree):
    root = input_tree
    adapter = root / "adapter"
    dump(adapter / "adapter_config.json", {"synthetic": True})
    hashes = {"adapter_config.json": file_sha256(adapter / "adapter_config.json")}
    snapshot = {"synthetic_fixture_not_a_model": True}
    gate = root / "gate"
    dump(gate / "protocol.json", {"protocol": "opaque_qa1_reader_continuation_v1", "snapshot": snapshot})
    dump(gate / "results.json", {"reader_accepted": True, "final_adapter": "adapter", "adapter_sha256": hashes})
    dump(root / builder.STUDY, {"reader_gate": "gate", "snapshot": snapshot, "adapter_sha256": hashes,
        "reader_gate_protocol_sha256": file_sha256(gate / "protocol.json"),
        "reader_gate_results_sha256": file_sha256(gate / "results.json")})
    spec = json.loads((root / builder.DESIGN).read_text())
    spec["old_study_protocol_sha256"] = file_sha256(root / builder.STUDY)
    dump(root / builder.DESIGN, spec)
    identity = shared_reader_identity(root=root)
    assert identity["adapter"] == "adapter" and identity["adapter_sha256"] == hashes
    dump(adapter / "adapter_config.json", {"changed": True})
    with pytest.raises(ValueError, match="identity changed"):
        shared_reader_identity(root=root)
