import hashlib
import json

import pytest

from tinymem.research import babi_memory_protocol as protocol


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    (root / 'src').mkdir(parents=True)
    (root / 'src/example.py').write_text('VALUE = 1\n')
    (root / 'pyproject.toml').write_text('[project]\nname="test"\n')
    (root / 'uv.lock').write_text('version = 1\n')
    source = tmp_path / protocol.TRAIN_FILENAME
    source.write_text(('1 Mary moved to the office.\n2 Where is Mary?\toffice\t1\n') * 201)
    monkeypatch.setattr(protocol, 'TRAIN_SHA256', hashlib.sha256(source.read_bytes()).hexdigest())
    study = tmp_path / 'study'
    protocol.prepare(root, study, source, {'test': 'no model used'})
    return root, study


def test_protocol_reconstructs_split_schedule_and_exact_source_inventory(prepared):
    root, study = prepared
    declaration, _, training, validation = protocol.verify(study, root)
    assert len(training) == 1 and len(validation) == 200
    assert len(declaration['schedule']) == 4
    assert all(epoch == [[training[0].source_example_id]] for epoch in declaration['schedule'])
    protocol.verify(study, study / 'source')
    (root / 'src/addition.py').write_text('VALUE = 2\n')
    with pytest.raises(ValueError, match='inventory'):
        protocol.verify(study, root)


@pytest.mark.parametrize('target', ['training_source', 'split', 'schedule', 'copied_source'])
def test_protocol_rejects_tampering(prepared, target):
    root, study = prepared
    if target == 'training_source':
        with (study / protocol.TRAIN_FILENAME).open('a') as handle:
            handle.write('\n')
    elif target == 'copied_source':
        (study / 'source/src/example.py').write_text('VALUE = 9\n')
    else:
        path = study / 'protocol.json'
        value = json.loads(path.read_text())
        if target == 'split':
            value['validation_ids'][0] = value['training_ids'][0]
        else:
            value['schedule'][0][0] = []
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        protocol.verify(study, root)
