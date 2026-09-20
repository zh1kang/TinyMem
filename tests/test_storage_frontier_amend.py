"""Evaluation-only protocol amendment for a fitted storage-frontier study."""
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts' / 'storage_frontier'
sys.path.insert(0, str(SCRIPTS))

from frontier_prepare import AMENDABLE, amend, frozen_files, sha, write
from frontier_report import complete
from frontier_run import FITTING_STAGES, sealed_protocol_hashes


def _study(tmp_path: Path, *, cells: int = 2, fitted: bool = True) -> Path:
    study = tmp_path / 'study'
    (study / 'source/src/tinymem/research').mkdir(parents=True)
    (study / 'inputs').mkdir()
    (study / 'inputs/dataset.json').write_text('{}')
    (study / 'source/src/tinymem/research/storage_frontier_eval.py').write_text('EVAL = 1\n')
    (study / 'source/src/tinymem/research/storage_frontier_fit.py').write_text('FIT = 1\n')
    for name in ('frontier_run.py', 'frontier_score.py', 'frontier_prepare.py'):
        (study / name).write_text(f'# {name}\n')
    (study / 'frontier_gpu.slurm').write_text('#!/bin/bash\n')
    declared = [{'kind': 'learned', 'budget': 64, 'seed': i} for i in range(cells - 1)]
    declared.append({'kind': 'text', 'budget': None, 'seed': cells - 1})
    write(study / 'protocol.json', {'version': 1, 'files': frozen_files(study), 'cells': declared})
    parent = sha(study / 'protocol.json')
    if fitted:
        for stage in ('features', 'preflight'):
            (study / stage).mkdir()
            write(study / stage / 'complete.json', {'protocol_sha256': parent})
        for cell in range(cells):
            (study / 'training' / str(cell)).mkdir(parents=True)
            write(study / 'training' / str(cell) / 'complete.json',
                  {'protocol_sha256': parent, 'cell': cell, 'files': {}})
    return study


def test_amend_rebinds_evaluation_code_and_keeps_parent_seals(tmp_path):
    study = _study(tmp_path)
    parent = sha(study / 'protocol.json')
    (study / 'frontier_score.py').write_text('# fixed\n')
    (study / 'source/src/tinymem/research/storage_frontier_eval.py').write_text('EVAL = 2\n')
    amend(study, 'restore frozen reader after base encoder')
    protocol = json.loads((study / 'protocol.json').read_text())
    assert protocol['amends']['parent_protocol_sha256'] == parent
    assert protocol['amends']['changed_files'] == ['frontier_score.py',
                                                   'source/src/tinymem/research/storage_frontier_eval.py']
    assert protocol['files'] == frozen_files(study)
    archived = study / protocol['amends']['parent_protocol_file']
    assert sha(archived) == parent
    assert protocol['cells'] == json.loads(archived.read_text())['cells']
    accepted = sealed_protocol_hashes(study, protocol)
    assert accepted == {sha(study / 'protocol.json'), parent}
    for cell in range(2):
        seal = json.loads((study / 'training' / str(cell) / 'complete.json').read_text())
        assert seal['protocol_sha256'] in accepted
    assert set(FITTING_STAGES) == {'features', 'preflight', 'train'}


def test_amend_rejects_fitting_bound_changes(tmp_path):
    study = _study(tmp_path)
    (study / 'source/src/tinymem/research/storage_frontier_fit.py').write_text('FIT = 2\n')
    with pytest.raises(ValueError, match='fitting-bound'):
        amend(study, 'no')
    assert 'storage_frontier_fit.py' not in ' '.join(AMENDABLE)
    (study / 'source/src/tinymem/research/storage_frontier_fit.py').write_text('FIT = 1\n')
    (study / 'inputs/dataset.json').write_text('{"changed": true}')
    with pytest.raises(ValueError, match='fitting-bound'):
        amend(study, 'no')


def test_amend_requires_complete_fitting_and_a_real_change(tmp_path):
    study = _study(tmp_path)
    with pytest.raises(ValueError, match='nothing changed'):
        amend(study, 'no')
    unfitted = _study(tmp_path / 'other', fitted=False)
    (unfitted / 'frontier_score.py').write_text('# fixed\n')
    with pytest.raises(ValueError, match='must be complete'):
        amend(unfitted, 'no')


def test_amend_is_not_chained(tmp_path):
    study = _study(tmp_path)
    (study / 'frontier_score.py').write_text('# fixed\n')
    amend(study, 'first')
    (study / 'frontier_score.py').write_text('# fixed again\n')
    with pytest.raises(ValueError, match='once'):
        amend(study, 'second')


def _seal(study: Path, stage: str, cell: int, protocol_sha: str) -> None:
    (study / stage / str(cell)).mkdir(parents=True, exist_ok=True)
    write(study / stage / str(cell) / 'complete.json',
          {'protocol_sha256': protocol_sha, 'cell': cell, 'files': {}})


def test_report_accepts_parent_seals_only_for_training_and_text_evaluation(tmp_path):
    study = _study(tmp_path, cells=2)
    parent = sha(study / 'protocol.json')
    (study / 'frontier_score.py').write_text('# fixed\n')
    amend(study, 'fix')
    protocol = json.loads((study / 'protocol.json').read_text())
    current = sha(study / 'protocol.json')
    learned, text = 0, 1
    _seal(study, 'evaluation', text, parent)
    _seal(study, 'evaluation', learned, parent)
    _seal(study, 'transfer', text, parent)
    assert complete(study, protocol, 'training', learned).name == '0'
    assert complete(study, protocol, 'evaluation', text).name == '1'
    with pytest.raises(ValueError, match='evaluation/0 provenance'):
        complete(study, protocol, 'evaluation', learned)
    with pytest.raises(ValueError, match='transfer/1 provenance'):
        complete(study, protocol, 'transfer', text)
    _seal(study, 'evaluation', learned, current)
    _seal(study, 'transfer', text, current)
    assert complete(study, protocol, 'evaluation', learned).name == '0'
    assert complete(study, protocol, 'transfer', text).name == '1'


def test_unamended_protocol_accepts_only_its_own_hash(tmp_path):
    study = _study(tmp_path)
    protocol = json.loads((study / 'protocol.json').read_text())
    assert sealed_protocol_hashes(study, protocol) == {sha(study / 'protocol.json')}
