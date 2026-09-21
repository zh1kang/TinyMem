"""Evaluation-only protocol amendment for a fitted storage-frontier study."""
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts' / 'frontier'
sys.path.insert(0, str(SCRIPTS))

from frontier_prepare import (
    AMENDABLE,
    accepted_seal_hashes,
    amend,
    frozen_files,
    lineage,
    parse_waiver,
    sha,
    snapshot_source,
    write,
)
from frontier_report import complete
from frontier_run import FITTING_STAGES, sealed_protocol_hashes


def _study(tmp_path: Path, *, cells: int = 2, fitted: bool = True) -> Path:
    study = tmp_path / 'study'
    (study / 'source/src/tinymem/studies/frontier').mkdir(parents=True)
    (study / 'inputs').mkdir()
    (study / 'inputs/dataset.json').write_text('{}')
    (study / 'source/src/tinymem/studies/frontier/eval.py').write_text('EVAL = 1\n')
    (study / 'source/src/tinymem/studies/frontier/fit.py').write_text('FIT = 1\n')
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
    (study / 'source/src/tinymem/studies/frontier/eval.py').write_text('EVAL = 2\n')
    amend(study, 'restore frozen reader after base encoder')
    protocol = json.loads((study / 'protocol.json').read_text())
    assert protocol['amends']['parent_protocol_sha256'] == parent
    assert protocol['amends']['changed_files'] == ['frontier_score.py',
                                                   'source/src/tinymem/studies/frontier/eval.py']
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
    (study / 'source/src/tinymem/studies/frontier/fit.py').write_text('FIT = 2\n')
    with pytest.raises(ValueError, match='fitting-bound'):
        amend(study, 'no')
    assert 'fit.py' not in ' '.join(AMENDABLE)
    (study / 'source/src/tinymem/studies/frontier/fit.py').write_text('FIT = 1\n')
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


def _seal(study: Path, stage: str, cell: int, protocol_sha: str) -> None:
    (study / stage / str(cell)).mkdir(parents=True, exist_ok=True)
    write(study / stage / str(cell) / 'complete.json',
          {'protocol_sha256': protocol_sha, 'cell': cell, 'files': {}})


def _protocol(study: Path) -> dict:
    return json.loads((study / 'protocol.json').read_text())


def test_seal_acceptance_follows_stage_dependencies_through_a_chain(tmp_path):
    study = _study(tmp_path, cells=2)
    original = sha(study / 'protocol.json')
    (study / 'frontier_score.py').write_text('# fixed\n')
    amend(study, 'evaluation fix')
    first = sha(study / 'protocol.json')
    _seal(study, 'evaluation', 0, first)
    _seal(study, 'evaluation', 1, original)
    _seal(study, 'transfer', 0, first)
    protocol = _protocol(study)
    # frontier_score.py changed: evaluation seals under the original are stale,
    # transfer and training seals are not.
    assert accepted_seal_hashes(study, protocol, 'training', 0) == {first, original}
    assert accepted_seal_hashes(study, protocol, 'transfer', 0) == {first, original}
    assert accepted_seal_hashes(study, protocol, 'evaluation', 0) == {first}
    assert complete(study, protocol, 'evaluation', 0).name == '0'
    with pytest.raises(ValueError, match='evaluation/1 provenance'):
        complete(study, protocol, 'evaluation', 1)
    # A second, report-only amendment keeps every evaluation and transfer seal.
    (study / 'frontier_report.py').write_text('# clamp\n')
    amend(study, 'figure fix')
    second = sha(study / 'protocol.json')
    protocol = _protocol(study)
    assert [r['changed_files'] for r in lineage(study, protocol)] == [['frontier_report.py'], ['frontier_score.py']]
    assert accepted_seal_hashes(study, protocol, 'evaluation', 0) == {second, first}
    assert accepted_seal_hashes(study, protocol, 'transfer', 0) == {second, first, original}
    assert accepted_seal_hashes(study, protocol, 'training', 1) == {second, first, original}
    assert complete(study, protocol, 'evaluation', 0).name == '0'
    assert complete(study, protocol, 'transfer', 0).name == '0'
    assert complete(study, protocol, 'training', 1).name == '1'


def test_lineage_rejects_a_tampered_archived_parent(tmp_path):
    study = _study(tmp_path)
    (study / 'frontier_score.py').write_text('# fixed\n')
    amend(study, 'fix')
    protocol = _protocol(study)
    archived = study / protocol['amends']['parent_protocol_file']
    archived.write_text(archived.read_text() + '\n')
    with pytest.raises(ValueError, match='archived parent protocol differs'):
        lineage(study, protocol)


def test_waiver_keeps_a_named_seal_valid_and_is_verified(tmp_path):
    study = _study(tmp_path, cells=2)
    original = sha(study / 'protocol.json')
    (study / 'frontier_score.py').write_text('# fixed\n')
    amend(study, 'first')
    first = sha(study / 'protocol.json')
    _seal(study, 'evaluation', 1, original)
    _seal(study, 'evaluation', 0, first)
    (study / 'frontier_report.py').write_text('# clamp\n')
    waiver = {'stage': 'evaluation', 'cells': [1], 'protocol_sha256': original,
              'reason': 'text cells never use the changed encoder path'}
    with pytest.raises(ValueError, match='not sealed under that hash'):
        amend(study, 'second', ({**waiver, 'cells': [0]},))
    with pytest.raises(ValueError, match='known stage'):
        amend(study, 'second', ({**waiver, 'stage': 'report'},))
    with pytest.raises(ValueError, match='known stage'):
        amend(study, 'second', ({**waiver, 'protocol_sha256': 'f' * 64},))
    amend(study, 'second', (waiver,))
    protocol = _protocol(study)
    assert protocol['amends']['unaffected_seals'] == [waiver]
    assert original in accepted_seal_hashes(study, protocol, 'evaluation', 1)
    assert original not in accepted_seal_hashes(study, protocol, 'evaluation', 0)
    assert complete(study, protocol, 'evaluation', 1).name == '1'


def test_parse_waiver_expands_ranges():
    parsed = parse_waiver('evaluation:9-11,3:abc:text cells: unchanged path')
    assert parsed == {'stage': 'evaluation', 'cells': [9, 10, 11, 3], 'protocol_sha256': 'abc',
                      'reason': 'text cells: unchanged path'}


def test_unamended_protocol_accepts_only_its_own_hash(tmp_path):
    study = _study(tmp_path)
    protocol = json.loads((study / 'protocol.json').read_text())
    assert sealed_protocol_hashes(study, protocol) == {sha(study / 'protocol.json')}


def test_snapshot_source_copies_package_and_frontier_scripts_byte_for_byte(tmp_path):
    repo = SCRIPTS.parents[1]
    study = tmp_path / 'study'
    snapshot_source(study, repo)
    copied = frozen_files(study)
    assert 'source/src/tinymem/studies/frontier/eval.py' in copied
    assert {'frontier_run.py', 'frontier_gpu.slurm', 'frontier_cpu.slurm'} <= set(copied)
    assert copied['frontier_run.py'] == sha(SCRIPTS / 'frontier_run.py')
    assert copied['source/src/tinymem/memory/quantized_slots.py'] == sha(repo / 'src/tinymem/memory/quantized_slots.py')
    assert not list((study / 'source').rglob('__pycache__'))
