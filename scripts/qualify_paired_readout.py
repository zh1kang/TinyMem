"""Qualify fixed paired histories with the original full-text reader, without training."""
import argparse
from dataclasses import asdict
from pathlib import Path
import json

from scripts.fit_adapted_readout import sources as readiness_sources
from scripts.profile_adapted_readout import write_json
from tinymem.research.paired_readout_data import ROOM_SWAP, SELECTION_SALT, audit_pairs, build_pairs, encode_pairs
from tinymem.research.readout_evaluation import evaluate_full_text
from tinymem.research.readout_experiment import _reader_hash
from tinymem.research.study_runtime import REPOSITORY, execution_record, prepare_device, validate_execution
from tinymem.research.update_protocol import file_sha256, load_development_data, load_shared_reader, shared_reader_identity


SETTINGS = {'pairs': 4, 'selection_salt': SELECTION_SALT, 'room_swap': ROOM_SWAP,
            'known_minimum_correct': 61, 'missing_minimum_correct': 16,
            'max_new_tokens': 8, 'optimizer_updates': 0, 'full_text_only': True}


def sources():
    names = ('src/tinymem/research/paired_readout_data.py', 'scripts/qualify_paired_readout.py')
    return {**readiness_sources(), **{name: file_sha256(REPOSITORY / name) for name in names}}


def require_qualification(directory: Path, *, report_sha256: str) -> dict:
    """A completed negative result must never authorize optimizer updates."""
    report = json.loads((directory / 'report.json').read_text())
    if (file_sha256(directory / 'report.json') != report_sha256 or report.get('status') != 'complete'
            or report.get('qualified') is not True or report.get('known_correct', 0) < 61
            or report.get('known_total') != 64 or report.get('missing_correct') != 16
            or report.get('missing_total') != 16 or report.get('training_steps') != 0
            or report.get('reader_unchanged') is not True):
        raise ValueError('a hash-verified positive qualification is required before training')
    seal = json.loads((directory / 'complete.json').read_text())
    expected = {'protocol.json', 'report.json', 'paired_cases.json', 'encodings.json', 'input_audit.json', 'predictions.jsonl'}
    if seal['kind'] != 'paired_readout_qualification_complete_v1' or set(seal['files']) != expected:
        raise ValueError('qualification artifact coverage differs')
    if any(file_sha256(directory / name) != seal['files'][name] for name in expected):
        raise ValueError('qualification artifacts changed')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'output', 'declaration'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device = prepare_device('cuda')
    data, identity = load_development_data(args.data), shared_reader_identity()
    runtime, hashes = execution_record(device), sources()
    declaration = json.loads(args.declaration.read_text())
    if (declaration['settings'] != SETTINGS or declaration['source_sha256'] != hashes
            or declaration['portable_source_sha256'] != runtime['source_sha256']
            or declaration['reader'] != identity or declaration['data_protocol_sha256'] != data.protocol_sha256
            or declaration['runtime'] != runtime['runtime']):
        raise ValueError('qualification declaration differs from execution inputs')
    pairs = build_pairs(data.train)
    if [row.source.episode_id for row in pairs[::2]] != declaration['source_history_ids']:
        raise ValueError('source history selection differs from declaration')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'protocol.json', {'kind': 'paired_readout_qualification_v1',
               'declaration_sha256': file_sha256(args.declaration), 'settings': SETTINGS,
               'reader': identity, 'source_sha256': hashes, 'execution': runtime,
               'data_protocol_sha256': data.protocol_sha256, 'training_steps': 0,
               'development_scored': False, 'confirmation_opened': False})
    reader = load_shared_reader(identity, device)
    initial = _reader_hash(reader)
    if initial != declaration['reader_initial_sha256']:
        raise ValueError('initial reader differs from the declared qualified reader')
    rows = encode_pairs(reader, pairs)
    write_json(args.output / 'paired_cases.json', [
        {'history_id': pair.history_id, 'pair_id': pair.pair_id,
         'source_history_id': pair.source.episode_id, 'cases': [asdict(case) for case in pair.cases]}
        for pair in pairs])
    write_json(args.output / 'encodings.json', [asdict(row) for row in rows])
    write_json(args.output / 'input_audit.json', audit_pairs(rows))
    predictions = evaluate_full_text(reader, rows, max_new_tokens=SETTINGS['max_new_tokens'])
    with (args.output / 'predictions.jsonl').open('x') as handle:
        for row in predictions:
            handle.write(json.dumps(row, sort_keys=True) + '\n')
    known = sum(row['correct'] for row in predictions if row['category'] == 'update_known')
    missing = sum(row['correct'] for row in predictions if row['category'] == 'update_missing')
    if _reader_hash(reader) != initial or sources() != hashes:
        raise ValueError('reader values or execution sources changed during qualification')
    if file_sha256(args.declaration) != json.loads((args.output / 'protocol.json').read_text())['declaration_sha256']:
        raise ValueError('qualification declaration changed')
    validate_execution(execution_record(device), expected=runtime)
    report = {'status': 'complete', 'known_correct': known, 'known_total': 64,
              'missing_correct': missing, 'missing_total': 16,
              'qualified': known >= 61 and missing == 16, 'reader_sha256': initial,
              'reader_unchanged': True, 'training_steps': 0, 'generalization_tested': False}
    write_json(args.output / 'report.json', report)
    write_json(args.output / 'complete.json', {'kind': 'paired_readout_qualification_complete_v1',
               'files': {p.name: file_sha256(p) for p in args.output.iterdir() if p.is_file()}})
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
