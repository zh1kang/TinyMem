"""Measure the original full-text reference for four fixed independent facts."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from scripts.fit_oracle_readout import sources as prior_sources
from scripts.profile_adapted_readout import write_json
from tinymem.research.independent_fact_data import build_worlds, encode_worlds, audit_design
from tinymem.research.independent_fact_evaluation import full_text_records, summarize_reference
from tinymem.research.readout_experiment import _reader_hash
from tinymem.research.study_runtime import REPOSITORY, execution_record, prepare_device, validate_execution
from tinymem.research.update_protocol import file_sha256, load_shared_reader, shared_reader_identity


def sources():
    names = ('src/tinymem/research/independent_fact_data.py', 'src/tinymem/research/independent_fact_evaluation.py',
             'scripts/qualify_independent_facts.py')
    return {**prior_sources(), **{name: file_sha256(REPOSITORY / name) for name in names}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'output', 'declaration'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    declaration = json.loads(args.declaration.read_text())
    declaration_hash = file_sha256(args.declaration)
    identity = shared_reader_identity()
    if (sources() != declaration['source_sha256'] or identity != declaration['reader']
            or file_sha256(args.inputs) != declaration['input_sha256']
            or file_sha256(REPOSITORY / 'data/raw/pretrained/qwen3-1.7b/upstream.json') != declaration['upstream_sha256']):
        raise ValueError('source, reader, or input declaration differs')
    device = prepare_device('cuda')
    execution = execution_record(device)
    validate_execution(execution, expected=declaration['execution'])
    reader = load_shared_reader(identity, device)
    reader_hash = _reader_hash(reader)
    if reader_hash != declaration['reader_sha256']:
        raise ValueError('original qualified reader differs')
    worlds = build_worlds()
    rows = encode_worlds(reader, worlds)
    current = {'worlds': [asdict(w) for w in worlds], 'encodings': [asdict(r) for r in rows], 'design': audit_design()}
    if json.loads(json.dumps(current)) != json.loads(args.inputs.read_text()):
        raise ValueError('native inputs or source labels differ from fixed input file')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'protocol.json', {'kind':'independent_fact_reference_v1', 'declaration_sha256': declaration_hash,
               'input_sha256': declaration['input_sha256'], 'execution': execution, 'source_sha256': sources(),
               'reader': identity, 'training_steps': 0, 'reliability_target': .95, 'hard_scientific_cutoff': False})
    records = full_text_records(reader, rows)
    with (args.output / 'predictions.jsonl').open('x') as handle:
        for r in records:
            handle.write(json.dumps(r, sort_keys=True) + '\n')
    if _reader_hash(reader) != reader_hash or sources() != declaration['source_sha256']:
        raise ValueError('reader or execution sources changed')
    if file_sha256(args.declaration) != declaration_hash or file_sha256(args.inputs) != declaration['input_sha256']:
        raise ValueError('declaration or inputs changed')
    validate_execution(execution_record(device), expected=execution)
    report = {'status':'complete', **summarize_reference(records, rows), 'reader_sha256':reader_hash,
              'reader_unchanged':True, 'training_steps':0, 'generalization_tested':False}
    write_json(args.output / 'report.json', report)
    write_json(args.output / 'complete.json', {'kind':'independent_fact_reference_complete_v1',
               'files': {p.name:file_sha256(p) for p in args.output.iterdir() if p.is_file()}})
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
