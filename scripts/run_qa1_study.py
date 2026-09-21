"""Prepare, fit, and evaluate QA1 readers using only the official training file."""

import argparse
import json
from pathlib import Path

import torch

from tinymem.studies.qa1.fit import train_cell
from tinymem.studies.qa1.protocol import prepare, verify
from tinymem.studies.qa1.scoring import aggregate, evaluate_cell
from tinymem.studies.artifacts import write_json
from tinymem.studies.delta.protocol import cell_identity
from tinymem.reader.pretrained import load_qwen_reader, verify_qwen_snapshot
from tinymem.studies.runtime import REPOSITORY, prepare_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='stage', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--training-file', type=Path, required=True)
    prep.add_argument('--snapshot', type=Path, required=True)
    for stage in ('train', 'evaluate', 'report', 'verify'):
        command = commands.add_parser(stage)
        command.add_argument('--study', type=Path, required=True)
        if stage in ('train', 'evaluate'):
            command.add_argument('--snapshot', type=Path, required=True)
            command.add_argument('--cell', type=int, required=True)
    args = parser.parse_args()
    if args.stage == 'prepare':
        protocol = prepare(REPOSITORY, args.output, args.training_file, verify_qwen_snapshot(args.snapshot))
        print(json.dumps({'status': 'prepared', 'cells': len(protocol['cells']), 'official_test_loaded': False}))
        return
    protocol, vocabulary, training, validation = verify(args.study, REPOSITORY)
    if args.stage == 'verify':
        print(json.dumps({'status': 'verified', 'training_questions': len(training),
                          'development_questions': len(validation), 'official_test_loaded': False}))
        return
    if args.stage == 'report':
        aggregate(args.study, protocol, vocabulary, training, validation)
        print(json.dumps({'status': 'reported', 'path': str(args.study / 'report.json')}))
        return
    stage = 'training' if args.stage == 'train' else 'evaluation'
    cell_identity(args.study, protocol, args.cell, stage)
    if verify_qwen_snapshot(args.snapshot) != protocol['snapshot']:
        raise ValueError('reader snapshot differs from protocol')
    directory = args.study / stage / str(args.cell)
    if directory.exists():
        raise FileExistsError(f'preserve the existing attempt: {directory}')
    device = prepare_device(protocol['settings']['device'])
    torch.set_num_threads(4)
    reader = load_qwen_reader(args.snapshot, device=device, dtype=torch.bfloat16)
    try:
        if args.stage == 'train':
            result = train_cell(reader, args.study, protocol, vocabulary, training, args.cell)
        else:
            result = evaluate_cell(reader, args.study, protocol, vocabulary, training, validation, args.cell)
        verify(args.study, REPOSITORY)
    except Exception as error:
        if directory.exists() and not (directory / 'complete.json').exists():
            write_json(directory / 'failure.json', {'stage': args.stage, 'error_type': type(error).__name__,
                       'error': str(error), 'retry': 'none; preserve the failed attempt'})
        raise
    print(json.dumps({'status': 'complete', 'stage': args.stage, 'result': result}, allow_nan=False))


if __name__ == '__main__':
    main()
