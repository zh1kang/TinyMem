"""Require the complete experiment and report the accuracy-storage curve."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare import accepted_seal_hashes, lineage, sha, write
from run import checked

from tinymem.studies.frontier.analysis import analyze
from tinymem.studies.frontier.data import load_questions
from tinymem.studies.frontier.eval import normalized_exact_match

CODECS = ('compressed_recent', 'compressed_diverse', 'dictionary_recent')


def packed_qa1_reference(study: Path, data: dict) -> dict:
    movement = re.compile(r'([A-Z][a-z]+) (?:moved|went|went back|journeyed|travelled) to the ([a-z]+)\.')
    matches = [movement.fullmatch(record) for q in data['training'] if q['task'] == 'qa1' for record in q['records']]
    if any(match is None for match in matches):
        raise ValueError('QA1 reference encountered unsupported source grammar')
    entities = sorted({m[1] for m in matches})
    rooms = sorted({m[2] for m in matches})
    if len(entities) != 4 or len(rooms) != 6:
        raise ValueError('QA1 packed reference vocabulary changed')
    correct = 0
    cases = [q for q in load_questions(study / 'inputs', 'test') if q.task == 'qa1']
    for q in cases:
        payload = bytes(2)
        for record in q.records:
            match = movement.fullmatch(record)
            entity, room = entities.index(match[1]), rooms.index(match[2]) + 1
            state = int.from_bytes(payload, 'little')
            state = (state & ~(7 << (3 * entity))) | (room << (3 * entity))
            payload = state.to_bytes(2, 'little')
        query = re.fullmatch(r'Where is ([A-Z][a-z]+)\?', q.question)
        code = (int.from_bytes(payload, 'little') >> (3 * entities.index(query[1]))) & 7
        correct += code > 0 and rooms[code - 1] == q.answer
    return {'task': 'qa1', 'level': 0, 'bytes': 2, 'questions': len(cases), 'correct': correct,
            'accuracy': correct / len(cases), 'privilege': 'handwritten grammar and task-specific state',
            'scope': 'reference only; this is not learned and does not generalize to arbitrary text'}


def complete(study: Path, protocol: dict, stage: str, cell: int) -> Path:
    directory = study / stage / str(cell)
    seal = json.loads((directory / 'complete.json').read_text())
    if seal['protocol_sha256'] not in accepted_seal_hashes(study, protocol, stage, cell) or seal['cell'] != cell:
        raise ValueError(f'{stage}/{cell} provenance differs')
    for name, expected in seal['files'].items():
        if sha(directory / name) != expected:
            raise ValueError(f'{stage}/{cell}/{name} differs from its seal')
    return directory


def report(study: Path) -> None:
    protocol, data = checked(study)
    directory = study / 'final_report'
    directory.mkdir(exist_ok=False)
    rows, training, transfer, controls, seals = [], [], [], [], []
    official = {q.id: q for q in load_questions(study / 'inputs', 'test')}
    transfer_inputs = {}
    for task in ('qa1', 'qa2', 'qa3', 'qa4', 'qa5'):
        sources = json.loads((study / 'inputs' / f'babilong-{task}-1k.json').read_text())
        if len(sources) != 100:
            raise ValueError('transfer source inventory differs')
        transfer_inputs.update({f'babilong:{task}:1k:{i}': (task, row) for i, row in enumerate(sources)})
    for cell, declaration in enumerate(protocol['cells']):
        trained = complete(study, protocol, 'training', cell)
        result = json.loads((trained / 'report.json').read_text())
        batch_size = protocol['settings']['batch_size']
        expected_steps = protocol['settings']['epochs'] * ((len(data['training']) + batch_size - 1) // batch_size)
        if (result['steps'] != expected_steps or result['base_hash_before'] != result['base_hash_after']
                or result['official_test_loaded'] or result['seed'] != declaration['seed']):
            raise ValueError('a fitted cell is incomplete or violates the training contract')
        training.append(result)
        evaluated = complete(study, protocol, 'evaluation', cell)
        seals.append({'cell': cell, 'kind': declaration['kind'],
                      'training_protocol_sha256': json.loads((trained / 'complete.json').read_text())['protocol_sha256'],
                      'evaluation_protocol_sha256': json.loads((evaluated / 'complete.json').read_text())['protocol_sha256']})
        cell_rows = [json.loads(line) for line in (evaluated / 'predictions.jsonl').read_text().splitlines()]
        expected = {}
        for task in ('qa1', 'qa2', 'qa3', 'qa4', 'qa5'):
            for level in (0, 2):
                if declaration['kind'] == 'learned':
                    expected.update({(mode, declaration['budget'], level, task): count
                                     for mode, count in (('learned', 1000), ('zero', 100), ('donor', 100))})
                else:
                    expected.update({(mode, budget, level, task): 1000 for mode in CODECS
                                     for budget in protocol['settings']['budgets']})
            if declaration['kind'] == 'text':
                expected['full_text', None, 0, task] = 1000
        if Counter((r['mode'], r['budget'], r['level'], r['task']) for r in cell_rows) != expected:
            raise ValueError('primary or control coverage differs')
        if len({(r['mode'], r['budget'], r['level'], r['id']) for r in cell_rows}) != len(cell_rows):
            raise ValueError('duplicate primary or control prediction')
        for row in cell_rows:
            if row['cell'] != cell or row['seed'] != declaration['seed']:
                raise ValueError('prediction cell or seed identity differs')
            if declaration['kind'] == 'learned' and row['budget'] != declaration['budget']:
                raise ValueError('learned prediction budget differs')
            source = official[row['id']]
            if (row['task'], row['story'], row['answer']) != (source.task, source.story, source.answer):
                raise ValueError('prediction target differs from official source')
            if (row['correct'] != (row['prediction'].strip().lower() == source.answer.strip().lower())
                    or row['normalized_correct'] != normalized_exact_match(row['prediction'], source.answer)):
                raise ValueError('recorded correctness differs from independent scoring')
            if row['mode'] == 'donor' and official[row['donor_id']].story == source.story:
                raise ValueError('donor control reused the current story')
        control_groups = defaultdict(list)
        for row in cell_rows:
            if row['mode'] in ('zero', 'donor', 'full_text'):
                control_groups[(row['mode'], row['level'], row['task'])].append(row['correct'])
        controls.append({'cell': cell, 'results': {f'{m}:{l}:{t}': {'correct': sum(v), 'questions': len(v),
                          'accuracy': sum(v) / len(v)} for (m, l, t), v in control_groups.items()}})
        rows.extend(cell_rows)
        transferred = complete(study, protocol, 'transfer', cell)
        seals[-1]['transfer_protocol_sha256'] = json.loads((transferred / 'complete.json').read_text())['protocol_sha256']
        transfer_rows = [json.loads(line) for line in (transferred / 'predictions.jsonl').read_text().splitlines()]
        modes = ([('learned', declaration['budget'])] if declaration['kind'] == 'learned' else
                 [(mode, budget) for mode in CODECS for budget in protocol['settings']['budgets']])
        expected_transfer = {(mode, budget, identity) for mode, budget in modes for identity in transfer_inputs}
        if (len(transfer_rows) != len(expected_transfer)
                or {(r['mode'], r['budget'], r['id']) for r in transfer_rows} != expected_transfer):
            raise ValueError('transfer task, method, or example coverage differs')
        transfer_groups = defaultdict(list)
        for row in transfer_rows:
            task, source = transfer_inputs[row['id']]
            if (row['cell'] != cell or row['seed'] != declaration['seed'] or row['task'] != task
                    or row['answer'] != source['target'] or row['payload_bytes'] > row['budget']
                    or row['input_sha256'] != hashlib.sha256(source['input'].encode()).hexdigest()
                    or row['correct'] != (row['prediction'].strip().lower() == source['target'].strip().lower())):
                raise ValueError('transfer provenance, storage, or scoring differs')
            transfer_groups[(row['mode'], row['budget'], task)].append(row['correct'])
        transfer.append({'cell': cell, 'seed': declaration['seed'], 'raw_examples': len(transfer_inputs),
                         'results': {f'{m}:{b}:{t}': {'correct': sum(v), 'questions': len(v), 'accuracy': sum(v) / len(v)}
                                     for (m, b, t), v in transfer_groups.items()}})
    result = analyze(rows)
    result.update(protocol_sha256=sha(study / 'protocol.json'), training=training, transfer=transfer, controls=controls,
                  packed_qa1_reference=packed_qa1_reference(study, data), limits=protocol['limits'],
                  amendments=lineage(study, protocol), cell_seals=seals)
    write(directory / 'report.json', result)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, level, title in zip(axes, (0, 2), ('Clean official tasks', 'With text distractors'), strict=True):
        for mode in result['design']['modes']:
            points = sorted((s for s in result['summaries'] if s['mode'] == mode and s['level'] == level),
                            key=lambda s: s['budget'])
            x = [p['budget'] for p in points]
            mean = [100 * p['correct_accuracy']['mean'] for p in points]
            # Identical seeds give differences of a few ulp; matplotlib rejects negatives.
            lower = [max(0.0, m - 100 * p['correct_accuracy']['min']) for m, p in zip(mean, points)]
            upper = [max(0.0, 100 * p['correct_accuracy']['max'] - m) for m, p in zip(mean, points)]
            axis.errorbar(x, mean, yerr=[lower, upper], marker='o', capsize=3, label=mode.replace('_', ' '))
        axis.set(xscale='log', ylim=(0, 102), title=title, xlabel='Retained payload bytes')
        axis.set_xticks([64, 256, 1024], labels=['64', '256', '1,024'])
        axis.grid(alpha=.2)
    axes[0].set_ylabel('Exact answer accuracy (%)')
    axes[1].legend(fontsize=8)
    fig.suptitle('Five-task mean; bars show the range across three training seeds')
    fig.tight_layout()
    fig.savefig(directory / 'accuracy_by_bytes.png', dpi=180)
    plt.close(fig)
    write(directory / 'complete.json', {'protocol_sha256': sha(study / 'protocol.json'),
                                       'cells': len(training), 'predictions': len(rows),
                                       'files': {p.name: sha(p) for p in directory.iterdir() if p.is_file()}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    report(parser.parse_args().study)
