"""Paired read controls with complete per-question development predictions."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.studies.qa1.data import query_entity, replay_locations, story_id
from tinymem.studies.qa1.state import (
    oracle_locations, oracle_state, packed_answer, packed_size, packed_update,
)
from tinymem.studies.qa1.training import encode_question
from tinymem.studies.delta.fit import execution_record
from tinymem.studies.artifacts import file_hash, frozen_base_hash, write_json
from tinymem.studies.delta.protocol import cell_identity, seal_directory, verify_completion
from tinymem.studies.delta.readout import read_answer
from tinymem.studies.oracle.fit import freeze, load_trained
from tinymem.reader.prefix import generate_prefix_answer


def donors(validation) -> dict[str, str]:
    groups = defaultdict(list)
    for source in validation:
        groups[len(source.context.splitlines())].append(source)
    result = {}
    for group in groups.values():
        ordered = sorted(group, key=lambda e: e.source_example_id)
        for index, source in enumerate(ordered):
            candidates = ordered[index + 1:] + ordered[:index]
            donor = next((e for e in candidates if story_id(e) != story_id(source)), None)
            if donor is None:
                raise ValueError('donor control requires a different story with the same fact count')
            result[source.source_example_id] = donor.source_example_id
    return result


def exact(prediction: str, answer: str) -> bool:
    return prediction.strip().lower() == answer.strip().lower()


def summarize(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row['mode']].append(row)
    result = {}
    for mode, group in groups.items():
        correct = [exact(r['prediction'], r['answer']) for r in group]
        entry = {'correct': sum(correct), 'questions': len(group), 'accuracy': sum(correct) / len(group)}
        for field in ('entity', 'answer', 'fact_count', 'context_seen_in_training'):
            subgroups = defaultdict(list)
            for row, value in zip(group, correct, strict=True):
                subgroups[str(row[field])].append(value)
            entry['by_' + field] = {
                key: {'correct': sum(values), 'questions': len(values), 'accuracy': sum(values) / len(values)}
                for key, values in sorted(subgroups.items())}
        if mode == 'donor':
            known = [r for r in group if r['donor_answer'] != 'unknown']
            conflicting = [r for r in known if r['donor_answer'] != r['answer']]
            entry['donor_known'] = len(known)
            entry['donor_unknown'] = len(group) - len(known)
            entry['following_known_donor'] = sum(exact(r['prediction'], r['donor_answer']) for r in known)
            entry['conflicting_known_donor'] = len(conflicting)
            entry['following_conflicting_donor'] = sum(exact(r['prediction'], r['donor_answer']) for r in conflicting)
        result[mode] = entry
    return result


def evaluate_cell(reader, study: Path, protocol: dict, vocabulary, training, validation, index: int) -> dict:
    spec = protocol['settings']
    trained = study / 'training' / str(index)
    verify_completion(trained, cell_identity(study, protocol, index, 'training'))
    training_report = json.loads((trained / 'report.json').read_text())
    expected_steps = sum(len(epoch) for epoch in protocol['schedule'])
    if (training_report['optimizer_steps'] != expected_steps
            or training_report['epochs'] != spec['epochs']
            or training_report['base_before_sha256'] != training_report['base_after_sha256']):
        raise ValueError('training is incomplete or changed frozen base weights')
    if frozen_base_hash(reader) != training_report['unadapted_base_sha256']:
        raise ValueError('evaluation base differs from training')
    runtime = execution_record(reader)
    if runtime != training_report['runtime']:
        raise ValueError('evaluation runtime differs from training')
    directory = study / 'evaluation' / str(index)
    directory.mkdir(parents=True, exist_ok=False)
    bridge = load_trained(reader, protocol, protocol['cells'][index], trained / 'checkpoint.safetensors')
    base_before = frozen_base_hash(reader)
    encoded = {e.source_example_id: encode_question(reader, e, vocabulary) for e in validation}
    donor_map = donors(validation)
    fitting_contexts = {e.context for e in training}
    device = reader.model.device
    states = {str(i) + '.' + part: getattr(encoded[e.source_example_id].state, part)
              for i, e in enumerate(validation) for part in ('values', 'valid')}
    save_file(states, str(directory / 'states.safetensors'))
    rows = []
    with (directory / 'predictions.jsonl').open('x') as handle:
        for source in validation:
            encoded_source = encoded[source.source_example_id]
            native, state = encoded_source.native, encoded_source.state
            before = torch.tensor(native.before_ids, device=device)
            after = torch.tensor(native.after_ids, device=device)
            donor_id = donor_map[source.source_example_id]
            donor = encoded[donor_id].state
            donor_code = oracle_locations(donor, vocabulary)[query_entity(source.question, vocabulary)]
            donor_answer = 'unknown' if donor_code == 0 else vocabulary.rooms[donor_code - 1]
            payload = bytes(packed_size(vocabulary))
            for fact in source.context.splitlines():
                payload = packed_update(payload, fact, vocabulary)
            for mode in spec['controls']:
                if mode == 'packed_explicit':
                    output = {'prediction': packed_answer(payload, source.question, vocabulary),
                              'persistent_bytes': len(payload), 'state_hex': payload.hex()}
                elif mode == 'full_text':
                    with reader.model.disable_adapter():
                        output = generate_prefix_answer(
                            reader, torch.tensor(native.before_ids + native.history_ids, device=device),
                            torch.empty(0, bridge.reader_width, device=device), after,
                            max_new_tokens=spec['max_new_tokens'])
                    # PEFT enables adapter gradients when it restores disabled adapters.
                    freeze(reader, bridge)
                else:
                    selected = donor if mode == 'donor' else state
                    values = torch.zeros_like(selected.values) if mode == 'zero' else selected.values.clone()
                    transferred = LatentSlotState(values.to(device), selected.valid.clone().to(device))
                    output = read_answer(reader, bridge, transferred, before, after,
                                         max_new_tokens=spec['max_new_tokens'])
                    output['persistent_bytes'] = transferred.nbytes
                row = {'source_id': source.source_example_id, 'story_id': story_id(source),
                       'mode': mode, 'question': source.question, 'answer': source.answer,
                       'entity': vocabulary.entities[query_entity(source.question, vocabulary)],
                       'fact_count': len(source.context.splitlines()),
                       'context_sha256': hashlib.sha256(source.context.encode()).hexdigest(),
                       'context_seen_in_training': source.context in fitting_contexts,
                       'donor_id': donor_id if mode == 'donor' else None,
                       'donor_answer': donor_answer if mode == 'donor' else None, **output}
                rows.append(row)
                handle.write(json.dumps(row, allow_nan=False) + '\n')
                handle.flush()
            if len(rows) % 250 == 0:
                print(json.dumps({'cell': index, 'questions_evaluated': len(rows) // len(spec['controls'])}), flush=True)
    if frozen_base_hash(reader) != base_before or base_before != training_report['base_after_sha256']:
        raise ValueError('evaluation changed or used different frozen base weights')
    report = {'metrics': summarize(rows), 'official_test_scored': False, 'runtime': runtime,
              'training_completion_sha256': file_hash(trained / 'complete.json'),
              'checkpoint_sha256': file_hash(trained / 'checkpoint.safetensors'),
              'state_order': [e.source_example_id for e in validation],
              'full_text_reader': 'unadapted base; LoRA disabled',
              'state_code': spec['state_code'], 'persistent_bytes': 258,
              'packed_persistent_bytes': packed_size(vocabulary)}
    write_json(directory / 'report.json', report)
    seal_directory(directory, cell_identity(study, protocol, index, 'evaluation'))
    return report


def audit_rows(directory: Path, rows: list[dict], report: dict, vocabulary, training, validation) -> None:
    order = [e.source_example_id for e in validation]
    if report['state_order'] != order:
        raise ValueError('saved state order differs from development questions')
    tensors = load_file(str(directory / 'states.safetensors'))
    if set(tensors) != {f'{i}.{part}' for i in range(len(order)) for part in ('values', 'valid')}:
        raise ValueError('saved state inventory differs')
    for i, source in enumerate(validation):
        expected = oracle_state(source.context.splitlines(), vocabulary)
        if any(not torch.equal(tensors[f'{i}.{part}'], getattr(expected, part)) for part in ('values', 'valid')):
            raise ValueError('saved state differs from fact-only oracle replay')
    by_id = {e.source_example_id: e for e in validation}
    fitting_contexts = {e.context for e in training}
    donor_map = donors(validation)
    for row in rows:
        source = by_id[row['source_id']]
        expected = {'story_id': story_id(source), 'question': source.question, 'answer': source.answer,
                    'entity': vocabulary.entities[query_entity(source.question, vocabulary)],
                    'fact_count': len(source.context.splitlines()),
                    'context_sha256': hashlib.sha256(source.context.encode()).hexdigest(),
                    'context_seen_in_training': source.context in fitting_contexts}
        if any(row[key] != value for key, value in expected.items()):
            raise ValueError('prediction metadata differs from source')
        donor_id, donor_answer = None, None
        if row['mode'] == 'donor':
            donor_id = donor_map[row['source_id']]
            code = replay_locations(by_id[donor_id].context.splitlines(), vocabulary)[query_entity(source.question, vocabulary)]
            donor_answer = 'unknown' if code == 0 else vocabulary.rooms[code - 1]
        if row['donor_id'] != donor_id or row['donor_answer'] != donor_answer:
            raise ValueError('donor metadata differs from independent replay')


def aggregate(study: Path, protocol: dict, vocabulary, training, validation) -> dict:
    cells = []
    expected = {(e.source_example_id, mode) for e in validation for mode in protocol['settings']['controls']}
    by_id = {e.source_example_id: e for e in validation}
    for cell in protocol['cells']:
        index = cell['index']
        directory = study / 'evaluation' / str(index)
        verify_completion(directory, cell_identity(study, protocol, index, 'evaluation'))
        trained = study / 'training' / str(index)
        verify_completion(trained, cell_identity(study, protocol, index, 'training'))
        report = json.loads((directory / 'report.json').read_text())
        rows = [json.loads(line) for line in (directory / 'predictions.jsonl').read_text().splitlines()]
        if (len(rows) != len(expected) or {(r['source_id'], r['mode']) for r in rows} != expected
                or any(r['answer'] != by_id[r['source_id']].answer for r in rows)
                or report['training_completion_sha256'] != file_hash(trained / 'complete.json')
                or report['checkpoint_sha256'] != file_hash(trained / 'checkpoint.safetensors')
                or report['metrics'] != summarize(rows)):
            raise ValueError('evaluation rows, checkpoints, or metrics differ')
        audit_rows(directory, rows, report, vocabulary, training, validation)
        cells.append({'seed': cell['seed'], 'metrics': report['metrics']})
    result = {'protocol_sha256': file_hash(study / 'protocol.json'), 'cells': cells,
              'official_test_scored': False, 'accuracy_exclusions': [],
              'interpretation': protocol['settings']['interpretation']}
    write_json(study / 'report.json', result)
    return result
