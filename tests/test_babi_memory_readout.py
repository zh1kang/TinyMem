from copy import deepcopy
from dataclasses import replace
import json

import pytest
import torch

from tinymem.data.babi import parse_babi_lines
from tinymem.research.babi_memory_data import fit_vocabulary, split_training
from tinymem.research.babi_memory_fit import train_cell
from tinymem.research.babi_memory_protocol import schedule, settings
from tinymem.research.babi_memory_scoring import aggregate, audit_rows, donors, evaluate_cell, exact, summarize
from tinymem.research.babi_memory_training import encode_question, train_batch
from tinymem.research.delta_fact_profile import write_json
from tinymem.research.oracle_fact_fit import new_readout

pytest_plugins = ['test_readout_runner']


def examples():
    lines = []
    for room in ('office', 'kitchen', 'office', 'kitchen'):
        lines += [f'1 Mary went to the {room}.', '2 John moved to the kitchen.',
                  f'3 Where is Mary?\t{room}\t1', '4 Mary travelled to the garden.',
                  '5 Where is John?\tkitchen\t2']
    rows = parse_babi_lines(lines, task_id='qa1', split='train', source_name='qa1_train.txt')
    training, validation = split_training(rows, validation_stories=2, seed=8)
    return fit_vocabulary(training), training, validation


def test_fit_reload_controls_and_aggregation(tiny_reader, tmp_path):
    vocabulary, training, validation = examples()
    for row in (*training, *validation):
        encode_question(tiny_reader, row, vocabulary)
    unadapted = deepcopy(tiny_reader)
    spec = {**settings(), 'device': 'cpu', 'epochs': 1, 'batch_size': 2, 'lora_rank': 2, 'max_new_tokens': 2}
    protocol = {'settings': spec, 'schedule': schedule(training, spec),
                'cells': [{'index': 0, 'seed': 5101, 'adapter_seed': 15101, 'bridge_seed': 5102}]}
    write_json(tmp_path / 'protocol.json', protocol)
    fitted = train_cell(tiny_reader, tmp_path, protocol, vocabulary, training, 0)
    assert fitted['optimizer_steps'] == 2
    assert fitted['base_before_sha256'] == fitted['base_after_sha256']
    evaluated = evaluate_cell(unadapted, tmp_path, protocol, vocabulary, training, validation, 0)
    assert evaluated['metrics']['packed_explicit']['accuracy'] == 1
    assert all(m['questions'] == 4 for m in evaluated['metrics'].values())
    assert evaluated['full_text_reader'] == 'unadapted base; LoRA disabled'
    result = aggregate(tmp_path, protocol, vocabulary, training, validation)
    assert result['accuracy_exclusions'] == []
    assert not result['official_test_scored']
    rows = [json.loads(line) for line in (tmp_path / 'evaluation/0/predictions.jsonl').read_text().splitlines()]
    assert all(r['memory_positions'] == 2 for r in rows if r['mode'] in ('oracle', 'zero', 'donor'))
    assert all(r['memory_positions'] == 0 for r in rows if r['mode'] == 'full_text')
    assert all(r['native_envelope_tokens'] < next(s['native_envelope_tokens'] for s in rows
               if s['mode'] == 'full_text' and s['source_id'] == r['source_id'])
               for r in rows if r['mode'] == 'oracle')
    altered = deepcopy(rows)
    next(r for r in altered if r['mode'] == 'donor')['donor_answer'] = 'wrong'
    with pytest.raises(ValueError, match='donor metadata'):
        audit_rows(tmp_path / 'evaluation/0', altered, evaluated, vocabulary, training, validation)
    altered = deepcopy(rows)
    altered[0]['context_seen_in_training'] = not altered[0]['context_seen_in_training']
    with pytest.raises(ValueError, match='prediction metadata'):
        audit_rows(tmp_path / 'evaluation/0', altered, evaluated, vocabulary, training, validation)
    with pytest.raises(FileExistsError):
        train_cell(unadapted, tmp_path, protocol, vocabulary, training, 0)
    with (tmp_path / 'evaluation/0/predictions.jsonl').open('a') as handle:
        handle.write('{}\n')
    with pytest.raises(ValueError, match='outputs changed'):
        aggregate(tmp_path, protocol, vocabulary, training, validation)


def test_training_rejects_validation_and_wrong_optimizer(tiny_reader):
    vocabulary, training, validation = examples()
    source = encode_question(tiny_reader, training[0], vocabulary)
    development = encode_question(tiny_reader, validation[0], vocabulary)
    bridge, adapters = new_readout(tiny_reader, {'settings': {'lora_rank': 2}},
                                   {'adapter_seed': 7, 'bridge_seed': 8})
    optimizer = torch.optim.AdamW([*bridge.parameters(), *adapters])
    with pytest.raises(ValueError, match='training questions'):
        train_batch(tiny_reader, bridge, (development,), optimizer, adapters=adapters,
                    vocabulary=vocabulary, clip_norm=1)
    assert not optimizer.state
    with pytest.raises(ValueError, match='optimizer'):
        train_batch(tiny_reader, bridge, (source,), torch.optim.AdamW(bridge.parameters()),
                    adapters=adapters, vocabulary=vocabulary, clip_norm=1)


def test_donor_selection_is_answer_independent_and_scoring_retains_errors():
    _, _, validation = examples()
    selected = donors(validation)
    assert selected == donors(tuple(replace(e, answer='changed') for e in validation))
    by_id = {e.source_example_id: e for e in validation}
    for source_id, donor_id in selected.items():
        assert source_id.rsplit(':question-', 1)[0] != donor_id.rsplit(':question-', 1)[0]
        assert len(by_id[source_id].context.splitlines()) == len(by_id[donor_id].context.splitlines())
    assert exact('  OFFICE\n', 'office')
    assert not exact('office kitchen', 'office')
    assert not exact('office.', 'office')
    rows = [{'mode': 'donor', 'prediction': 'office', 'answer': 'kitchen', 'donor_answer': 'office',
             'entity': 'Mary', 'fact_count': 2, 'context_seen_in_training': False}]
    report = summarize(rows)['donor']
    assert report['correct'] == 0 and report['following_conflicting_donor'] == 1
