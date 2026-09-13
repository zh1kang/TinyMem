"""Behavioral oracles for coordinate recovery, parity shortcuts, and missing evidence."""
from copy import deepcopy

import pytest

from test_readout_runner import tiny_reader
from tinymem.research.independent_fact_data import build_worlds, encode_worlds, ROOM_PAIRS
from tinymem.research.independent_fact_fit_evaluation import summarize_codes


def records_for(rows, predict):
    labels = sorted({q.answer for row in rows for q in row.queries})

    def output(code, i):
        prediction = predict(code, i)
        return {'prediction':prediction,'generated_ids':[labels.index(prediction)]}

    records = [{'condition':'normal','code':row.code,'query_index':i,**output(row.code,i)}
               for row in rows for i in range(6)]
    records += [{'condition':condition,'code':None,'query_index':i,**output(0,i)}
                for condition in ('zero','no_memory') for i in range(6)]
    records += [{'condition':'donor_replay','recipient_code':code,'axis':axis,'code':code^(1<<axis),
                 'query_index':i,**output(code^(1<<axis),i)}
                for code in (0,1) for axis in range(4) for i in range(6)]
    records += [{'condition':'full_text','code':row.code,'history_id':row.history_id,'case_id':q.case_id,
                 'category':q.category,'answer':q.answer,'prediction':q.answer,'correct':True,
                 'generated_ids':[labels.index(q.answer)]} for row in rows for q in row.queries]
    return records


@pytest.mark.parametrize('parity',[0,1])
def test_coordinate_reader_generalizes_and_changes_only_requested_fact(tiny_reader, parity):
    rows = encode_worlds(tiny_reader, build_worlds())
    report = summarize_codes(records_for(rows,lambda code,i:rows[code].queries[i].answer),rows,parity)
    for split in report['splits'].values():
        scores = split['scores']
        assert scores['normal']['known']['correct'] == 32
        assert scores['normal']['absent']['correct'] == 16
        assert scores['zero']['known']['correct'] == scores['no_memory']['known']['correct'] == 16
    assert report['splits']['heldout']['scores']['nearest_training_world']['known']['correct'] == 24
    assert report['coordinate_totals'] == {
        'unique_edges':32,'target_answer_changed':32,'target_both_correct':32,
        'untouched_known_answer_changes':0,'untouched_known_both_correct':96,
        'untouched_train_to_heldout_correct_to_wrong':0,'untouched_heldout_to_train_correct_to_wrong':0,
        'absent_answer_changes':0,'absent_both_correct':64}
    assert report['complement_totals'] == {'unique_pairs':8,'known_answer_changes':32,'known_both_correct':32,
                                           'absent_answer_changes':0,'absent_both_correct':16}
    assert report['generation_records'] == 252 and not report['derived_controls_are_independent_samples']


@pytest.mark.parametrize('parity',[0,1])
def test_three_other_bit_shortcut_fits_training_but_inverts_heldout(tiny_reader, parity):
    rows = encode_worlds(tiny_reader, build_worlds())

    def shortcut(code, i):
        if i >= 4:
            return 'unknown'
        other_bits = (code & ~(1 << i)).bit_count() % 2
        return ROOM_PAIRS[i][other_bits ^ parity]

    report = summarize_codes(records_for(rows,shortcut),rows,parity)
    assert report['splits']['train']['scores']['normal']['known']['correct'] == 32
    assert report['splits']['heldout']['scores']['normal']['known']['correct'] == 0
    totals = report['coordinate_totals']
    assert totals['target_answer_changed'] == totals['target_both_correct'] == 0
    assert totals['untouched_known_answer_changes'] == totals['untouched_train_to_heldout_correct_to_wrong'] == 96
    assert totals['untouched_heldout_to_train_correct_to_wrong'] == 0
    # Complement preserves parity, so the shortcut changes every known answer yet stays wrong on held-out worlds.
    assert report['complement_totals']['known_answer_changes'] == 32
    assert report['complement_totals']['known_both_correct'] == 16
    assert report['scientific_decision'] == 'review_generalization_controls_and_errors'


def test_question_only_reader_stays_at_balanced_ceiling(tiny_reader):
    rows = encode_worlds(tiny_reader, build_worlds())
    report = summarize_codes(records_for(rows,lambda code,i:rows[0].queries[i].answer),rows,0)
    for split in report['splits'].values():
        assert split['scores']['normal']['known']['accuracy'] == .5
        assert split['normal_known_gaps']['zero'] == split['normal_known_gaps']['no_memory'] == 0
    assert report['coordinate_totals']['target_answer_changed'] == 0
    assert report['complement_totals']['known_answer_changes'] == 0
    assert report['hard_scientific_cutoff'] is False


@pytest.mark.parametrize('corruption',['missing','duplicate','replay','label'])
def test_missing_or_inconsistent_evidence_is_rejected(tiny_reader,corruption):
    rows = encode_worlds(tiny_reader, build_worlds())
    records = deepcopy(records_for(rows,lambda code,i:rows[code].queries[i].answer))
    if corruption == 'missing':
        records.pop()
    elif corruption == 'duplicate':
        records[1] = records[0]
    elif corruption == 'replay':
        next(r for r in records if r['condition']=='donor_replay')['generated_ids'] = [999]
    else:
        records[-1]['answer'] = 'invalid'
    with pytest.raises(ValueError):
        summarize_codes(records,rows,0)
