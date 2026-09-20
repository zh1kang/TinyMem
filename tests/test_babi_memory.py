from dataclasses import replace
import itertools

import pytest
import torch

from tinymem.data.babi import parse_babi_lines
from tinymem.research.babi_memory_data import (
    Vocabulary, fit_vocabulary, query_entity, replay_locations, split_training, story_id, validate_example,
)
from tinymem.research.babi_memory_state import (
    oracle_locations, oracle_state, packed_answer, packed_size, packed_update, unpack_locations,
)


@pytest.fixture
def vocabulary():
    return Vocabulary(('Daniel', 'John', 'Mary', 'Sandra'),
                      ('bathroom', 'bedroom', 'garden', 'hallway', 'kitchen', 'office'))


def test_official_questions_preserve_story_and_fact_line_gaps():
    rows = parse_babi_lines(['1 Mary moved to the bathroom.', '2 John went to the kitchen.',
                            '3 Where is Mary?\tbathroom\t1', '4 Mary travelled to the office.',
                            '5 Where is Mary?\toffice\t4'],
                           task_id='qa1', split='train', source_name='qa1_train.txt')
    assert story_id(rows[0]) == story_id(rows[1])
    assert rows[1].context_fact_ids == (1, 2, 4)
    assert len(rows[1].context.splitlines()) == 3
    vocab = fit_vocabulary(rows)
    validate_example(rows[1], vocab)
    with pytest.raises(ValueError, match='official answer'):
        validate_example(replace(rows[1], answer='bathroom'), vocab)


def test_story_split_never_shares_overlapping_prefixes():
    lines = ['1 Mary went to the kitchen.', '2 Where is Mary?\tkitchen\t1',
             '3 Mary moved to the office.', '4 Where is Mary?\toffice\t3'] * 10
    rows = parse_babi_lines(lines, task_id='qa1', split='train', source_name='qa1_train.txt')
    training, validation = split_training(rows, validation_stories=2, seed=9)
    assert len(training) == 16 and len(validation) == 4
    assert {story_id(e) for e in training}.isdisjoint({story_id(e) for e in validation})
    assert split_training(rows, validation_stories=2, seed=9) == (training, validation)
    with pytest.raises(ValueError, match='training data'):
        split_training(validation, validation_stories=1, seed=9)
    with pytest.raises(ValueError, match='training examples'):
        fit_vocabulary(validation)


def test_all_entity_room_corrections_preserve_other_facts(vocabulary):
    for entity, first, last in itertools.product(vocabulary.entities, vocabulary.rooms, vocabulary.rooms):
        history = [f'{name} moved to the bathroom.' for name in vocabulary.entities]
        history += [f'{entity} went to the {first}.', f'{entity} travelled to the {last}.']
        expected = tuple(vocabulary.rooms.index(last) + 1 if name == entity else 1
                         for name in vocabulary.entities)
        state = oracle_state(history, vocabulary)
        assert oracle_locations(state, vocabulary) == expected
        assert state.nbytes == 258
        packed = bytes(packed_size(vocabulary))
        for text in history:
            packed = packed_update(packed, text, vocabulary)
        assert len(packed) == 2
        assert unpack_locations(packed, vocabulary) == expected
        assert packed_answer(packed, f'Where is {entity}?', vocabulary) == last


def test_long_repetitions_and_unknown_entities(vocabulary):
    history = ['John went to the office.'] + ['Mary moved to the kitchen.'] * 512
    state = oracle_state(history, vocabulary)
    assert oracle_locations(state, vocabulary) == (0, 6, 5, 0)
    assert replay_locations(history, vocabulary) == (0, 6, 5, 0)
    initial = bytes(packed_size(vocabulary))
    assert packed_answer(initial, 'Where is Mary?', vocabulary) == 'unknown'
    assert query_entity('Where is Mary?', vocabulary) == 2
    for text in ('Alice went to the kitchen.', 'Mary went to the airport.', 'irrelevant prose'):
        with pytest.raises(ValueError):
            oracle_state([text], vocabulary)


def test_oracle_rejects_ambiguous_or_nonfinite_states(vocabulary):
    state = oracle_state(['Mary went to the office.'], vocabulary)
    state.values[0, 0, 16] = state.values[0, 0, 21]
    with pytest.raises(ValueError, match='ambiguous'):
        oracle_locations(state, vocabulary)
    state.values[0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        oracle_locations(state, vocabulary)


def test_packed_rejects_unused_codes_and_padding(vocabulary):
    with pytest.raises(ValueError, match='room code'):
        unpack_locations(bytes([7, 0]), vocabulary)
    with pytest.raises(ValueError, match='padding'):
        unpack_locations(bytes([0, 128]), vocabulary)
    with pytest.raises(ValueError, match='byte count'):
        unpack_locations(bytes([0]), vocabulary)


def test_state_uses_history_only_and_owns_storage(vocabulary):
    history = ['John went to the office.', 'Mary went to the garden.']
    state = oracle_state(history, vocabulary)
    assert state.values._base is None
    assert state.valid._base is None
    assert state.values.is_contiguous() and state.valid.is_contiguous()
    assert not state.values.requires_grad
    before = state.values.clone()
    for question in ('Where is John?', 'Where is Mary?'):
        query_entity(question, vocabulary)
    assert torch.equal(state.values, before)
    assert torch.equal(oracle_state(history, vocabulary).values, before)
