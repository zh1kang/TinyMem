from dataclasses import replace

import pytest

from tinymem.research.storage_frontier_data import (
    StorageQuestion,
    dictionary_from_training,
    split_questions,
    write_records,
)


def questions():
    return tuple(StorageQuestion(f'{s}:{i}', str(s), 'qa1', 'train',
                                 ('Mary went home.',) * i, 'Where?', 'home')
                 for s in range(20) for i in (1, 2))


def test_whole_story_split_and_dictionary_ownership():
    train, dev = split_questions(questions(), seed=7)
    assert len(train) == 36 and len(dev) == 4
    assert {q.story for q in train}.isdisjoint({q.story for q in dev})
    assert dictionary_from_training(train) == ('Mary went home.',)
    with pytest.raises(ValueError, match='training text'):
        dictionary_from_training(dev)


def test_writer_stream_is_independent_of_question_and_answer():
    original = questions()[0]
    changed = replace(original, id='other', question='Other?', answer='elsewhere')
    for level in (0, 1, 2):
        assert write_records(original, ('a', 'b'), level=level, seed=3) == write_records(
            changed, ('a', 'b'), level=level, seed=3,
        )


def test_noise_preserves_all_facts_and_declared_delay():
    q = replace(questions()[0], records=tuple(f'fact {i}' for i in range(8)))
    stream = write_records(q, ('noise',), level=2, seed=3)
    assert tuple(s for s in stream if s != 'noise') == q.records
    assert stream[-16:] == ('noise',) * 16
    assert len(stream) == 32
