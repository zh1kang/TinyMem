"""Independent facts, balanced splits, and equal non-memory native inputs."""
from collections import Counter
from itertools import combinations
from dataclasses import replace

import pytest
import torch

from test_readout_runner import tiny_reader
from tinymem.data.memory_updates import replay_update_chunks
from tinymem.research.independent_fact_data import build_worlds, encode_worlds, oracle_state


def test_worlds_cover_independent_facts_and_balanced_parity_halves():
    worlds = build_worlds()
    assert [w.code for w in worlds] == list(range(16))
    for world in worlds:
        assert len(world.cases) == 6
        truth = replay_update_chunks((world.cases[0].context,))
        assert len(truth) == 4
        for q in world.cases:
            entity = q.question.removeprefix('Where is ').removesuffix('?')
            assert q.answer == truth.get(entity, 'unknown')
    for parity in (0, 1):
        codes = [w.code for w in worlds if w.code.bit_count() % 2 == parity]
        assert len(codes) == 8
        for size in (1, 2, 3):
            for axes in combinations(range(4), size):
                counts = Counter(tuple((code >> axis) & 1 for axis in axes) for code in codes)
                assert len(counts) == 2 ** size and set(counts.values()) == {8 // 2 ** size}
        for query in range(4):
            assert sorted(Counter(worlds[c].cases[query].answer for c in codes).values()) == [4, 4]
    for code in range(16):
        for axis in range(4):
            donor = code ^ (1 << axis)
            changed = [j for j in range(6) if worlds[code].cases[j].answer != worlds[donor].cases[j].answer]
            assert changed == [axis]


def test_oracle_states_have_four_owned_bits_and_no_other_values():
    values = []
    for code in range(16):
        state = oracle_state(code, torch.device('cpu'))
        assert state.nbytes == 66 and state.valid.all() and not state.values.requires_grad
        expected = torch.zeros(1, 2, 8)
        expected[0, 0, :4] = torch.tensor([1 if code & (1 << i) else -1 for i in range(4)])
        assert torch.equal(state.values, expected)
        values.append(state.values.numpy().tobytes())
    assert len(set(values)) == 16
    original, another = oracle_state(0, torch.device('cpu')), oracle_state(0, torch.device('cpu'))
    original.values.zero_()
    assert another.values.count_nonzero() == 4
    for code in (-1, 16, True, 1.0):
        with pytest.raises(ValueError):
            oracle_state(code, torch.device('cpu'))


def test_native_queries_and_fixed_prefix_are_equal_across_all_codes(tiny_reader):
    worlds = build_worlds()
    rows = encode_worlds(tiny_reader, worlds)
    assert len(rows) == 16 and len({r.before_ids for r in rows}) == 1
    assert len({r.history_ids for r in rows}) == 16
    for j in range(6):
        assert len({r.queries[j].after_ids for r in rows}) == 1
    wrong = list(worlds)
    wrong[0] = replace(wrong[0], cases=(replace(wrong[0].cases[0], answer='office'), *wrong[0].cases[1:]))
    with pytest.raises(ValueError):
        encode_worlds(tiny_reader, wrong)
    tiny_reader.model.config.max_position_embeddings = 8
    with pytest.raises(ValueError):
        encode_worlds(tiny_reader, worlds)


def test_reference_target_miss_keeps_counts_and_requires_interpretation(tiny_reader):
    from tinymem.research.independent_fact_evaluation import summarize_reference

    rows = encode_worlds(tiny_reader, build_worlds())
    records = [dict(case_id=q.case_id, history_id=row.history_id, code=row.code, category=q.category,
                    answer=q.answer, prediction=q.answer, condition='full_text', correct=True)
               for row in rows for q in row.queries]
    for r in records[:6]:
        r.update(prediction='wrong', correct=False)
    report = summarize_reference(records, rows)
    assert report['scores']['all']['known']['correct'] == 60
    assert not report['scores']['all']['known']['reliability_target_met']
    assert report['scientific_decision'] == 'interpret_reference_errors_before_training'
    assert report['hard_scientific_cutoff'] is False
    with pytest.raises(ValueError, match='coverage'):
        summarize_reference(records[:-1], rows)


def test_explicit_nearest_neighbor_reference_copies_three_of_four_facts():
    from tinymem.research.independent_fact_data import nearest_training_code

    worlds = build_worlds()
    for parity in (0, 1):
        correct = 0
        for code in range(16):
            donor = nearest_training_code(code, parity)
            assert donor.bit_count() % 2 == parity
            if code.bit_count() % 2 == parity:
                assert donor == code
                continue
            assert (code ^ donor).bit_count() == 1
            matches = sum(worlds[code].cases[i].answer == worlds[donor].cases[i].answer for i in range(4))
            assert matches == 3
            correct += matches
        assert correct == 24
