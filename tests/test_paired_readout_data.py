"""Conflicting native questions must prevent question-only perfect recall."""
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json

import pytest

from test_memory_updates import episode
from test_readout_runner import tiny_reader
from tinymem.research.paired_readout_data import audit_pairs, build_pairs, encode_pairs
from scripts.qualify_paired_readout import require_qualification


def test_pairs_replay_visible_facts_and_bound_question_only_accuracy():
    sources = [episode(i) for i in range(4)]
    rows = build_pairs(sources)
    assert len(rows) == 8
    assert rows == build_pairs(list(reversed(sources)))
    answers = defaultdict(Counter)
    for row in rows:
        state = {}
        for line in row.cases[0].context.splitlines():
            if line:
                words = line.split()
                state[words[0]] = words[-1][:-1]
        assert len(row.cases) == 10
        for q in row.cases:
            entity = q.question.removeprefix('Where is ').removesuffix('?')
            assert q.answer == state.get(entity, 'unknown')
            if q.category == 'update_known':
                answers[q.question][q.answer] += 1
    assert len(answers) == 32
    assert all(sorted(count.values()) == [1, 1] for count in answers.values())
    assert sum(max(count.values()) for count in answers.values()) == 32
    for left, right in zip(rows[::2], rows[1::2], strict=True):
        assert left.pair_id == right.pair_id
        assert [q.question for q in left.cases] == [q.question for q in right.cases]
        assert all(a.answer != b.answer for a, b in zip(left.cases[:8], right.cases[:8], strict=True))
    assert all(source == episode(i) for i, source in enumerate(sources))


def test_reject_corrupt_original_labels_and_duplicate_sources():
    sources = [episode(i) for i in range(4)]
    bad = replace(sources[0], before=(replace(sources[0].before[0], answer='unknown'), *sources[0].before[1:]))
    with pytest.raises(ValueError):
        build_pairs([bad, *sources[1:]])
    with pytest.raises(ValueError):
        build_pairs([sources[0]] * 4)


def test_native_read_inputs_match_and_prefix_leak_is_rejected(tiny_reader):
    rows = encode_pairs(tiny_reader, build_pairs([episode(i) for i in range(4)]))
    assert audit_pairs(rows)['known_question_only_ceiling'] == 0.5
    assert len({q.after_ids for row in rows for q in row.queries}) == 40
    bad = (replace(rows[0], before_ids=(*rows[0].before_ids, 1)), *rows[1:])
    with pytest.raises(ValueError, match='prefix'):
        audit_pairs(bad)
    first = rows[0].queries[0]
    bad = (replace(rows[0], queries=(replace(first, after_ids=(*first.after_ids, 1)), *rows[0].queries[1:])), *rows[1:])
    with pytest.raises(ValueError, match='question tokens'):
        audit_pairs(bad)
    bad = (*rows[:2], replace(rows[2], source_group_ids=rows[0].source_group_ids),
           replace(rows[3], source_group_ids=rows[0].source_group_ids), *rows[4:])
    with pytest.raises(ValueError, match='ownership overlaps'):
        audit_pairs(bad)


@pytest.mark.parametrize('claimed_pass', [False, True])
def test_failed_full_text_gate_cannot_authorize_training(tmp_path, claimed_pass):
    path = tmp_path / 'report.json'
    path.write_text(json.dumps({'status': 'complete', 'qualified': claimed_pass,
                               'known_correct': 60, 'known_total': 64, 'missing_correct': 16,
                               'missing_total': 16, 'training_steps': 0, 'reader_unchanged': True}))
    with pytest.raises(ValueError, match='positive qualification'):
        require_qualification(tmp_path, report_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
