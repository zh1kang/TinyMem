from copy import deepcopy

import pytest
import torch

from tinymem.research.independent_fact_data import ENTITIES, ROOM_PAIRS
from tinymem.research.independent_fact_placement import placement_state


def _features(width=16):
    return {
        (fact, bit): torch.arange((fact + bit + 2) * width, dtype=torch.float32).reshape(-1, width)
        for fact in range(4)
        for bit in range(2)
    }


def test_update_cases_cover_ordered_corrections_and_repetitions():
    from tinymem.research.independent_fact_updates import build_update_cases

    cases = build_update_cases()
    assert len(cases) == 128
    assert [case.case_id for case in cases[:4]] == [
        "code-00:fact-0:value-0", "code-00:fact-0:value-1",
        "code-00:fact-1:value-0", "code-00:fact-1:value-1",
    ]
    assert [case.case_id for case in cases[-2:]] == [
        "code-15:fact-3:value-0", "code-15:fact-3:value-1",
    ]
    assert sum(case.split == "train" for case in cases) == 64
    assert sum(case.split == "heldout" for case in cases) == 64
    assert sum(case.kind == "correction" for case in cases) == 64
    assert sum(case.kind == "repetition" for case in cases) == 64
    for case in cases:
        assert case.event_text == (
            f"{ENTITIES[case.target_fact]} moved to the "
            f"{ROOM_PAIRS[case.target_fact][case.new_bit]}."
        )
        assert case.after_code == (
            case.before_code & ~(1 << case.target_fact)
        ) | (case.new_bit << case.target_fact)
        expected_split = (
            "train" if (case.before_code & ~(1 << case.target_fact)).bit_count() % 2 == 0 else "heldout"
        )
        assert case.split == expected_split
        expected_kind = "repetition" if ((case.before_code >> case.target_fact) & 1) == case.new_bit else "correction"
        assert case.kind == expected_kind


def test_pack_update_batch_repeats_features_and_masks_padding():
    from tinymem.research.independent_fact_updates import build_update_cases, pack_update_batch

    cases = build_update_cases()[:3]
    features = _features()
    old, hidden, valid, target = pack_update_batch(cases, features)
    assert hidden.shape == (3, 3, 16)
    assert valid.shape == (3, 3)
    assert valid.dtype == torch.bool and valid.tolist() == [[True, True, False], [True, True, True], [True, True, True]]
    assert all(tensor.device.type == "cpu" and tensor.dtype == torch.float32 for tensor in (old.values, hidden, target))
    for row, case in enumerate(cases):
        expected_old = placement_state(case.before_code, torch.device("cpu"), "separate_fact0")
        expected_target = placement_state(case.after_code, torch.device("cpu"), "separate_fact0")
        torch.testing.assert_close(old.values[row], expected_old.values[0])
        torch.testing.assert_close(target[row], expected_target.values[0])
        expected = features[(case.target_fact, case.new_bit)]
        torch.testing.assert_close(hidden[row, : len(expected)], expected)
        assert valid[row, len(expected):].logical_not().all()
    assert old.nbytes // len(cases) == 66


def test_batch_training_matches_independent_serial_loss_and_gradients():
    from tinymem.research.independent_fact_updates import (
        build_update_cases, new_update_writer, pack_update_batch, train_update_batch,
    )

    cases = build_update_cases()[:5]
    batch = pack_update_batch(cases, _features())
    writer = new_update_writer(16, seed=19)
    serial = deepcopy(writer)
    serial_outputs = []
    for row in range(len(cases)):
        state = type(batch[0])(batch[0].values[row:row + 1], batch[0].valid[row:row + 1])
        serial_outputs.append(serial(state, batch[1][row:row + 1], batch[2][row:row + 1]).values[0])
    expected_output = torch.stack(serial_outputs)
    expected_loss = torch.nn.functional.mse_loss(expected_output, batch[3])
    expected_loss.backward()
    expected_norm = torch.nn.utils.clip_grad_norm_(tuple(serial.parameters()), 1.0)
    result = train_update_batch(writer, batch, torch.optim.SGD(writer.parameters(), lr=0.0))
    assert result["state_mse"] == pytest.approx(float(expected_loss.detach()))
    assert result["gradient_norm"] == pytest.approx(float(expected_norm))
    for actual, expected in zip(writer.parameters(), serial.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-7)
    output = writer(batch[0], batch[1], batch[2])
    assert output.valid.all() and output.nbytes // len(cases) == 66


def test_training_rejects_optimizer_omitting_writer_parameter_and_preserves_inputs():
    from tinymem.research.independent_fact_updates import (
        build_update_cases, new_update_writer, pack_update_batch, train_update_batch,
    )

    cases = build_update_cases()[:2]
    features = _features()
    batch = pack_update_batch(cases, features)
    before = tuple(value.clone() for value in features.values())
    writer = new_update_writer(16, seed=7)
    with pytest.raises(ValueError, match="optimizer"):
        train_update_batch(writer, batch, torch.optim.SGD(tuple(writer.parameters())[:-1], lr=0.01))
    train_update_batch(writer, batch, torch.optim.SGD(writer.parameters(), lr=0.01))
    assert all(torch.equal(features[key], value) for key, value in zip(features, before, strict=True))
    assert torch.equal(batch[0].valid, torch.ones_like(batch[0].valid))


def test_events_replay_truth_and_hold_reverse_directions_together():
    from tinymem.data.memory_updates import replay_update_chunks
    from tinymem.research.independent_fact_data import build_worlds
    from tinymem.research.independent_fact_updates import build_update_cases

    worlds = build_worlds()
    groups = {}
    for case in build_update_cases():
        before = replay_update_chunks((worlds[case.before_code].cases[0].context,))
        after = replay_update_chunks((worlds[case.before_code].cases[0].context, case.event_text))
        expected = replay_update_chunks((worlds[case.after_code].cases[0].context,))
        assert after == expected
        assert sum(before[k] != after[k] for k in before) == (case.kind == 'correction')
        context = tuple(before[ENTITIES[i]] for i in range(4) if i != case.target_fact)
        groups.setdefault((case.target_fact, context), []).append(case)
    assert len(groups) == 32
    assert all(len(group) == 4 and len({c.split for c in group}) == 1 for group in groups.values())


def test_batch_rejects_text_label_disagreement():
    from dataclasses import replace
    from tinymem.research.independent_fact_updates import build_update_cases, pack_update_batch

    case = build_update_cases()[0]
    with pytest.raises(ValueError, match='event text'):
        pack_update_batch([replace(case, event_text='different fact sentence')], _features())
