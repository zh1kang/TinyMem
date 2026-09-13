"""Independent scalar-loss oracles for normalized correction weighting."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from test_readout_runner import tiny_reader
from test_independent_fact_answer_training import _setup, _features, _bridge_state
from test_independent_fact_joint_update import teacher_states
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.independent_fact_updates import new_update_writer
from tinymem.research.prefix_reader import prefix_answer_loss


def test_changed_query_maps_every_bit_and_repetition():
    from tinymem.research.independent_fact_correction_weight import correction_query

    for code in range(16):
        assert correction_query(code, code) is None
        for fact in range(4):
            assert correction_query(code, code ^ (1 << fact)) == fact
    for before, after in ((0, 3), (0, 16), (-1, 0), (True, 1)):
        with pytest.raises(ValueError):
            correction_query(before, after)


def direct_objective(reader, bridge, writer, batch, worlds, teacher, multiplier, weight):
    state = LatentSlotState(teacher.values[batch.initial_codes], teacher.valid[batch.initial_codes])
    states, losses, unweighted = [], [], []
    device = reader.model.device
    for step in range(4):
        state = writer(state, batch.events[:, step], batch.event_valid[:, step])
        states.append(state.values)
        for row in range(batch.events.shape[0]):
            code = int(batch.after_codes[row, step])
            world = worlds[code]
            memory = _bridge_state(bridge, state.values, state.valid, row, device)
            per_query = torch.stack([
                prefix_answer_loss(reader, torch.tensor(world.before_ids, device=device), memory,
                    torch.tensor(query.after_ids, device=device),
                    torch.tensor(query.answer_ids, device=device))
                for query in world.queries
            ])
            changed = [q for q in range(4)
                       if ((int(batch.before_codes[row, step]) >> q) & 1) != ((code >> q) & 1)]
            weights = torch.ones(6, device=device)
            if changed:
                weights[changed[0]] = multiplier
            losses.append((per_query * weights / weights.sum()).sum())
            unweighted.append(per_query.mean())
    ce = torch.stack(losses).mean()
    mse = (torch.stack(states, 1) - teacher.values[batch.after_codes]).square().mean()
    return ce + weight * mse, ce, torch.stack(unweighted).mean()


@pytest.mark.parametrize('multiplier', [1., 2.])
def test_matches_independent_loss_gradients_and_adamw(tiny_reader, multiplier):
    from tinymem.research.independent_fact_correction_weight import train_correction_weight_batch

    batch, worlds, bridge = _setup(tiny_reader, 2)
    teacher = teacher_states(batch)
    writer = new_update_writer(16, 151)
    oracle = deepcopy(writer)
    expected, ce, unweighted = direct_objective(tiny_reader, bridge, oracle, batch, worlds, teacher, multiplier, .25)
    expected.backward()
    norm = torch.nn.utils.clip_grad_norm_(oracle.parameters(), 1.)
    torch.optim.AdamW(oracle.parameters(), lr=.001, weight_decay=.01).step()
    result = train_correction_weight_batch(writer, batch,
        torch.optim.AdamW(writer.parameters(), lr=.001, weight_decay=.01),
        reader=tiny_reader, bridge=bridge, worlds=worlds, targets=teacher,
        state_weight=.25, correction_multiplier=multiplier)
    assert result['objective'] == pytest.approx(float(expected.detach()), rel=1e-5)
    assert result['answer_ce'] == pytest.approx(float(ce.detach()), rel=1e-5)
    assert result['unweighted_answer_ce'] == pytest.approx(float(unweighted.detach()), rel=1e-5)
    assert result['gradient_norm'] == pytest.approx(float(norm), rel=1e-5)
    assert result['correction_event_count'] == 8
    assert result['repetition_event_count'] == 0
    assert result['repetition_known_ce_sum'] == result['repetition_absent_ce_sum'] == 0
    for actual, reference in zip(writer.parameters(), oracle.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, reference.grad, rtol=2e-4, atol=2e-6)
        torch.testing.assert_close(actual, reference, rtol=2e-4, atol=2e-6)


def mixed_batch(reader, actions):
    from tinymem.research.independent_fact_data import ENTITIES, ROOM_PAIRS
    from tinymem.research.independent_fact_recurrent_training import pack_recurrent_batch

    current, events = 0, []
    for step, (fact, action) in enumerate(zip(range(4), actions, strict=True), 1):
        bit = 1 if action == 'C' else 0
        after = current | (bit << fact)
        events.append({'step': step, 'before_code': current, 'after_code': after,
            'target_fact': fact, 'new_bit': bit, 'action': action,
            'text': f'{ENTITIES[fact]} moved to the {ROOM_PAIRS[fact][bit]}.'})
        current = after
    histories, features = _features(reader.model.get_input_embeddings().embedding_dim)
    return pack_recurrent_batch([{'id': 'mixed', 'initial_code': 0, 'events': events}], histories, features)


@pytest.mark.parametrize('actions', ['RRRR', 'CRCR'])
def test_repetitions_and_diagnostic_sums_match_scalar_loss(tiny_reader, actions):
    from tinymem.research.independent_fact_correction_weight import train_correction_weight_batch

    _, worlds, bridge = _setup(tiny_reader)
    batch = mixed_batch(tiny_reader, actions)
    teacher = teacher_states(batch)
    writer = new_update_writer(16, 213)
    expected, ce, plain = direct_objective(tiny_reader, bridge, writer, batch, worlds, teacher, 2., .25)
    result = train_correction_weight_batch(writer, batch, torch.optim.AdamW(writer.parameters()),
        reader=tiny_reader, bridge=bridge, worlds=worlds, targets=teacher,
        state_weight=.25, correction_multiplier=2., measure_only=True)
    assert result['answer_ce'] == pytest.approx(float(ce.detach()), rel=1e-5)
    assert result['unweighted_answer_ce'] == pytest.approx(float(plain.detach()), rel=1e-5)
    assert result['correction_event_count'] == actions.count('C')
    assert result['repetition_event_count'] == actions.count('R')
    correction = sum(result[k] for k in ('correction_target_ce_sum', 'correction_untouched_ce_sum', 'correction_absent_ce_sum'))
    repetition = result['repetition_known_ce_sum'] + result['repetition_absent_ce_sum']
    assert (correction + repetition) / 24 == pytest.approx(result['unweighted_answer_ce'])
    assert ((correction + result['correction_target_ce_sum']) / 7 + repetition / 6) / 4 == pytest.approx(result['answer_ce'])


def test_multiplier_one_reproduces_existing_kernel_and_frozen_ownership(tiny_reader):
    from tinymem.research.independent_fact_correction_weight import train_correction_weight_batch
    from tinymem.research.independent_fact_joint_update import train_joint_update_batch

    _, worlds, bridge = _setup(tiny_reader)
    batch = mixed_batch(tiny_reader, 'CRCR')
    teacher = teacher_states(batch)
    writer = new_update_writer(16, 231)
    oracle = deepcopy(writer)
    before = {k: v.clone() for k, v in batch.__dict__.items() if isinstance(v, torch.Tensor)}
    teacher_before = teacher.values.clone()
    reader_before = deepcopy(tiny_reader.model.state_dict())
    bridge_before = deepcopy(bridge.state_dict())
    kwargs = dict(reader=tiny_reader, bridge=bridge, worlds=worlds, targets=teacher, state_weight=.25)
    result = train_correction_weight_batch(writer, batch, torch.optim.AdamW(writer.parameters(), lr=.001), **kwargs)
    old = train_joint_update_batch(oracle, batch, torch.optim.AdamW(oracle.parameters(), lr=.001), **kwargs)
    assert all(result[k] == v for k, v in old.items())
    assert all(torch.equal(a, b) for a, b in zip(writer.parameters(), oracle.parameters(), strict=True))
    assert all(torch.equal(v, getattr(batch, k)) for k, v in before.items())
    assert torch.equal(teacher.values, teacher_before)
    assert all(torch.equal(v, reader_before[k]) for k, v in tiny_reader.model.state_dict().items())
    assert all(torch.equal(v, bridge_before[k]) for k, v in bridge.state_dict().items())
    assert all(p.grad is None for p in (*tiny_reader.model.parameters(), *bridge.parameters()))


def test_recurrence_uses_only_previous_output_and_event_features(tiny_reader):
    from tinymem.research.independent_fact_correction_weight import train_correction_weight_batch

    batch, worlds, bridge = _setup(tiny_reader)
    teacher = teacher_states(batch)
    writer = new_update_writer(16, 241)
    other = deepcopy(writer)
    inputs, outputs = [], []
    def observe(module, args, output):
        inputs.append(args[0].values.detach().clone())
        outputs.append(output.values.detach().clone())
    handle = writer.register_forward_hook(observe)
    kwargs = dict(reader=tiny_reader, bridge=bridge, worlds=worlds, targets=teacher,
                  state_weight=.25, correction_multiplier=2., measure_only=True)
    before = deepcopy(writer.state_dict())
    optimizer = torch.optim.AdamW(writer.parameters())
    train_correction_weight_batch(writer, batch, optimizer, **kwargs)
    handle.remove()
    assert not optimizer.state
    assert all(torch.equal(v, before[k]) for k, v in writer.state_dict().items())
    assert len(inputs) == 4 and torch.equal(inputs[0], teacher.values[batch.initial_codes])
    assert all(torch.equal(inputs[s], outputs[s-1]) for s in range(1, 4))
    # Labels can change supervision but must not change any forward state.
    changed = replace(batch, after_codes=batch.before_codes.clone(),
        histories=torch.randn_like(batch.histories), after_targets=torch.randn_like(batch.after_targets))
    new_outputs = []
    handle = other.register_forward_hook(lambda module, args, output: new_outputs.append(output.values.detach().clone()))
    train_correction_weight_batch(other, changed, torch.optim.AdamW(other.parameters()), **kwargs)
    handle.remove()
    assert all(torch.equal(a, b) for a, b in zip(outputs, new_outputs, strict=True))


def test_rejects_invalid_multiplier_and_multiple_changed_facts(tiny_reader):
    from tinymem.research.independent_fact_correction_weight import train_correction_weight_batch

    batch, worlds, bridge = _setup(tiny_reader)
    teacher = teacher_states(batch)
    writer = new_update_writer(16, 251)
    optimizer = torch.optim.AdamW(writer.parameters())
    kwargs = dict(reader=tiny_reader, bridge=bridge, worlds=worlds, targets=teacher, state_weight=.25)
    for multiplier in (0, -1, float('nan'), float('inf'), True):
        with pytest.raises(ValueError, match='correction_multiplier'):
            train_correction_weight_batch(writer, batch, optimizer, correction_multiplier=multiplier, **kwargs)
    after = batch.after_codes.clone()
    after[0, 0] = 3
    with pytest.raises(ValueError, match='exactly one fact'):
        train_correction_weight_batch(writer, replace(batch, after_codes=after), optimizer, **kwargs)
    assert not optimizer.state
