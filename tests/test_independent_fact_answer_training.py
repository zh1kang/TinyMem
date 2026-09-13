from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from test_readout_runner import tiny_reader
from tinymem.memory.readout_interface import ReadoutBridge
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.independent_fact_data import build_worlds, encode_worlds
from tinymem.research.independent_fact_updates import new_update_writer
from tinymem.research.prefix_reader import prefix_answer_loss


def _features(width):
    histories = {
        code: torch.arange((code % 3 + 1) * width, dtype=torch.float32).reshape(-1, width)
        for code in range(16)
    }
    events = {
        (fact, bit): torch.arange((fact + bit + 2) * width, dtype=torch.float32).reshape(-1, width)
        for fact in range(4)
        for bit in range(2)
    }
    return histories, events


def _stream():
    current = 0
    events = []
    for step, fact in enumerate((0, 1, 2, 3), start=1):
        new_bit = 1
        after = current | (1 << fact)
        events.append({
            "step": step,
            "before_code": current,
            "after_code": after,
            "target_fact": fact,
            "new_bit": new_bit,
            "action": "C",
            "text": "unused",
        })
        current = after
    from tinymem.research.independent_fact_data import ENTITIES, ROOM_PAIRS
    for event in events:
        event["text"] = f"{ENTITIES[event['target_fact']]} moved to the {ROOM_PAIRS[event['target_fact']][event['new_bit']]}."
    return [{"id": "answer-train-0", "initial_code": 0, "events": events}]


def _setup(tiny_reader, batch_size=1):
    from tinymem.research.independent_fact_recurrent_training import pack_recurrent_batch

    width = tiny_reader.model.get_input_embeddings().embedding_dim
    histories, events = _features(width)
    streams = _stream()
    if batch_size == 2:
        second = deepcopy(streams[0])
        second['id'] = 'answer-train-3'
        second['initial_code'] = current = 3
        from tinymem.research.independent_fact_data import ENTITIES, ROOM_PAIRS
        for step, (event, fact) in enumerate(zip(second['events'], (3, 2, 1, 0), strict=True), start=1):
            new_bit = 1 - ((current >> fact) & 1)
            after = (current & ~(1 << fact)) | (new_bit << fact)
            event.update(step=step, before_code=current, after_code=after, target_fact=fact,
                         new_bit=new_bit, text=f'{ENTITIES[fact]} moved to the {ROOM_PAIRS[fact][new_bit]}.')
            current = after
        streams.append(second)
    batch = pack_recurrent_batch(streams, histories, events)
    rows = encode_worlds(tiny_reader, build_worlds())
    tiny_reader.model.requires_grad_(False).eval()
    bridge = ReadoutBridge(width, "affine").eval()
    bridge.requires_grad_(False)
    return batch, rows, bridge


def _bridge_state(bridge, values, valid, index, device):
    return bridge(LatentSlotState(
        values[index:index + 1].clone().to(device),
        valid[index:index + 1].clone().to(device),
    ))


def _query_ce(reader, bridge, values, valid, index, row):
    device = reader.model.device
    memory = _bridge_state(bridge, values, valid, index, device)
    before = torch.tensor(row.before_ids, dtype=torch.long, device=device)
    return torch.stack([
        prefix_answer_loss(
            reader,
            before,
            memory,
            torch.tensor(query.after_ids, dtype=torch.long, device=device),
            torch.tensor(query.answer_ids, dtype=torch.long, device=device),
        )
        for query in row.queries
    ]).mean()


def _direct_objective(reader, bridge, writer, batch, rows, coordinate_weight, *, detach_between=False):
    initial = writer(writer.empty(16), batch.histories, batch.history_valid)
    state = LatentSlotState(
        initial.values.index_select(0, batch.initial_codes),
        initial.valid.index_select(0, batch.initial_codes),
    )
    updates = []
    for step in range(4):
        state = writer(state, batch.events[:, step], batch.event_valid[:, step])
        updates.append(state.values)
        if detach_between:
            state = LatentSlotState(state.values.detach(), state.valid)
    initial_ce = torch.stack([
        _query_ce(reader, bridge, initial.values, initial.valid, code, rows[code])
        for code in range(16)
    ]).mean()
    update_ce = torch.stack([
        _query_ce(reader, bridge, updates[step], state.valid, row, rows[int(batch.after_codes[row, step])])
        for step in range(4) for row in range(batch.events.shape[0])
    ]).mean()
    objective = 0.5 * initial_ce + 0.5 * update_ce
    initial_mse = torch.nn.functional.mse_loss(initial.values, batch.initial_targets)
    update_mse = torch.nn.functional.mse_loss(torch.stack(updates, dim=1), batch.after_targets)
    if coordinate_weight:
        objective = objective + 0.5 * coordinate_weight * (initial_mse + update_mse)
    return objective, initial_ce, update_ce, initial_mse, update_mse


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("coordinate_weight", [0, 1])
@pytest.mark.parametrize("query_microbatch", [1, 6])
def test_answer_training_matches_direct_ce_and_coordinate_oracle(tiny_reader, coordinate_weight, query_microbatch, batch_size):
    from tinymem.research.independent_fact_answer_training import train_answer_recurrent_batch

    batch, rows, bridge = _setup(tiny_reader, batch_size)
    writer = new_update_writer(tiny_reader.model.get_input_embeddings().embedding_dim, seed=41)
    oracle = deepcopy(writer)
    expected, initial_ce, update_ce, initial_mse, update_mse = _direct_objective(
        tiny_reader, bridge, oracle, batch, rows, coordinate_weight,
    )
    expected.backward()
    expected_norm = torch.nn.utils.clip_grad_norm_(tuple(oracle.parameters()), 1.0)
    result = train_answer_recurrent_batch(
        writer, batch, torch.optim.SGD(writer.parameters(), lr=0.0),
        reader=tiny_reader, bridge=bridge, worlds=rows, coordinate_weight=coordinate_weight,
        query_microbatch=query_microbatch,
    )
    assert result["answer_ce"] == pytest.approx(float((0.5 * initial_ce + 0.5 * update_ce).detach()), rel=1e-5)
    assert result["initial_answer_ce"] == pytest.approx(float(initial_ce.detach()), rel=1e-5)
    assert result["updated_answer_ce"] == pytest.approx(float(update_ce.detach()), rel=1e-5)
    if coordinate_weight:
        assert result["coordinate_mse"] == pytest.approx(float((0.5 * initial_mse + 0.5 * update_mse).detach()), rel=1e-5)
    else:
        assert result["coordinate_mse"] is None
    assert result["objective"] == pytest.approx(float(expected.detach()), rel=1e-5)
    assert result["gradient_norm"] == pytest.approx(float(expected_norm), rel=1e-5)
    for actual, expected_param in zip(writer.parameters(), oracle.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected_param.grad, rtol=2e-4, atol=2e-6)


def test_answer_weight_zero_ignores_coordinate_targets(tiny_reader):
    from tinymem.research.independent_fact_answer_training import train_answer_recurrent_batch

    batch, rows, bridge = _setup(tiny_reader)
    changed = replace(
        batch,
        initial_targets=torch.randn_like(batch.initial_targets),
        after_targets=torch.randn_like(batch.after_targets),
    )
    left = new_update_writer(16, seed=43)
    right = deepcopy(left)
    left_result = train_answer_recurrent_batch(
        left, batch, torch.optim.SGD(left.parameters(), lr=0.0),
        reader=tiny_reader, bridge=bridge, worlds=rows, coordinate_weight=0,
    )
    right_result = train_answer_recurrent_batch(
        right, changed, torch.optim.SGD(right.parameters(), lr=0.0),
        reader=tiny_reader, bridge=bridge, worlds=rows, coordinate_weight=0,
        query_microbatch=1,
    )
    assert left_result["objective"] == pytest.approx(right_result["objective"], abs=1e-6)
    for actual, expected in zip(left.parameters(), right.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-4, atol=2e-6)


def test_answer_training_rejects_ownership_and_unfrozen_read_modules(tiny_reader):
    from tinymem.research.independent_fact_answer_training import train_answer_recurrent_batch

    batch, rows, bridge = _setup(tiny_reader)
    writer = new_update_writer(16, seed=47)
    with pytest.raises(ValueError, match="coordinate_weight"):
        train_answer_recurrent_batch(
            writer, batch, torch.optim.SGD(writer.parameters(), lr=0.0),
            reader=tiny_reader, bridge=bridge, worlds=rows, coordinate_weight=2,
        )
    with pytest.raises(ValueError, match="optimizer"):
        train_answer_recurrent_batch(
            writer, batch, torch.optim.SGD(tuple(writer.parameters())[:-1], lr=0.0),
            reader=tiny_reader, bridge=bridge, worlds=rows, coordinate_weight=0,
        )
    bridge.requires_grad_(True)
    with pytest.raises(ValueError, match="bridge"):
        train_answer_recurrent_batch(
            writer, batch, torch.optim.SGD(writer.parameters(), lr=0.0),
            reader=tiny_reader, bridge=bridge, worlds=rows, coordinate_weight=0,
        )


def test_answer_training_keeps_reader_bridge_frozen_and_reports_counts(tiny_reader):
    from tinymem.research.independent_fact_answer_training import train_answer_recurrent_batch

    batch, rows, bridge = _setup(tiny_reader)
    writer = new_update_writer(16, seed=53)
    reader_before = deepcopy(tiny_reader.model.state_dict())
    bridge_before = deepcopy(bridge.state_dict())
    result = train_answer_recurrent_batch(
        writer, batch, torch.optim.AdamW(writer.parameters(), lr=1e-3),
        reader=tiny_reader, bridge=bridge, worlds=rows, coordinate_weight=0,
    )
    assert result["history_examples"] == 16
    assert result["event_examples"] == 4
    assert result["writer_examples"] == 20
    assert result["coordinate_weight"] == 0
    assert result["gradient_norm"] >= 0
    assert all(torch.equal(value, reader_before[name]) for name, value in tiny_reader.model.state_dict().items())
    assert all(torch.equal(value, bridge_before[name]) for name, value in bridge.state_dict().items())
    assert all(parameter.grad is None for parameter in tiny_reader.model.parameters())
    assert all(parameter.grad is None for parameter in bridge.parameters())
    assert all(torch.isfinite(parameter).all() for parameter in writer.parameters())


def test_direct_oracle_detects_detached_recurrence(tiny_reader):
    batch, rows, bridge = _setup(tiny_reader)
    complete = new_update_writer(16, seed=41)
    detached = deepcopy(complete)
    _direct_objective(tiny_reader, bridge, complete, batch, rows, 0)[0].backward()
    _direct_objective(tiny_reader, bridge, detached, batch, rows, 0, detach_between=True)[0].backward()
    complete_gradient = torch.cat([p.grad.flatten() for p in complete.parameters()])
    detached_gradient = torch.cat([p.grad.flatten() for p in detached.parameters()])
    assert (complete_gradient - detached_gradient).norm() > 1e-5


def test_writer_finishes_query_blind_trajectory_before_any_read(tiny_reader):
    from tinymem.research.independent_fact_answer_training import train_answer_recurrent_batch

    batch, rows, bridge = _setup(tiny_reader)
    writer = new_update_writer(16, seed=41)
    states = []
    capture = writer.register_forward_hook(lambda module, args, output: states.append(output.values.detach().clone()))
    def check_complete(module, args):
        assert len(states) == 5
    check = tiny_reader.model.register_forward_pre_hook(check_complete)
    try:
        train_answer_recurrent_batch(writer, batch, torch.optim.SGD(writer.parameters(), lr=0),
            reader=tiny_reader, bridge=bridge, worlds=rows, coordinate_weight=0)
    finally:
        capture.remove()
        check.remove()
    first = states.copy()
    states.clear()
    changed_rows = tuple(replace(row, queries=tuple(replace(q, after_ids=q.after_ids + (2,)) for q in row.queries)) for row in rows)
    capture = writer.register_forward_hook(lambda module, args, output: states.append(output.values.detach().clone()))
    try:
        train_answer_recurrent_batch(writer, batch, torch.optim.SGD(writer.parameters(), lr=0),
            reader=tiny_reader, bridge=bridge, worlds=changed_rows, coordinate_weight=0)
    finally:
        capture.remove()
    assert len(first) == len(states) == 5
    assert all(torch.equal(a, b) for a, b in zip(first, states, strict=True))
