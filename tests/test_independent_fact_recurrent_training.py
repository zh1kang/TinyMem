from copy import deepcopy

import pytest
import torch

from tinymem.research.independent_fact_data import ENTITIES, ROOM_PAIRS
from tinymem.research.independent_fact_placement import placement_state
from tinymem.research.independent_fact_updates import new_update_writer


def _features(width=8):
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


def _stream_dict(initial_code, order=(0, 1, 2, 3), actions=("C", "R", "C", "R")):
    current = initial_code
    events = []
    for step, (fact, action) in enumerate(zip(order, actions, strict=True), start=1):
        old_bit = (current >> fact) & 1
        new_bit = old_bit if action == "R" else 1 - old_bit
        after = (current & ~(1 << fact)) | (new_bit << fact)
        events.append({
            "step": step, "before_code": current, "after_code": after,
            "target_fact": fact, "new_bit": new_bit, "action": action,
            "text": f"{ENTITIES[fact]} moved to the {ROOM_PAIRS[fact][new_bit]}.",
        })
        current = after
    return {"id": f"test-{initial_code}", "initial_code": initial_code, "events": events}


def _batch():
    from tinymem.research.independent_fact_recurrent_training import pack_recurrent_batch

    histories, events = _features()
    streams = [_stream_dict(0), _stream_dict(3, order=(3, 2, 1, 0), actions=("C", "C", "R", "R"))]
    return pack_recurrent_batch(streams, histories, events)


def _recurrent_loss(writer, batch, *, detach_between=False):
    initial = writer(writer.empty(16), batch.histories, batch.history_valid)
    states = []
    state = initial.values.index_select(0, batch.initial_codes)
    valid = initial.valid.index_select(0, batch.initial_codes)
    for step in range(4):
        state_in = type(initial)(state, valid)
        state_out = writer(state_in, batch.events[:, step], batch.event_valid[:, step])
        states.append(state_out.values)
        state = state_out.values.detach() if detach_between else state_out.values
        valid = state_out.valid
    updates = torch.stack(states, dim=1)
    return 0.5 * torch.nn.functional.mse_loss(initial.values, batch.initial_targets) + 0.5 * torch.nn.functional.mse_loss(updates, batch.after_targets)


def _reset_loss(writer, batch):
    initial = writer(writer.empty(16), batch.histories, batch.history_valid)
    updates = []
    for step in range(4):
        state = type(initial)(
            initial.values.index_select(0, batch.before_codes[:, step]),
            initial.valid.index_select(0, batch.before_codes[:, step]),
        )
        updates.append(writer(state, batch.events[:, step], batch.event_valid[:, step]).values)
    return 0.5 * torch.nn.functional.mse_loss(initial.values, batch.initial_targets) + 0.5 * torch.nn.functional.mse_loss(torch.stack(updates, dim=1), batch.after_targets)


def test_pack_recurrent_batch_contains_all_initializers_and_four_event_steps():
    batch = _batch()
    assert batch.histories.shape[0] == 16
    assert batch.events.shape[0:2] == (2, 4)
    assert batch.initial_codes.tolist() == [0, 3]
    assert batch.before_codes.shape == (2, 4)
    assert batch.after_codes.shape == (2, 4)
    assert batch.initial_targets.shape == (16, 2, 8)
    assert batch.after_targets.shape == (2, 4, 2, 8)
    for tensor in (batch.histories, batch.events, batch.initial_targets, batch.after_targets):
        assert tensor.device.type == "cpu" and tensor.dtype == torch.float32
        assert not tensor.requires_grad and torch.isfinite(tensor).all()
    assert batch.history_valid.dtype == torch.bool and batch.event_valid.dtype == torch.bool


def test_targets_use_separate_fact0_layout_for_all_sixteen_worlds():
    batch = _batch()
    expected = torch.stack([placement_state(code, torch.device("cpu"), "separate_fact0").values[0] for code in range(16)])
    torch.testing.assert_close(batch.initial_targets, expected)
    torch.testing.assert_close(batch.before_targets[:, 0], expected.index_select(0, batch.before_codes[:, 0]))


def test_recurrent_training_matches_independent_unroll_and_full_gradient():
    from tinymem.research.independent_fact_recurrent_training import train_recurrent_batch

    batch = _batch()
    writer = new_update_writer(8, seed=17)
    oracle = deepcopy(writer)
    expected = _recurrent_loss(oracle, batch)
    expected.backward()
    expected_norm = torch.nn.utils.clip_grad_norm_(tuple(oracle.parameters()), 1.0)
    result = train_recurrent_batch(writer, batch, torch.optim.SGD(writer.parameters(), lr=0.0), arm="recurrent")
    assert result["state_mse"] == pytest.approx(float(expected.detach()))
    assert result["gradient_norm"] == pytest.approx(float(expected_norm))
    assert result["history_examples"] == 16
    assert result["event_examples"] == 8
    assert result["writer_examples"] == 24
    for actual, expected_param in zip(writer.parameters(), oracle.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected_param.grad, rtol=1e-5, atol=1e-7)


def test_reset_training_matches_independent_four_pass_oracle():
    from tinymem.research.independent_fact_recurrent_training import train_recurrent_batch

    batch = _batch()
    writer = new_update_writer(8, seed=19)
    oracle = deepcopy(writer)
    expected = _reset_loss(oracle, batch)
    expected.backward()
    expected_norm = torch.nn.utils.clip_grad_norm_(tuple(oracle.parameters()), 1.0)
    result = train_recurrent_batch(writer, batch, torch.optim.SGD(writer.parameters(), lr=0.0), arm="reset")
    assert result["state_mse"] == pytest.approx(float(expected.detach()))
    assert result["gradient_norm"] == pytest.approx(float(expected_norm))
    for actual, expected_param in zip(writer.parameters(), oracle.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected_param.grad, rtol=1e-5, atol=1e-7)


def test_recurrent_arm_keeps_cross_step_gradient_and_differs_from_reset():
    from tinymem.research.independent_fact_recurrent_training import train_recurrent_batch

    batch = _batch()
    recurrent = new_update_writer(8, seed=29)
    reset = deepcopy(recurrent)
    detached = deepcopy(recurrent)
    recurrent_loss = _recurrent_loss(recurrent, batch)
    recurrent_loss.backward()
    detached_loss = _recurrent_loss(detached, batch, detach_between=True)
    detached_loss.backward()
    assert not torch.allclose(
        recurrent.input_projection.weight.grad,
        detached.input_projection.weight.grad,
    )
    recurrent_result = train_recurrent_batch(
        recurrent, batch, torch.optim.SGD(recurrent.parameters(), lr=0.0), arm="recurrent",
    )
    reset_result = train_recurrent_batch(
        reset, batch, torch.optim.SGD(reset.parameters(), lr=0.0), arm="reset",
    )
    assert recurrent_result["state_mse"] != pytest.approx(reset_result["state_mse"])


def test_training_rejects_bad_arm_features_stream_and_optimizer_and_preserves_batch():
    from tinymem.research.independent_fact_recurrent_training import (
        pack_recurrent_batch, train_recurrent_batch,
    )

    histories, events = _features()
    streams = [_stream_dict(0)]
    batch = pack_recurrent_batch(streams, histories, events)
    snapshot = tuple(value.clone() for value in batch.__dict__.values() if isinstance(value, torch.Tensor))
    writer = new_update_writer(8, seed=31)
    with pytest.raises(ValueError, match="arm"):
        train_recurrent_batch(writer, batch, torch.optim.SGD(writer.parameters(), lr=0.01), arm="bad")
    with pytest.raises(ValueError, match="optimizer"):
        train_recurrent_batch(writer, batch, torch.optim.SGD(tuple(writer.parameters())[:-1], lr=0.01), arm="reset")
    train_recurrent_batch(writer, batch, torch.optim.SGD(writer.parameters(), lr=0.01), arm="reset")
    for actual, expected in zip(
        (value for value in batch.__dict__.values() if isinstance(value, torch.Tensor)),
        snapshot,
        strict=True,
    ):
        assert torch.equal(actual, expected)

    bad_histories = dict(histories)
    bad_histories.pop(4)
    with pytest.raises(ValueError, match="history"):
        pack_recurrent_batch(streams, bad_histories, events)
    bad_events = dict(events)
    bad_events[(0, 0)] = torch.full_like(bad_events[(0, 0)], float("nan"))
    with pytest.raises(ValueError, match="event"):
        pack_recurrent_batch(streams, histories, bad_events)
    bad_stream = _stream_dict(0)
    bad_stream["events"][0]["after_code"] = 7
    with pytest.raises(ValueError, match="stream"):
        pack_recurrent_batch([bad_stream], histories, events)


def test_reset_update_is_finite_and_reports_clip_status():
    from tinymem.research.independent_fact_recurrent_training import train_recurrent_batch

    batch = _batch()
    writer = new_update_writer(8, seed=37)
    result = train_recurrent_batch(writer, batch, torch.optim.AdamW(writer.parameters(), lr=0.001), arm="reset")
    assert result["state_mse"] >= 0
    assert result["gradient_norm"] >= 0
    assert isinstance(result["gradient_clipped"], bool)
    assert all(torch.isfinite(parameter).all() for parameter in writer.parameters())
