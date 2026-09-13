from copy import deepcopy

import pytest
import torch

from tinymem.research.independent_fact_placement import placement_state
from tinymem.research.independent_fact_updates import build_update_cases, new_update_writer


def _features(width=16):
    histories = {
        code: torch.arange((code + 1) * width, dtype=torch.float32).reshape(-1, width)
        for code in range(16)
    }
    events = {
        (fact, bit): torch.arange((fact + bit + 2) * width, dtype=torch.float32).reshape(-1, width)
        for fact in range(4)
        for bit in range(2)
    }
    return histories, events


def _serial_loss(writer, batch):
    initial = writer(writer.empty(len(batch.histories)), batch.histories, batch.history_valid)
    updated = writer(initial, batch.events, batch.event_valid)
    initial_loss = torch.nn.functional.mse_loss(initial.values, batch.before_targets)
    updated_loss = torch.nn.functional.mse_loss(updated.values, batch.after_targets)
    return 0.5 * initial_loss + 0.5 * updated_loss, initial, updated


def test_pack_initial_update_batch_uses_only_selected_histories_and_masks_padding():
    from tinymem.research.independent_fact_initial_write import pack_initial_update_batch

    histories, events = _features()
    cases = [build_update_cases()[index] for index in (0, 1, 8)]
    batch = pack_initial_update_batch(cases, histories, events)
    assert batch.histories.shape == (3, 2, 16)
    assert batch.events.shape == (3, 3, 16)
    assert batch.history_valid.tolist() == [[True, False], [True, False], [True, True]]
    assert batch.event_valid.tolist() == [[True, True, False], [True, True, True], [True, True, False]]
    for tensor in (batch.histories, batch.events, batch.before_targets, batch.after_targets):
        assert tensor.device.type == "cpu"
        assert tensor.dtype == torch.float32
        assert not tensor.requires_grad
    for row, case in enumerate(cases):
        expected_before = placement_state(case.before_code, torch.device("cpu"), "separate_fact0")
        expected_after = placement_state(case.after_code, torch.device("cpu"), "separate_fact0")
        torch.testing.assert_close(batch.before_targets[row], expected_before.values[0])
        torch.testing.assert_close(batch.after_targets[row], expected_after.values[0])
        assert batch.before_targets[row].numel() * 4 + 2 == 66


def test_joint_loss_matches_independent_serial_unroll_gradients():
    from tinymem.research.independent_fact_initial_write import (
        pack_initial_update_batch, train_initial_update_batch,
    )

    histories, events = _features()
    batch = pack_initial_update_batch(build_update_cases()[:5], histories, events)
    writer = new_update_writer(16, seed=19)
    serial = deepcopy(writer)
    losses = []
    for row in range(len(batch.histories)):
        history = batch.histories[row:row+1, batch.history_valid[row]]
        event = batch.events[row:row+1, batch.event_valid[row]]
        first = serial(serial.empty(1), history, torch.ones(history.shape[:2], dtype=torch.bool))
        second = serial(first, event, torch.ones(event.shape[:2], dtype=torch.bool))
        losses.append(0.5 * (first.values - batch.before_targets[row:row+1]).square().mean()
                      + 0.5 * (second.values - batch.after_targets[row:row+1]).square().mean())
    expected_loss = torch.stack(losses).mean()
    expected_loss.backward()
    expected_norm = torch.nn.utils.clip_grad_norm_(tuple(serial.parameters()), 1.0)
    result = train_initial_update_batch(writer, batch, torch.optim.SGD(writer.parameters(), lr=0.0))
    assert result["state_mse"] == pytest.approx(float(expected_loss.detach()))
    assert result["initial_state_mse"] > 0
    assert result["updated_state_mse"] > 0
    assert result["gradient_norm"] == pytest.approx(float(expected_norm))
    for actual, expected in zip(writer.parameters(), serial.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-7)


def test_updated_loss_backpropagates_through_initial_write_call():
    from tinymem.research.independent_fact_initial_write import (
        pack_initial_update_batch, train_initial_update_batch,
    )

    histories, events = _features()
    batch = pack_initial_update_batch(build_update_cases()[:2], histories, events)
    writer = new_update_writer(16, seed=23)
    captured = []

    def capture_first(_module, _inputs, output):
        if not captured:
            output.values.retain_grad()
            captured.append(output.values)

    hook = writer.register_forward_hook(capture_first)
    try:
        train_initial_update_batch(writer, batch, torch.optim.SGD(writer.parameters(), lr=0.0))
    finally:
        hook.remove()
    initial = captured[0]
    initial_term_gradient = (initial.detach() - batch.before_targets) / initial.numel()
    assert initial.grad is not None
    assert (initial.grad - initial_term_gradient).abs().max() > 1e-5


def test_training_preserves_batch_and_rejects_bad_features_or_optimizer():
    from tinymem.research.independent_fact_initial_write import (
        pack_initial_update_batch, train_initial_update_batch,
    )

    histories, events = _features()
    cases = build_update_cases()[:2]
    batch = pack_initial_update_batch(cases, histories, events)
    snapshot = tuple(tensor.clone() for tensor in batch.__dict__.values())
    writer = new_update_writer(16, seed=7)
    with pytest.raises(ValueError, match="optimizer"):
        train_initial_update_batch(writer, batch, torch.optim.SGD(tuple(writer.parameters())[:-1], lr=0.01))
    train_initial_update_batch(writer, batch, torch.optim.SGD(writer.parameters(), lr=0.01))
    for actual, expected in zip(batch.__dict__.values(), snapshot, strict=True):
        assert torch.equal(actual, expected)

    bad_histories = dict(histories)
    bad_histories.pop(3)
    with pytest.raises(ValueError, match="history"):
        pack_initial_update_batch(cases, bad_histories, events)
    bad_histories = dict(histories)
    bad_histories[0] = bad_histories[0].requires_grad_()
    with pytest.raises(ValueError, match="history"):
        pack_initial_update_batch(cases, bad_histories, events)
    bad_events = dict(events)
    bad_events[(0, 0)] = torch.full_like(bad_events[(0, 0)], float("nan"))
    with pytest.raises(ValueError, match="features"):
        pack_initial_update_batch(cases, histories, bad_events)


def test_outputs_remain_valid_two_slot_66_byte_states_after_update():
    from tinymem.research.independent_fact_initial_write import (
        pack_initial_update_batch, train_initial_update_batch,
    )

    histories, events = _features()
    batch = pack_initial_update_batch(build_update_cases()[:2], histories, events)
    writer = new_update_writer(16, seed=31)
    train_initial_update_batch(writer, batch, torch.optim.AdamW(writer.parameters(), lr=0.001))
    initial = writer(writer.empty(2), batch.histories, batch.history_valid)
    updated = writer(initial, batch.events, batch.event_valid)
    assert initial.valid.all() and updated.valid.all()
    assert initial.nbytes // 2 == 66
    assert updated.nbytes // 2 == 66
    assert torch.isfinite(initial.values).all() and torch.isfinite(updated.values).all()
