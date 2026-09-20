from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tinymem.memory.delta_slots import DeltaSlotWriter
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.distilled_fact_training import (
    DistilledExample,
    _direct_bits,
    rollout_loss,
    train_batch,
)


def _example(writer: DeltaSlotWriter, writes: int = 2) -> DistilledExample:
    features = tuple(torch.randn(2, writer.reader_width) for _ in range(writes))
    targets = tuple(torch.zeros(1, 2, writer.memory_width) for _ in range(writes))
    truth = tuple((None, None, None, None) for _ in range(writes))
    return DistilledExample("train/example", "train", features, targets, truth)


def test_train_batch_supervises_every_write_and_all_coordinates() -> None:
    torch.manual_seed(4)
    writer = DeltaSlotWriter(4, 32, key_width=8)
    example = _example(writer, writes=3)
    optimizer = torch.optim.SGD(writer.parameters(), lr=0.0)

    metric = train_batch(writer, (example,), optimizer)

    assert metric["write_calls"] == 3
    assert metric["histories"] == 1
    assert metric["state_loss"] > 0
    assert metric["off_fact_loss"] > 0
    assert all(parameter.grad is not None for parameter in writer.parameters())


def test_train_batch_accepts_fixed_beta_without_unused_trainable_parameters() -> None:
    torch.manual_seed(5)
    writer = DeltaSlotWriter(4, 32, key_width=8, fixed_beta=0.75)
    example = _example(writer, writes=2)
    optimizer = torch.optim.SGD(writer.parameters(), lr=0.0)

    metric = train_batch(writer, (example,), optimizer)

    assert metric["write_calls"] == 2
    assert metric["persistent_bytes"] == 258
    assert all(parameter.grad is not None for parameter in writer.parameters())


def test_rollout_loss_matches_independent_full_coordinate_sum() -> None:
    torch.manual_seed(8)
    writer = DeltaSlotWriter(4, 32, key_width=8)
    example = _example(writer, writes=2)
    with torch.no_grad():
        state = writer.empty(1)
        expected = torch.zeros((), dtype=torch.float32)
        expected_off_fact = torch.zeros((), dtype=torch.float32)
        for feature in example.features:
            state = writer(state, feature.unsqueeze(0), torch.ones((1, feature.shape[0]), dtype=torch.bool))
            flat = state.values.reshape(-1)
            expected = expected + flat.square().sum()
            expected_off_fact = expected_off_fact + flat.square()[[i for i in range(64) if i not in (0, 8, 16, 24)]].sum()
    loss, off_fact_loss = rollout_loss(writer, (example,))
    assert torch.allclose(loss, expected / 2)
    assert torch.allclose(off_fact_loss, expected_off_fact / 2)


def test_rollout_loss_keeps_the_previous_state_attached() -> None:
    torch.manual_seed(21)
    writer_attached = DeltaSlotWriter(4, 32, key_width=8)
    writer_detached = DeltaSlotWriter(4, 32, key_width=8)
    writer_detached.load_state_dict(writer_attached.state_dict())
    example = _example(writer_attached, writes=2)
    attached, _ = rollout_loss(writer_attached, (example,))
    attached.backward()
    attached_gradients = torch.cat([
        parameter.grad.reshape(-1) for parameter in writer_attached.parameters() if parameter.grad is not None
    ])

    state = writer_detached.empty(1)
    detached_loss = torch.zeros(())
    for feature, target in zip(example.features, example.targets):
        state = writer_detached(state, feature.unsqueeze(0), torch.ones((1, feature.shape[0]), dtype=torch.bool))
        detached_loss = detached_loss + (state.values - target).square().sum()
        state = LatentSlotState(state.values.detach(), state.valid.detach())
    detached_loss = detached_loss / 2
    torch.testing.assert_close(attached, detached_loss)
    detached_loss.backward()
    detached_gradients = torch.cat([
        parameter.grad.reshape(-1) for parameter in writer_detached.parameters() if parameter.grad is not None
    ])

    assert not torch.allclose(attached_gradients, detached_gradients)


def test_loss_weights_histories_equally_and_includes_initial_off_fact_targets() -> None:
    torch.manual_seed(12)
    writer = DeltaSlotWriter(4, 32, key_width=8)
    short, long = _example(writer, writes=1), _example(writer, writes=3)
    target = short.targets[0].clone()
    target[0, 1, 31] = 2.0
    short = replace(short, targets=(target,))
    expected = []
    with torch.no_grad():
        for example in (short, long):
            state = writer.empty(1)
            errors = []
            for feature, goal in zip(example.features, example.targets, strict=True):
                state = writer(state, feature.unsqueeze(0), torch.ones((1, feature.shape[0]), dtype=torch.bool))
                errors.append(sum((float(value) - float(wanted)) ** 2
                                  for value, wanted in zip(state.values.flatten(), goal.flatten(), strict=True)))
            expected.append(errors)
    loss, _ = rollout_loss(writer, (short, long))
    mean_histories = (sum(expected[0]) + sum(expected[1]) / 3) / 2
    assert float(loss.detach()) == pytest.approx(mean_histories, rel=1e-6)
    assert float(loss.detach()) != pytest.approx(sum(map(sum, expected)) / 4, rel=1e-6)


@pytest.mark.parametrize("split", ("validation", "test"))
def test_train_batch_rejects_non_training_examples(split) -> None:
    writer = DeltaSlotWriter(4, 32, key_width=8)
    example = DistilledExample(
        f"{split}/example", split, (torch.zeros(1, 4),),
        (torch.zeros(1, 2, 32),), ((None, None, None, None),),
    )
    optimizer = torch.optim.SGD(writer.parameters(), lr=0.0)

    with pytest.raises(ValueError, match="training"):
        train_batch(writer, (example,), optimizer)


def test_exact_zero_has_no_direct_bit() -> None:
    assert _direct_bits(torch.zeros(1, 2, 32)) == [None, None, None, None]
