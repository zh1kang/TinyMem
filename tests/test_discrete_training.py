import math

import pytest
import torch

from tinymem.training.discrete import (
    GumbelTemperatureSchedule,
    codebook_usage_loss,
)


def test_gumbel_temperature_schedule_anneals_and_clamps() -> None:
    schedule = GumbelTemperatureSchedule(
        start=2.0,
        end=0.5,
        anneal_steps=100,
    )

    assert schedule.value(0) == 2.0
    assert schedule.value(50) == 1.25
    assert schedule.value(100) == 0.5
    assert schedule.value(200) == 0.5


def test_usage_loss_rewards_balanced_aggregate_assignments() -> None:
    balanced = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    collapsed = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    valid = torch.ones(1, 2, dtype=torch.bool)

    balanced_loss = codebook_usage_loss(balanced, valid)
    collapsed_loss = codebook_usage_loss(collapsed, valid)

    assert balanced_loss == pytest.approx(-math.log(2))
    assert collapsed_loss == pytest.approx(0.0)
    assert balanced_loss < collapsed_loss


def test_usage_loss_ignores_invalid_assignments_and_gradients() -> None:
    assignments = torch.tensor(
        [[[0.75, 0.25], [0.1, 0.9]]],
        requires_grad=True,
    )
    valid = torch.tensor([[True, False]])

    loss = codebook_usage_loss(assignments, valid)
    loss.backward()

    assert assignments.grad is not None
    assert torch.count_nonzero(assignments.grad[0, 0]) > 0
    assert torch.equal(assignments.grad[0, 1], torch.zeros(2))


@pytest.mark.parametrize(
    ("start", "end", "steps", "error"),
    [
        (True, 0.5, 10, TypeError),
        (1.0, "0.5", 10, TypeError),
        (0.0, 0.5, 10, ValueError),
        (1.0, float("inf"), 10, ValueError),
        (1.0, 0.5, True, TypeError),
        (1.0, 0.5, 0, ValueError),
    ],
)
def test_gumbel_temperature_schedule_rejects_invalid_values(
    start: object,
    end: object,
    steps: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        GumbelTemperatureSchedule(start=start, end=end, anneal_steps=steps)
