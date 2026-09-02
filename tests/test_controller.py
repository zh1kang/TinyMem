import pytest
import torch

from tinymem.memory.controller import (
    KEEP_ACTION,
    WRITE_ACTION,
    AdaptiveWriteController,
)


def make_inputs() -> tuple[torch.Tensor, ...]:
    return (
        torch.randn(3, 4),
        torch.randn(3, 4),
        torch.rand(3, 1),
        torch.tensor([True, True, False]),
    )


def test_controller_returns_hard_binary_actions_and_probabilities() -> None:
    torch.manual_seed(7)
    controller = AdaptiveWriteController(4, hidden_width=6)

    output = controller(*make_inputs())

    assert output.logits.shape == (3, 2)
    assert output.probabilities.shape == (3, 2)
    assert output.assignments.shape == (3, 2)
    assert output.actions.shape == (3,)
    assert output.write_strength.shape == (3, 1)
    assert torch.all((output.assignments == 0) | (output.assignments == 1))
    assert torch.equal(output.assignments.sum(dim=-1), torch.ones(3))
    assert torch.equal(output.write_strength[:, 0], output.assignments[:, 1])
    assert output.actions[2] == KEEP_ACTION
    assert output.write_strength[2] == 0


def test_controller_straight_through_gradient_reaches_features() -> None:
    torch.manual_seed(11)
    controller = AdaptiveWriteController(4)
    segment = torch.randn(2, 4, requires_grad=True)
    memory = torch.randn(2, 4, requires_grad=True)
    surprise = torch.rand(2, 1, requires_grad=True)
    valid = torch.ones(2, dtype=torch.bool)

    output = controller(segment, memory, surprise, valid)
    output.write_strength.sum().backward()

    for tensor in (segment, memory, surprise):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
    assert controller.input_projection.weight.grad is not None
    assert controller.action_projection.weight.grad is not None


def test_controller_evaluation_is_deterministic() -> None:
    controller = AdaptiveWriteController(4)
    controller.eval()
    inputs = make_inputs()

    torch.manual_seed(1)
    first = controller(*inputs)
    torch.manual_seed(99)
    second = controller(*inputs)

    assert torch.equal(first.actions, second.actions)
    assert torch.equal(first.assignments, second.assignments)
    torch.testing.assert_close(first.probabilities, second.probabilities)


def test_controller_initially_prefers_write_for_valid_rows() -> None:
    controller = AdaptiveWriteController(4)
    controller.eval()

    output = controller(*make_inputs())

    assert torch.equal(
        output.actions,
        torch.tensor([WRITE_ACTION, WRITE_ACTION, KEEP_ACTION]),
    )


def test_controller_temperature_is_checkpointed() -> None:
    controller = AdaptiveWriteController(4)
    controller.set_temperature(0.25)
    restored = AdaptiveWriteController(4)

    restored.load_state_dict(controller.state_dict())

    assert restored.temperature == pytest.approx(0.25)


@pytest.mark.parametrize("value", [True, 0, -1, 2.5, "4"])
def test_controller_rejects_invalid_model_width(value: object) -> None:
    expected = (
        ValueError
        if isinstance(value, int) and not isinstance(value, bool)
        else TypeError
    )

    with pytest.raises(expected):
        AdaptiveWriteController(value)


@pytest.mark.parametrize("temperature", [True, 0.0, -1.0, "1", float("inf")])
def test_controller_rejects_invalid_temperature(temperature: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        AdaptiveWriteController(4, temperature=temperature)


@pytest.mark.parametrize(
    ("index", "replacement", "error"),
    [
        (0, [[1.0] * 4], TypeError),
        (0, torch.ones(3, 5), ValueError),
        (1, torch.ones(3, 5), ValueError),
        (2, torch.ones(3), ValueError),
        (3, torch.ones(3), TypeError),
    ],
)
def test_controller_rejects_invalid_inputs(
    index: int,
    replacement: object,
    error: type[Exception],
) -> None:
    controller = AdaptiveWriteController(4)
    inputs = list(make_inputs())
    inputs[index] = replacement

    with pytest.raises(error):
        controller(*inputs)
