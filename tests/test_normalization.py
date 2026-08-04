import pytest
import torch

from tinymem.model.normalization import RMSNorm


def test_rmsnorm_preserves_shape_and_normalizes_each_token() -> None:
    layer = RMSNorm(d_model=4)
    inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, 2.0, 3.0]])

    outputs = layer(inputs)

    assert outputs.shape == inputs.shape
    assert torch.allclose(layer.weight, torch.ones(4))
    rms_squared = outputs.square().mean(dim=-1)
    assert torch.allclose(rms_squared, torch.ones(2), atol=1e-5)


def test_rmsnorm_supports_any_leading_shape() -> None:
    layer = RMSNorm(d_model=8)
    inputs = torch.randn(2, 3, 4, 8)

    outputs = layer(inputs)

    assert outputs.shape == inputs.shape
    assert torch.isfinite(outputs).all()


def test_rmsnorm_learns_featurewise_scale() -> None:
    layer = RMSNorm(d_model=2)
    layer.weight.data = torch.tensor([2.0, 3.0])
    outputs = layer(torch.ones(1, 2))

    assert torch.allclose(outputs, torch.tensor([[2.0, 3.0]]), atol=1e-6)


@pytest.mark.parametrize("d_model", [0, -1, True, 2.0])
def test_rmsnorm_rejects_invalid_d_model(d_model: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        RMSNorm(d_model=d_model)


@pytest.mark.parametrize("eps", [0.0, -1.0, True, "1e-6"])
def test_rmsnorm_rejects_invalid_eps(eps: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        RMSNorm(d_model=4, eps=eps)


def test_rmsnorm_rejects_wrong_final_dimension() -> None:
    layer = RMSNorm(d_model=4)

    with pytest.raises(ValueError, match="final dimension"):
        layer(torch.randn(2, 5))


def test_rmsnorm_produces_finite_gradients() -> None:
    layer = RMSNorm(d_model=4)
    inputs = torch.randn(2, 3, 4, requires_grad=True)

    layer(inputs).sum().backward()

    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
