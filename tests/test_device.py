import pytest
import torch

from tinymem.utils.device import select_device


def set_availability(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cuda: bool,
    mps: bool,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


def test_select_device_rejects_nonstring_preference() -> None:
    with pytest.raises(TypeError, match="preference must be a string"):
        select_device(None)


def test_select_device_rejects_unknown_preference() -> None:
    with pytest.raises(ValueError, match="unsupported device 'gpu'"):
        select_device("gpu")


def test_select_device_always_accepts_explicit_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_availability(monkeypatch, cuda=True, mps=True)

    assert select_device("cpu") == torch.device("cpu")


def test_select_device_accepts_available_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_availability(monkeypatch, cuda=True, mps=False)

    assert select_device("cuda") == torch.device("cuda")


def test_select_device_rejects_unavailable_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_availability(monkeypatch, cuda=False, mps=True)

    with pytest.raises(RuntimeError, match="CUDA was requested but is not available"):
        select_device("cuda")


def test_select_device_accepts_available_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_availability(monkeypatch, cuda=False, mps=True)

    assert select_device("mps") == torch.device("mps")


def test_select_device_rejects_unavailable_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_availability(monkeypatch, cuda=True, mps=False)

    with pytest.raises(RuntimeError, match="MPS was requested but is not available"):
        select_device("mps")


def test_select_device_auto_prefers_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_availability(monkeypatch, cuda=True, mps=True)

    assert select_device() == torch.device("cuda")


def test_select_device_auto_uses_mps_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_availability(monkeypatch, cuda=False, mps=True)

    assert select_device() == torch.device("mps")


def test_select_device_auto_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_availability(monkeypatch, cuda=False, mps=False)

    assert select_device() == torch.device("cpu")
