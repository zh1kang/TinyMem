import random

import numpy as np
import pytest
import torch

from tinymem.utils.seed import seed_everything


@pytest.mark.parametrize("invalid_seed", [True, False, 1.0, "1", None])
def test_seed_everything_rejects_noninteger_seed(invalid_seed: object) -> None:
    with pytest.raises(TypeError):
        seed_everything(invalid_seed)


def test_seed_everything_rejects_negative_seed() -> None:
    with pytest.raises(ValueError):
        seed_everything(-1)


def test_seed_everything_reproduces_each_random_generator() -> None:
    seed_everything(17)
    first_python = random.random()
    first_numpy = np.random.random()
    first_torch = torch.rand(4)

    seed_everything(17)

    assert random.random() == first_python
    assert np.random.random() == first_numpy
    assert torch.equal(torch.rand(4), first_torch)


def test_seed_everything_reproduces_model_parameters() -> None:
    seed_everything(23)
    first = torch.nn.Linear(4, 3)

    seed_everything(23)
    second = torch.nn.Linear(4, 3)

    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters(), strict=True
    ):
        assert torch.equal(first_parameter, second_parameter)


def test_seed_everything_enables_deterministic_algorithms() -> None:
    seed_everything(29)

    assert torch.are_deterministic_algorithms_enabled()


def test_seed_everything_seeds_all_cuda_devices_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(torch, "manual_seed", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", calls.append)

    seed_everything(31)

    assert calls == [31]
