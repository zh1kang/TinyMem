from pathlib import Path

import pytest
import torch

from tinymem.model.config import ExperimentConfig
from tinymem.training.checkpointing import load_checkpoint, save_checkpoint
from tinymem.utils.seed import seed_everything


def test_checkpoint_round_trip_reproduces_logits(tmp_path: Path) -> None:
    seed_everything(41)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 3),
    )
    inputs = torch.randn(2, 4)
    expected_logits = model(inputs).detach().clone()
    path = tmp_path / "nested" / "model.pt"

    save_checkpoint(
        path,
        model=model,
        step=12,
        config=ExperimentConfig(seed=41),
        extra={"validation_loss": 0.25},
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    metadata = load_checkpoint(path, model=model)
    restored_logits = model(inputs).detach()

    assert torch.equal(restored_logits, expected_logits)
    assert metadata["step"] == 12
    assert metadata["config"] == ExperimentConfig(seed=41).to_dict()
    assert metadata["extra"] == {"validation_loss": 0.25}


def test_checkpoint_round_trip_restores_optimizer_state(tmp_path: Path) -> None:
    seed_everything(43)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    path = tmp_path / "model.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, step=1)

    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.5)
    metadata = load_checkpoint(
        path,
        model=restored_model,
        optimizer=restored_optimizer,
    )

    assert metadata["step"] == 1
    assert restored_optimizer.param_groups[0]["lr"] == 0.01
    assert len(restored_optimizer.state) == len(optimizer.state)


@pytest.mark.parametrize("invalid_step", [True, 1.5, "1"])
def test_save_checkpoint_rejects_noninteger_step(
    tmp_path: Path,
    invalid_step: object,
) -> None:
    with pytest.raises(TypeError, match="step must be an integer"):
        save_checkpoint(
            tmp_path / "model.pt",
            model=torch.nn.Linear(1, 1),
            step=invalid_step,
        )


def test_save_checkpoint_rejects_negative_step(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="step must be nonnegative"):
        save_checkpoint(
            tmp_path / "model.pt",
            model=torch.nn.Linear(1, 1),
            step=-1,
        )


def test_load_checkpoint_requires_optimizer_state(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model=model)

    with pytest.raises(ValueError, match="does not contain optimizer state"):
        load_checkpoint(
            path,
            model=model,
            optimizer=torch.optim.AdamW(model.parameters()),
        )


def test_save_checkpoint_removes_temporary_file_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "model.pt"

    def fail_save(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated save failure")

    monkeypatch.setattr(torch, "save", fail_save)

    with pytest.raises(RuntimeError, match="simulated save failure"):
        save_checkpoint(destination, model=torch.nn.Linear(1, 1))

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []
