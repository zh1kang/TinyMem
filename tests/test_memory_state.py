import pytest
import torch

from tinymem.memory.state import MemoryState


def test_empty_memory_state_has_fixed_shapes_and_sentinels() -> None:
    state = MemoryState.empty(batch_size=2, capacity=3, model_width=4)

    assert state.values.shape == (2, 3, 4)
    assert state.valid.shape == (2, 3)
    assert state.positions.shape == (2, 3)
    assert state.token_ids is not None
    assert state.token_ids.shape == (2, 3)
    assert not state.valid.any()
    assert (state.positions == -1).all()
    assert (state.token_ids == -1).all()
    assert torch.equal(state.occupied_count, torch.zeros(2, dtype=torch.long))


def test_memory_state_shape_properties() -> None:
    state = MemoryState.empty(batch_size=2, capacity=3, model_width=4)

    assert state.batch_size == 2
    assert state.capacity == 3
    assert state.model_width == 4


def test_memory_state_reports_actual_tensor_bytes() -> None:
    state = MemoryState.empty(batch_size=2, capacity=3, model_width=4)
    tensors = [state.values, state.valid, state.positions, state.token_ids]
    expected = sum(
        tensor.numel() * tensor.element_size()
        for tensor in tensors
        if tensor is not None
    )

    assert state.nbytes == expected


def test_clear_resets_slots_in_place() -> None:
    state = MemoryState.empty(batch_size=1, capacity=2, model_width=3)
    values = state.values
    valid = state.valid
    state.values.fill_(2.0)
    state.valid.fill_(True)
    state.positions.copy_(torch.tensor([[4, 5]]))
    assert state.token_ids is not None
    state.token_ids.copy_(torch.tensor([[7, 8]]))

    state.clear()

    assert state.values is values
    assert state.valid is valid
    assert not state.values.any()
    assert not state.valid.any()
    assert (state.positions == -1).all()
    assert (state.token_ids == -1).all()


@pytest.mark.parametrize(
    "field, value",
    [
        ("batch_size", 0),
        ("capacity", -1),
        ("model_width", True),
        ("model_width", 4.0),
    ],
)
def test_empty_memory_state_rejects_invalid_dimensions(
    field: str,
    value: object,
) -> None:
    arguments: dict[str, object] = {
        "batch_size": 2,
        "capacity": 3,
        "model_width": 4,
    }
    arguments[field] = value

    with pytest.raises((TypeError, ValueError)):
        MemoryState.empty(**arguments)


@pytest.mark.parametrize(
    "field, tensor",
    [
        ("values", torch.zeros(2, 3, dtype=torch.float32)),
        ("values", torch.zeros(2, 3, 4, dtype=torch.long)),
        ("valid", torch.zeros(2, 3, dtype=torch.float32)),
        ("positions", torch.zeros(2, 3, dtype=torch.float32)),
        ("token_ids", torch.zeros(2, 3, dtype=torch.float32)),
    ],
)
def test_memory_state_rejects_invalid_tensors(
    field: str,
    tensor: torch.Tensor,
) -> None:
    arguments: dict[str, object] = {
        "values": torch.zeros(2, 3, 4),
        "valid": torch.zeros(2, 3, dtype=torch.bool),
        "positions": torch.zeros(2, 3, dtype=torch.long),
        "token_ids": torch.zeros(2, 3, dtype=torch.long),
    }
    arguments[field] = tensor

    with pytest.raises((TypeError, ValueError)):
        MemoryState(**arguments)
