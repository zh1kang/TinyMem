import pytest
import torch

from tinymem.memory.interfaces import MemoryPolicy
from tinymem.memory.state import MemoryState


class KeepStatePolicy(MemoryPolicy):
    def update(
        self,
        state: MemoryState,
        candidates: MemoryState,
        *,
        generator: torch.Generator | None = None,
    ) -> MemoryState:
        self._validate_update(state, candidates)
        return state


def test_memory_policy_is_abstract() -> None:
    with pytest.raises(TypeError):
        MemoryPolicy(capacity=2)


@pytest.mark.parametrize("capacity", [0, -1, True, 2.0])
def test_memory_policy_rejects_invalid_capacity(capacity: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        KeepStatePolicy(capacity=capacity)


def test_memory_policy_initializes_its_fixed_capacity() -> None:
    policy = KeepStatePolicy(capacity=3)

    state = policy.initialize(batch_size=2, model_width=4)

    assert state.values.shape == (2, 3, 4)
    assert state.capacity == policy.capacity
    assert not state.valid.any()


def test_memory_policy_read_returns_state_values_and_mask() -> None:
    policy = KeepStatePolicy(capacity=2)
    state = policy.initialize(batch_size=1, model_width=4)

    values, valid = policy.read(state)

    assert values is state.values
    assert valid is state.valid


def test_memory_policy_rejects_state_with_wrong_capacity() -> None:
    policy = KeepStatePolicy(capacity=2)
    state = MemoryState.empty(batch_size=1, capacity=3, model_width=4)

    with pytest.raises(ValueError, match="capacity"):
        policy.read(state)


@pytest.mark.parametrize(
    "candidates",
    [
        MemoryState.empty(batch_size=2, capacity=2, model_width=4),
        MemoryState.empty(batch_size=1, capacity=2, model_width=5),
        MemoryState.empty(
            batch_size=1,
            capacity=2,
            model_width=4,
            dtype=torch.float64,
        ),
    ],
)
def test_memory_policy_rejects_incompatible_candidates(
    candidates: MemoryState,
) -> None:
    policy = KeepStatePolicy(capacity=2)
    state = policy.initialize(batch_size=1, model_width=4)

    with pytest.raises(ValueError):
        policy.update(state, candidates)


def test_memory_policy_accepts_compatible_candidates() -> None:
    policy = KeepStatePolicy(capacity=2)
    state = policy.initialize(batch_size=1, model_width=4)
    candidates = MemoryState.empty(batch_size=1, capacity=3, model_width=4)

    result = policy.update(state, candidates)

    assert result is state
