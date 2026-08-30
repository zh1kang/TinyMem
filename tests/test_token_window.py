import pytest
import torch

from tinymem.memory.token_window import LocalTokenWindow, RawTokenBatch


def test_token_window_initializes_without_expiration() -> None:
    window = LocalTokenWindow(max_length=4)

    expired = window.append(
        torch.tensor([[10, 11], [20, 21]]),
        position_offset=3,
    )

    assert expired.positions.numel() == 0
    assert expired.token_ids.shape == (2, 0)
    assert torch.equal(window.state.positions, torch.tensor([3, 4]))
    assert torch.equal(
        window.state.token_ids,
        torch.tensor([[10, 11], [20, 21]]),
    )


def test_token_window_appends_without_exceeding_capacity() -> None:
    window = LocalTokenWindow(max_length=4)
    window.append(torch.tensor([[10, 11]]), position_offset=0)

    expired = window.append(torch.tensor([[12, 13]]), position_offset=2)

    assert expired.positions.numel() == 0
    assert torch.equal(window.state.positions, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(window.state.token_ids, torch.tensor([[10, 11, 12, 13]]))


def test_token_window_returns_oldest_tokens_when_window_slides() -> None:
    window = LocalTokenWindow(max_length=4)
    window.append(torch.tensor([[10, 11, 12, 13]]), position_offset=0)

    expired = window.append(torch.tensor([[14, 15]]), position_offset=4)

    assert torch.equal(expired.positions, torch.tensor([0, 1]))
    assert torch.equal(expired.token_ids, torch.tensor([[10, 11]]))
    assert torch.equal(window.state.positions, torch.tensor([2, 3, 4, 5]))
    assert torch.equal(window.state.token_ids, torch.tensor([[12, 13, 14, 15]]))


def test_token_window_bounds_an_oversized_first_append() -> None:
    window = LocalTokenWindow(max_length=3)

    expired = window.append(
        torch.tensor([[10, 11, 12, 13, 14]]),
        position_offset=5,
    )

    assert torch.equal(expired.positions, torch.tensor([5, 6]))
    assert torch.equal(expired.token_ids, torch.tensor([[10, 11]]))
    assert torch.equal(window.state.positions, torch.tensor([7, 8, 9]))
    assert torch.equal(window.state.token_ids, torch.tensor([[12, 13, 14]]))


def test_token_window_preserves_batch_alignment() -> None:
    window = LocalTokenWindow(max_length=2)
    window.append(
        torch.tensor([[1, 2], [11, 12]]),
        position_offset=0,
    )

    expired = window.append(
        torch.tensor([[3], [13]]),
        position_offset=2,
    )

    assert torch.equal(expired.token_ids, torch.tensor([[1], [11]]))
    assert torch.equal(window.state.token_ids, torch.tensor([[2, 3], [12, 13]]))


def test_token_window_owns_input_and_returned_tensors() -> None:
    window = LocalTokenWindow(max_length=2)
    input_ids = torch.tensor([[1, 2]])
    window.append(input_ids, position_offset=0)
    input_ids.fill_(9)
    expired = window.append(torch.tensor([[3]]), position_offset=2)
    expired.token_ids.fill_(8)

    assert torch.equal(window.state.token_ids, torch.tensor([[2, 3]]))


def test_token_window_state_is_a_defensive_copy() -> None:
    window = LocalTokenWindow(max_length=2)
    window.append(torch.tensor([[1, 2]]), position_offset=0)

    copied_state = window.state
    copied_state.positions.fill_(9)
    copied_state.token_ids.fill_(9)

    assert torch.equal(window.state.positions, torch.tensor([0, 1]))
    assert torch.equal(window.state.token_ids, torch.tensor([[1, 2]]))


def test_token_window_requires_contiguous_positions() -> None:
    window = LocalTokenWindow(max_length=2)
    window.append(torch.tensor([[1]]), position_offset=3)

    with pytest.raises(ValueError, match="position_offset must be 4"):
        window.append(torch.tensor([[2]]), position_offset=5)


def test_token_window_reset_allows_a_new_stream() -> None:
    window = LocalTokenWindow(max_length=2)
    window.append(torch.tensor([[1]]), position_offset=3)

    window.reset()
    expired = window.append(torch.tensor([[7]]), position_offset=0)

    assert expired.positions.numel() == 0
    assert torch.equal(window.state.positions, torch.tensor([0]))


@pytest.mark.parametrize("max_length", [True, 0, -1, 2.0])
def test_token_window_rejects_invalid_max_length(max_length: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        LocalTokenWindow(max_length=max_length)


@pytest.mark.parametrize(
    ("positions", "token_ids", "error"),
    [
        (torch.tensor([[0]]), torch.ones(1, 1, dtype=torch.long), ValueError),
        (torch.tensor([0.0]), torch.ones(1, 1, dtype=torch.long), TypeError),
        (torch.tensor([0]), torch.ones(1, 1), TypeError),
        (torch.tensor([0, 1]), torch.ones(1, 1, dtype=torch.long), ValueError),
        (torch.tensor([1, 1]), torch.ones(1, 2, dtype=torch.long), ValueError),
        (torch.tensor([-1]), torch.ones(1, 1, dtype=torch.long), ValueError),
    ],
)
def test_raw_token_batch_rejects_invalid_tensors(
    positions: torch.Tensor,
    token_ids: torch.Tensor,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        RawTokenBatch(positions=positions, token_ids=token_ids)
