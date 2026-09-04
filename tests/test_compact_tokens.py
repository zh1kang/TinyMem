import pytest
import torch

from tinymem.memory.compact_tokens import CompactTokenRetention, token_storage_dtype
from tinymem.memory.storage import tensor_storage_bytes


def test_storage_counts_shared_allocations_and_large_backing_views():
    whole = torch.zeros(100, dtype=torch.float32)
    view = whole[:2]
    assert tensor_storage_bytes([view]) == 400
    assert tensor_storage_bytes([whole, view, view]) == 400
    assert tensor_storage_bytes([view.clone()]) == 8


@pytest.mark.parametrize("vocab,dtype", [(256, torch.uint8), (260, torch.int16), (32768, torch.int16), (32769, torch.int32)])
def test_raw_ids_use_smallest_supported_storage(vocab, dtype):
    assert token_storage_dtype(vocab) == dtype


def test_raw_retention_is_bounded_out_of_place_and_has_no_embeddings():
    policy = CompactTokenRetention(3, 260)
    state = policy.empty(2, device="cpu")
    expected_bytes = 2 * (3 * 2 + 8)
    first = policy.append(state, torch.tensor([[1, 2], [3, 4]]), torch.tensor([[True, True], [True, False]]))
    final = policy.append(first, torch.tensor([[5, 6, 7], [8, 9, 10]]), torch.ones(2, 3, dtype=torch.bool))
    ids, valid = policy.materialize(final, pad_id=259)
    assert ids.tolist() == [[5, 6, 7], [8, 9, 10]]
    assert valid.all()
    assert first.token_ids.tolist() == [[1, 2, 0], [3, 0, 0]]
    assert state.lengths.tolist() == [0, 0]
    assert set(vars(final)) == {"token_ids", "lengths"}
    assert state.nbytes == first.nbytes == final.nbytes == expected_bytes


def test_materialized_padding_and_read_copy_do_not_change_state():
    policy = CompactTokenRetention(4, 260)
    state = policy.append(policy.empty(1, device="cpu"), torch.tensor([[42, 9]]), torch.tensor([[True, False]]))
    ids, valid = policy.materialize(state, pad_id=259)
    assert ids.tolist() == [[42, 259, 259, 259]]
    assert valid.tolist() == [[True, False, False, False]]
    ids.zero_()
    assert int(state.token_ids[0, 0]) == 42


def test_invalid_ids_do_not_silently_overflow_compact_storage():
    policy = CompactTokenRetention(2, 260)
    with pytest.raises(ValueError, match="vocabulary"):
        policy.append(policy.empty(1, device="cpu"), torch.tensor([[32768]]), torch.ones(1, 1, dtype=torch.bool))


def test_foreign_state_capacity_and_vocabulary_fail_at_both_boundaries():
    owner = CompactTokenRetention(4, 260)
    state = owner.append(owner.empty(1, device="cpu"), torch.tensor([[259]]), torch.ones(1, 1, dtype=torch.bool))
    for other in (CompactTokenRetention(1, 260), CompactTokenRetention(4, 257)):
        with pytest.raises(ValueError):
            other.materialize(state, pad_id=0)
        with pytest.raises(ValueError):
            other.append(state, torch.tensor([[1]]), torch.ones(1, 1, dtype=torch.bool))
