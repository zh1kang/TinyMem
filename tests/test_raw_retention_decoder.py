import torch

from tinymem.model.config import ModelConfig
from tinymem.model.raw_retention_decoder import RawRetentionDecoder
from tinymem.model.transformer import DecoderOnlyTransformer


def test_raw_reader_matches_explicit_prefix_without_padding_gap_or_cache():
    reader = DecoderOnlyTransformer(ModelConfig(vocab_size=260, d_model=8, n_layers=1, n_heads=2, d_ff=16, max_local_tokens=32, dropout=0.0))
    model = RawRetentionDecoder(reader, capacity=4, query_length=8)
    initial = model.empty(2)
    state = model.write(torch.tensor([[1, 2, 3], [4, 259, 259]]), torch.tensor([[True, True, True], [True, False, False]]), initial)
    query = torch.tensor([[10, 11], [12, 13]])
    actual = model(query, state)
    expected_first = reader(torch.tensor([[1, 2, 3, 10, 11]]))[:, -2:]
    expected_second = reader(torch.tensor([[4, 12, 13]]))[:, -2:]
    torch.testing.assert_close(actual, torch.cat((expected_first, expected_second)))
    assert state.nbytes == 2 * (4 + 8)
    assert initial.lengths.tolist() == [0, 0]
    assert set(vars(state)) == {"token_ids", "lengths"}
    assert not any("cache" in name for name in vars(model))
    actual.square().mean().backward()
    assert reader.token_embedding.weight.grad is not None


def test_expired_prefix_cannot_change_raw_reader_after_same_suffix():
    reader = DecoderOnlyTransformer(ModelConfig(vocab_size=260, d_model=8, n_layers=1, n_heads=2, d_ff=16, max_local_tokens=32, dropout=0.0))
    model = RawRetentionDecoder(reader, capacity=3, query_length=8)
    valid = torch.ones(1, 5, dtype=torch.bool)
    a = model.write(torch.tensor([[1, 2, 7, 8, 9]]), valid, model.empty(1))
    b = model.write(torch.tensor([[5, 6, 7, 8, 9]]), valid, model.empty(1))
    query = torch.tensor([[10, 11]])
    torch.testing.assert_close(model(query, a), model(query, b))
