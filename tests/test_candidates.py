import pytest
import torch
from torch import nn

from tinymem.memory.attention_tracker import AttentionScoreState
from tinymem.memory.candidates import build_scored_token_candidates
from tinymem.memory.token_window import RawTokenBatch


def make_aligned_inputs() -> tuple[RawTokenBatch, AttentionScoreState]:
    positions = torch.tensor([4, 5], dtype=torch.int64)
    tokens = RawTokenBatch(
        positions=positions,
        token_ids=torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
    )
    scores = AttentionScoreState(
        positions=positions.clone(),
        scores=torch.tensor([[0.2, 0.8], [0.7, 0.3]], dtype=torch.float32),
    )
    return tokens, scores


def make_embedding() -> nn.Embedding:
    embedding = nn.Embedding(num_embeddings=8, embedding_dim=3)
    with torch.no_grad():
        embedding.weight.copy_(
            torch.arange(24, dtype=torch.float32).reshape(8, 3)
        )
    return embedding


def test_candidates_align_values_provenance_and_scores() -> None:
    tokens, scores = make_aligned_inputs()
    embedding = make_embedding()

    candidates = build_scored_token_candidates(tokens, scores, embedding)

    assert candidates.values.shape == (2, 2, 3)
    torch.testing.assert_close(candidates.values, embedding(tokens.token_ids))
    assert torch.equal(candidates.valid, torch.ones(2, 2, dtype=torch.bool))
    assert torch.equal(candidates.positions, torch.tensor([[4, 5], [4, 5]]))
    assert torch.equal(candidates.token_ids, tokens.token_ids)
    assert torch.equal(candidates.scores, scores.scores)


def test_candidates_support_an_empty_expiration_batch() -> None:
    tokens = RawTokenBatch.empty(
        batch_size=2,
        device="cpu",
        dtype=torch.int64,
    )
    scores = AttentionScoreState.empty(batch_size=2, device="cpu")

    candidates = build_scored_token_candidates(tokens, scores, make_embedding())

    assert candidates.values.shape == (2, 0, 3)
    assert candidates.valid.shape == (2, 0)
    assert candidates.positions.shape == (2, 0)
    assert candidates.token_ids is not None
    assert candidates.token_ids.shape == (2, 0)
    assert candidates.scores is not None
    assert candidates.scores.shape == (2, 0)


def test_candidates_own_detached_tensor_state() -> None:
    tokens, scores = make_aligned_inputs()
    embedding = make_embedding()

    candidates = build_scored_token_candidates(tokens, scores, embedding)
    tokens.token_ids.fill_(0)
    scores.scores.fill_(9.0)
    with torch.no_grad():
        embedding.weight.zero_()

    assert not candidates.values.requires_grad
    assert torch.equal(candidates.token_ids, torch.tensor([[1, 2], [3, 4]]))
    assert torch.equal(
        candidates.scores,
        torch.tensor([[0.2, 0.8], [0.7, 0.3]]),
    )
    assert candidates.values.count_nonzero() > 0


@pytest.mark.parametrize(
    ("tokens", "scores", "message"),
    [
        (
            RawTokenBatch(
                positions=torch.tensor([4, 5]),
                token_ids=torch.tensor([[1, 2]]),
            ),
            AttentionScoreState(
                positions=torch.tensor([4, 6]),
                scores=torch.tensor([[0.2, 0.8]]),
            ),
            "identical positions",
        ),
        (
            RawTokenBatch(
                positions=torch.tensor([4, 5]),
                token_ids=torch.tensor([[1, 2], [3, 4]]),
            ),
            AttentionScoreState(
                positions=torch.tensor([4, 5]),
                scores=torch.tensor([[0.2, 0.8]]),
            ),
            "same batch size",
        ),
    ],
)
def test_candidates_reject_misaligned_inputs(
    tokens: RawTokenBatch,
    scores: AttentionScoreState,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_scored_token_candidates(tokens, scores, make_embedding())


def test_candidates_reject_token_ids_outside_the_embedding_vocabulary() -> None:
    tokens = RawTokenBatch(
        positions=torch.tensor([4]),
        token_ids=torch.tensor([[8]]),
    )
    scores = AttentionScoreState(
        positions=torch.tensor([4]),
        scores=torch.tensor([[0.5]]),
    )

    with pytest.raises(ValueError, match=r"\[0, 8\)"):
        build_scored_token_candidates(tokens, scores, make_embedding())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tokens", object(), "RawTokenBatch"),
        ("scores", object(), "AttentionScoreState"),
        ("embedding", object(), "nn.Embedding"),
    ],
)
def test_candidates_reject_invalid_object_types(
    field: str,
    value: object,
    message: str,
) -> None:
    tokens, scores = make_aligned_inputs()
    arguments = {
        "expired_tokens": tokens,
        "expired_scores": scores,
        "token_embedding": make_embedding(),
    }
    argument_names = {
        "tokens": "expired_tokens",
        "scores": "expired_scores",
        "embedding": "token_embedding",
    }
    arguments[argument_names[field]] = value

    with pytest.raises(TypeError, match=message):
        build_scored_token_candidates(**arguments)
