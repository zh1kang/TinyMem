from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from tinymem.research.native_memory_oracle import FixedProjectionMemoryOracle
from tinymem.research.prefix_reader import prefix_answer_loss
from tinymem.research.pretrained import PretrainedReader


def make_oracle():
    return FixedProjectionMemoryOracle(torch.linspace(-1, 1, 64).reshape(4, 2, 8), torch.randn(16, 8))


def test_only_codes_are_trainable_and_inputs_are_not_aliased():
    codes = torch.zeros(4, 2, 8, requires_grad=True)
    projection = torch.randn(16, 8, requires_grad=True)
    oracle = FixedProjectionMemoryOracle(codes, projection)
    assert list(dict(oracle.named_parameters())) == ["codes"]
    assert list(dict(oracle.named_buffers())) == ["projection"]
    assert not oracle.projection.requires_grad
    with torch.no_grad():
        codes.fill_(0.5)
        projection.zero_()
    assert oracle.codes.count_nonzero() == 0
    assert oracle.projection.count_nonzero() > 0
    oracle(1).sum().backward()
    assert codes.grad is None and projection.grad is None


def test_materialized_history_owns_66_bytes_and_preserves_only_its_gradient():
    oracle = make_oracle()
    state = oracle.state(2)
    assert state.nbytes == 66
    assert state.valid.all() and state.values.shape == (1, 2, 8)
    assert state.values.untyped_storage().data_ptr() != oracle.codes.untyped_storage().data_ptr()
    torch.testing.assert_close(oracle(2), F.linear(state.values[0], oracle.projection), rtol=0, atol=0)
    state.values.sum().backward()
    assert torch.equal(oracle.codes.grad[2], torch.ones(2, 8))
    assert oracle.codes.grad[:2].count_nonzero() == oracle.codes.grad[3:].count_nonzero() == 0
    with torch.no_grad():
        state.values.zero_()
    assert oracle.codes[2].count_nonzero() > 0


def test_projection_preserves_endpoints_and_rejects_nonfinite_values():
    oracle = make_oracle()
    original = deepcopy(oracle.state_dict())
    oracle.project_codes_()
    assert torch.equal(oracle.codes, original["codes"])
    with torch.no_grad():
        oracle.codes[0, 0, :3] = torch.tensor([-2.0, 2.0, 0.25])
    oracle.project_codes_()
    assert torch.equal(oracle.codes[0, 0, :3], torch.tensor([-1.0, 1.0, 0.25]))
    assert torch.equal(oracle.projection, original["projection"])
    for invalid in (float("nan"), float("inf"), -float("inf")):
        with torch.no_grad():
            oracle.codes[0, 0, 0] = invalid
        with pytest.raises(ValueError, match="nonfinite"):
            oracle.project_codes_()


@pytest.mark.parametrize("index", [-1, 4, True, 0.5])
def test_invalid_history_indices_are_rejected(index):
    with pytest.raises(ValueError, match="history"):
        make_oracle().state(index)


@pytest.mark.parametrize("change, error", [
    (lambda c, p: ([], p), TypeError),
    (lambda c, p: (c, []), TypeError),
    (lambda c, p: (c[0], p), ValueError),
    (lambda c, p: (c[:0], p), ValueError),
    (lambda c, p: (c.long(), p.long()), ValueError),
    (lambda c, p: (c, p[:, :7]), ValueError),
    (lambda c, p: (c, p[:0]), ValueError),
    (lambda c, p: (c, p.double()), ValueError),
    (lambda c, p: (c + 2, p), ValueError),
    (lambda c, p: (c * float("nan"), p), ValueError),
    (lambda c, p: (c, p * float("inf")), ValueError),
])
def test_invalid_initial_inputs_are_rejected(change, error):
    with pytest.raises(error):
        FixedProjectionMemoryOracle(*change(torch.zeros(4, 2, 8), torch.ones(16, 8)))


def test_projected_optimizer_and_reload_preserve_fixed_readout():
    oracle = make_oracle()
    projection = oracle.projection.clone()
    optimizer = torch.optim.AdamW(oracle.parameters(), lr=0.1)
    start = oracle.codes.detach().clone()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        sum(oracle(index).square().mean() for index in range(4)).backward()
        optimizer.step()
        oracle.project_codes_()
    assert oracle.codes.abs().max() <= 1
    assert not torch.equal(oracle.codes, start)
    assert torch.equal(oracle.projection, projection)
    restored = make_oracle()
    restored.load_state_dict(oracle.state_dict(), strict=True)
    for index in range(4):
        torch.testing.assert_close(restored(index), oracle(index), rtol=0, atol=0)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_shared_query_losses_reach_only_codes_through_frozen_reader(checkpointing):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(19)
    model = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=32,
    )).requires_grad_(False)
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    reader = PretrainedReader(model, None)
    oracle = make_oracle()
    frozen = deepcopy(model.state_dict())
    projection = oracle.projection.clone()
    memory = oracle(0)
    losses = [prefix_answer_loss(reader, torch.tensor([1, 2]), memory,
                                 torch.tensor([query, 3]), torch.tensor([answer, 4]))
              for query, answer in ((5, 8), (6, 9), (7, 10))]
    torch.stack(losses).mean().backward()
    assert torch.isfinite(oracle.codes.grad).all()
    assert oracle.codes.grad[0].abs().sum() > 0
    assert oracle.codes.grad[1:].count_nonzero() == 0
    assert oracle.projection.grad is None
    assert all(parameter.grad is None for parameter in model.parameters())
    assert torch.equal(oracle.projection, projection)
    assert all(torch.equal(value, frozen[name]) for name, value in model.state_dict().items())


def test_checkpoint_recomputation_preserves_loss_and_code_gradients():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(19)
    original = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=32,
    )).requires_grad_(False)
    initial = make_oracle()
    observed = []
    for checkpointing in (False, True):
        model, oracle = deepcopy(original), deepcopy(initial)
        if checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        reader = PretrainedReader(model, None)
        losses = []
        for index in range(4):
            memory = oracle(index)
            for query in range(5, 14):
                losses.append(prefix_answer_loss(reader, torch.tensor([1, 2]), memory,
                                                 torch.tensor([query, 3]), torch.tensor([query + 9, 4])))
        loss = torch.stack(losses).mean()
        loss.backward()
        observed.append((loss.detach(), oracle.codes.grad.clone()))
    torch.testing.assert_close(observed[0][0], observed[1][0], rtol=0, atol=0)
    torch.testing.assert_close(observed[0][1], observed[1][1], rtol=0, atol=0)
