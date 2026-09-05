import pytest
import torch

from tinymem.research.prefix_reader import prefix_answer_loss, prefix_answer_losses
from tinymem.research.pretrained import PretrainedReader


@pytest.fixture
def reader():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(9)
    model = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(
        vocab_size=16, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=32,
    )).requires_grad_(False)
    return PretrainedReader(model.eval(), None)


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_batched_loss_and_shared_memory_gradients_match_serial_reads(reader, device, checkpointing):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    reader.model.to(device)
    if checkpointing:
        reader.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        reader.model.train()
    memory = torch.randn(3, 16, device=device, requires_grad=True)
    reference = memory.detach().clone().requires_grad_(True)

    def examples(values):
        ids = lambda values: torch.tensor(values, device=device)
        return [(ids([1]), values[:2], ids([4, 5]), ids([6, 2])),
                (ids([1, 3, 7]), values, ids([4]), ids([8, 9, 2])),
                (ids([1]), values[:0], ids([4, 5, 6]), ids([2])),
                (ids([3, 4]), values[:1], ids([5, 6]), ids([7, 2]))]

    expected = torch.stack([prefix_answer_loss(reader, *item) for item in examples(reference)])
    expected.mean().backward()
    calls = []

    def inspect(module, args, kwargs, output):
        assert not kwargs["use_cache"] and output.past_key_values is None
        calls.append(kwargs)

    handle = reader.model.register_forward_hook(inspect, with_kwargs=True)
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        actual = prefix_answer_losses(reader, examples(memory))
        actual.mean().backward()
    finally:
        handle.remove()
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(memory.grad, reference.grad, atol=2e-6, rtol=2e-4)
    assert all(parameter.grad is None for parameter in reader.model.parameters())
    assert len(calls) == 3
    assert sorted(call["inputs_embeds"].shape[0] for call in calls) == [1, 1, 2]
    for call in calls:
        assert call["attention_mask"].all()
        batch, length = call["attention_mask"].shape
        assert call["position_ids"].tolist() == [list(range(length))] * batch


def test_equal_input_lengths_with_different_target_lengths_use_separate_groups(reader):
    ids = lambda values: torch.tensor(values)
    memory = torch.zeros(2, 16)
    examples = [(ids([1]), memory, ids([4, 5]), ids([6, 2])),
                (ids([1]), memory, ids([4]), ids([8, 9, 2]))]
    calls = []
    handle = reader.model.register_forward_hook(lambda module, args, kwargs, output: calls.append(kwargs), with_kwargs=True)
    try:
        actual = prefix_answer_losses(reader, examples)
    finally:
        handle.remove()
    assert [call["inputs_embeds"].shape for call in calls] == [(1, 6, 16), (1, 6, 16)]
    assert [call["logits_to_keep"] for call in calls] == [2, 3]
    expected = torch.stack([prefix_answer_loss(reader, *item) for item in examples])
    torch.testing.assert_close(actual, expected)


def test_batched_reads_reject_invalid_rows_before_model_forward(reader, monkeypatch):
    monkeypatch.setattr(reader.model, "forward", lambda **kwargs: pytest.fail("invalid read reached model"))
    ids = torch.tensor([1, 2])
    with pytest.raises(ValueError, match="at least one"):
        prefix_answer_losses(reader, [])
    with pytest.raises(ValueError, match="nonempty"):
        prefix_answer_losses(reader, [(ids, torch.zeros(2, 16), ids, ids[:0])])
    with pytest.raises(ValueError, match="truncation"):
        prefix_answer_losses(reader, [(ids, torch.zeros(30, 16), ids, ids)])
    with pytest.raises(ValueError, match="finite"):
        prefix_answer_losses(reader, [(ids, torch.full((2, 16), float("nan")), ids, ids)])
    with pytest.raises(ValueError, match="finite"):
        prefix_answer_losses(reader, [(ids, torch.zeros(2, 16), ids, ids),
                                      (ids, torch.full((2, 16), float("nan")), ids, ids)])
