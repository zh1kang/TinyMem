from copy import deepcopy

import pytest
import torch

from tinymem.reader.prefix import generate_prefix_answer, prefix_answer_loss
from tinymem.reader.pretrained import PretrainedReader


@pytest.fixture
def reader():
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(
        {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "hello": 3, "world": 4, "answer": 5}, unk_token="[UNK]",
    ))
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]")
    torch.manual_seed(6)
    config = transformers.Qwen3Config(vocab_size=6, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
                                     num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=32)
    model = transformers.Qwen3ForCausalLM(config).eval().requires_grad_(False)
    model.generation_config.eos_token_id = [2, 5]
    model.generation_config.pad_token_id = 0
    return PretrainedReader(model, tokenizer)


def test_raw_embedding_prefix_matches_native_loss_and_generation_without_cache(reader):
    from transformers import GenerationConfig

    before, history, after, answer = (torch.tensor(ids) for ids in ([3, 4], [4, 3, 4], [3, 5], [4, 2]))
    memory = reader.model.get_input_embeddings()(history)
    ids = torch.cat((before, history, after))
    full = torch.cat((ids, answer)).unsqueeze(0)
    labels = full.clone()
    labels[:, :len(ids)] = -100
    expected_loss = reader.model(input_ids=full, labels=labels, use_cache=False).loss
    expected_ids = reader.model.generate(input_ids=ids.unsqueeze(0), attention_mask=torch.ones_like(ids.unsqueeze(0)),
                                        generation_config=GenerationConfig(do_sample=False, max_new_tokens=4, eos_token_id=[2, 5], pad_token_id=0))
    before_memory = memory.clone()
    calls = []

    def inspect(module, args, kwargs, output):
        assert not kwargs["use_cache"]
        assert output.past_key_values is None
        assert "past_key_values" not in kwargs
        calls.append(kwargs["inputs_embeds"].shape[1])

    handle = reader.model.register_forward_hook(inspect, with_kwargs=True)
    try:
        torch.testing.assert_close(prefix_answer_loss(reader, before, memory, after, answer), expected_loss)
        actual = generate_prefix_answer(reader, before, memory, after, max_new_tokens=4)
    finally:
        handle.remove()
    assert actual["generated_ids"] == expected_ids[0, len(ids):].tolist()
    assert calls[1:] == list(range(len(ids), len(ids) + len(actual["generated_ids"])))
    assert actual["input_positions"] == len(ids)
    assert actual["memory_positions"] == len(history)
    assert torch.equal(memory, before_memory)
    assert not hasattr(reader, "past_key_values")


@pytest.mark.parametrize("lora", [False, True])
def test_frozen_reader_preserves_memory_gradients_with_checkpointing(reader, lora):
    if lora:
        pytest.importorskip("peft")
        from tinymem.reader.lora import attach_reader_lora

        attach_reader_lora(reader, rank=2, checkpointing=False)
        reader.model.requires_grad_(False).eval()
    before, after, answer = (torch.tensor(ids) for ids in ([3], [4, 5], [3, 2]))
    memory = torch.randn(2, 16, dtype=torch.float64, requires_grad=True)
    frozen = deepcopy(reader.model.state_dict())
    loss = prefix_answer_loss(reader, before, memory, after, answer)
    loss.backward()
    expected = memory.grad.clone()
    assert torch.isfinite(expected).all() and expected.abs().sum() > 0
    assert memory.grad.dtype == memory.dtype
    memory.grad = None
    reader.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reader.model.train()
    prefix_answer_loss(reader, before, memory, after, answer).backward()
    torch.testing.assert_close(memory.grad, expected)
    assert all(parameter.grad is None for parameter in reader.model.parameters())
    for name, value in reader.model.state_dict().items():
        assert torch.equal(value, frozen[name])


def test_empty_memory_is_exact_local_only_read(reader):
    before, after, answer = (torch.tensor(ids) for ids in ([3], [4, 5], [3, 2]))
    empty = torch.empty(0, 16)
    local = torch.cat((before, after, answer)).unsqueeze(0)
    labels = local.clone()
    labels[:, :len(before) + len(after)] = -100
    expected = reader.model(input_ids=local, labels=labels, use_cache=False).loss
    torch.testing.assert_close(prefix_answer_loss(reader, before, empty, after, answer), expected)
    assert generate_prefix_answer(reader, before, empty, after)["memory_positions"] == 0


def test_generation_stops_on_any_configured_eos_id(reader, monkeypatch):
    original = reader.model.forward
    next_tokens = iter([4, 5])

    def forced(*args, **kwargs):
        output = original(*args, **kwargs)
        output.logits.fill_(-100)
        output.logits[..., next(next_tokens)] = 100
        return output

    monkeypatch.setattr(reader.model, "forward", forced)
    result = generate_prefix_answer(reader, torch.tensor([3]), torch.empty(0, 16), torch.tensor([4]), max_new_tokens=4)
    assert result["generated_ids"] == [4, 5]


def test_memory_overflow_in_reader_dtype_fails_before_forward(reader, monkeypatch):
    memory = torch.full((2, 16), 1e100, dtype=torch.float64, requires_grad=True)
    assert torch.isfinite(memory).all()
    monkeypatch.setattr(reader.model, "forward", lambda *args, **kwargs: pytest.fail("overflow must fail before forward"))
    before, after, answer = (torch.tensor(ids) for ids in ([3], [4], [3, 2]))
    with pytest.raises(ValueError, match="finite in the reader dtype"):
        prefix_answer_loss(reader, before, memory, after, answer)
    with pytest.raises(ValueError, match="finite in the reader dtype"):
        generate_prefix_answer(reader, before, memory, after)


def test_overlength_fails_before_materializing_envelope_embeddings(reader, monkeypatch):
    embedding = reader.model.get_input_embeddings()
    monkeypatch.setattr(embedding, "forward", lambda *args, **kwargs: pytest.fail("length must fail before embedding"))
    before, after, answer = (torch.tensor(ids) for ids in ([3], [4], [3, 2]))
    with pytest.raises(ValueError, match="truncation"):
        prefix_answer_loss(reader, before, torch.zeros(30, 16), after, answer)
    with pytest.raises(ValueError, match="truncation"):
        generate_prefix_answer(reader, before, torch.zeros(20, 16), after, max_new_tokens=16)


def test_reader_prefix_rejects_invalid_inputs_without_truncation(reader):
    before, after, answer = (torch.tensor(ids) for ids in ([3], [4, 5], [3, 2]))
    memory = torch.zeros(2, 16)
    with pytest.raises(ValueError, match="reader_width"):
        prefix_answer_loss(reader, before, memory[:, :8], after, answer)
    with pytest.raises(ValueError, match="finite"):
        prefix_answer_loss(reader, before, memory + float("nan"), after, answer)
    with pytest.raises(ValueError, match="integer"):
        prefix_answer_loss(reader, before.float(), memory, after, answer)
    with pytest.raises(ValueError, match="vocabulary"):
        prefix_answer_loss(reader, before, memory, after, answer + 9)
    with pytest.raises(ValueError, match="assistant suffix"):
        prefix_answer_loss(reader, before, memory, after[:0], answer)
    with pytest.raises(ValueError, match="nonempty"):
        prefix_answer_loss(reader, before, memory, after, answer[:0])
    with pytest.raises(ValueError, match="truncation"):
        prefix_answer_loss(reader, before, torch.zeros(31, 16), after, answer)
    with pytest.raises(ValueError, match="truncation"):
        generate_prefix_answer(reader, before, memory, after, max_new_tokens=30)
    with pytest.raises(ValueError, match="positive"):
        generate_prefix_answer(reader, before, memory, after, max_new_tokens=0)
    reader.model.train()
    with pytest.raises(ValueError, match="evaluation mode"):
        generate_prefix_answer(reader, before, memory, after)
