from copy import deepcopy

import pytest
import torch

from tinymem.data.reader_gate import ReaderCase
from tinymem.research.pretrained import PretrainedReader
from tinymem.research.reader_adaptation import (
    ReaderAnswerTokens, attach_reader_lora, encode_reader_answer, reader_answer_loss,
)


@pytest.fixture
def adaptation_reader():
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    pytest.importorskip("peft")
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(
        {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "hello": 3, "world": 4, "kitchen": 5, "bathroom": 6}, unk_token="[UNK]",
    ))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]", padding_side="left",
        chat_template="hello world",
    )
    torch.manual_seed(4)
    config = transformers.Qwen3Config(vocab_size=7, hidden_size=16, intermediate_size=24, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=64, eos_token_id=2, pad_token_id=0)
    return PretrainedReader(transformers.Qwen3ForCausalLM(config).eval().requires_grad_(False), tokenizer)


def test_answer_position_loss_matches_standard_shifted_hf_loss(adaptation_reader):
    reader = adaptation_reader
    example = ReaderAnswerTokens("id", (3, 4, 3), (5, 2))
    efficient = reader_answer_loss(reader, example)
    full = torch.tensor([[*example.prompt_ids, *example.answer_ids]])
    labels = full.clone()
    labels[:, :len(example.prompt_ids)] = -100
    standard = reader.model(input_ids=full, attention_mask=torch.ones_like(full), labels=labels, use_cache=False).loss
    torch.testing.assert_close(efficient, standard)
    case = ReaderCase("id", "babi_qa1", "history", "hello", "world", "kitchen")
    encoded = encode_reader_answer(reader, case)
    assert encoded.prompt_ids == (3, 4)
    assert encoded.answer_ids == (5, 2)
    reader.model.config.max_position_embeddings = 2
    with pytest.raises(ValueError, match="truncation"):
        encode_reader_answer(reader, case)


def test_only_adapters_change_and_checkpoint_reloads_exactly(adaptation_reader, tmp_path):
    from peft import PeftModel

    reader = adaptation_reader
    original_base = deepcopy(reader.model)
    attach_reader_lora(reader, rank=2, checkpointing=False)
    frozen = {name: value.detach().clone() for name, value in reader.model.named_parameters() if not value.requires_grad}
    trainable = [(name, value) for name, value in reader.model.named_parameters() if value.requires_grad]
    assert trainable and all("lora_" in name for name, _ in trainable)
    before = {name: value.detach().clone() for name, value in trainable}
    optimizer = torch.optim.AdamW([value for _, value in trainable], lr=0.01)
    example = ReaderAnswerTokens("id", (3, 4, 3), (5, 2))
    loss = reader_answer_loss(reader, example)
    loss.backward()
    assert any(value.grad is not None and value.grad.abs().sum() > 0 for _, value in trainable)
    optimizer.step()
    assert any(not torch.equal(before[name], value) for name, value in trainable)
    for name, value in reader.model.named_parameters():
        if name in frozen:
            assert value.grad is None
            assert torch.equal(value, frozen[name])
    reader.model.eval().save_pretrained(tmp_path, save_embedding_layers=False)
    reloaded = PretrainedReader(PeftModel.from_pretrained(original_base, tmp_path).eval(), reader.tokenizer)
    torch.testing.assert_close(reader_answer_loss(reader, example), reader_answer_loss(reloaded, example), rtol=0, atol=0)


def test_checkpointing_preserves_lora_gradients(adaptation_reader):
    reader = adaptation_reader
    attach_reader_lora(reader, rank=2, checkpointing=False)
    example = ReaderAnswerTokens("id", (3, 4, 3), (5, 2))
    reader_answer_loss(reader, example).backward()
    expected = {name: parameter.grad.clone() for name, parameter in reader.model.named_parameters() if parameter.requires_grad}
    reader.model.zero_grad(set_to_none=True)
    reader.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reader_answer_loss(reader, example).backward()
    for name, parameter in reader.model.named_parameters():
        if name in expected:
            torch.testing.assert_close(parameter.grad, expected[name])


def test_frozen_reader_still_trains_memory_prefix_with_checkpointing(adaptation_reader):
    reader = adaptation_reader
    prefix = torch.randn(1, 2, reader.model.config.hidden_size, requires_grad=True)
    tokens = torch.tensor([[3, 4]])
    embedded = reader.model.get_input_embeddings()(tokens)

    def backward():
        output = reader.model(inputs_embeds=torch.cat((prefix, embedded), dim=1), use_cache=False, logits_to_keep=1)
        torch.nn.functional.cross_entropy(output.logits[:, -1].float(), torch.tensor([5])).backward()

    backward()
    expected = prefix.grad.clone()
    assert expected.abs().sum() > 0
    prefix.grad = None
    reader.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reader.model.train()
    backward()
    torch.testing.assert_close(prefix.grad, expected)
    assert all(parameter.grad is None for parameter in reader.model.parameters())
