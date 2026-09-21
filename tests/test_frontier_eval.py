import pytest
import torch

from tinymem.reader.adapter import (
    configure_read_adapter,
    frozen_history_features,
)
from tinymem.reader.prefix import generate_prefix_answer
from tinymem.reader.pretrained import PretrainedReader
from tinymem.reader.lora import attach_reader_lora
from tinymem.studies.frontier.eval import (
    base_encoder,
    generate,
    normalized_exact_match,
)
from tinymem.studies.frontier.training import AnswerTokens


@pytest.fixture
def reader():
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(
        {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "a": 3, "b": 4, "c": 5, "d": 6, "e": 7},
        unk_token="[UNK]",
    ))
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]",
    )
    torch.manual_seed(15)
    config = transformers.Qwen3Config(
        vocab_size=8, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=40,
    )
    model = transformers.Qwen3ForCausalLM(config).eval().requires_grad_(False)
    model.generation_config.eos_token_id = [2, 7]
    model.generation_config.pad_token_id = 0
    return PretrainedReader(model, tokenizer)


def _rows(reader):
    embedding = reader.model.get_input_embeddings()
    ids = ([3, 4], [5]), ([3], [4, 5]), ([3, 4, 5], [6])
    memories = [
        embedding(torch.tensor([6, 3])),
        embedding(torch.tensor([4])),
        embedding.weight.new_empty((0, embedding.embedding_dim)),
    ]
    tokens = [AnswerTokens(before, after, (6, 2)) for before, after in ids]
    return tokens, memories


def test_batched_native_generation_matches_serial_mixed_memory_lengths(reader):
    tokens, memories = _rows(reader)
    expected = [
        generate_prefix_answer(
            reader, torch.tensor(item.before), memory, torch.tensor(item.after), max_new_tokens=8,
        )
        for item, memory in zip(tokens, memories, strict=True)
    ]
    actual = generate(reader, tokens, memories, max_new_tokens=8)
    assert [row["generated_ids"] for row in actual] == [row["generated_ids"] for row in expected]
    assert [row["prediction"] for row in actual] == [row["prediction"] for row in expected]


def test_generation_never_reads_answers_and_preserves_reader_state(reader):
    tokens, memories = _rows(reader)
    changed = [AnswerTokens(item.before, item.after, (1, 1, 1, 1)) for item in tokens]
    before = {name: parameter.detach().clone() for name, parameter in reader.model.named_parameters()}
    calls = []

    def inspect(module, args, kwargs, output):
        assert kwargs["use_cache"] is False
        assert output.past_key_values is None
        calls.append(kwargs["inputs_embeds"].shape[1])

    handle = reader.model.register_forward_hook(inspect, with_kwargs=True)
    try:
        original = generate(reader, tokens, memories, max_new_tokens=8)
        altered = generate(reader, changed, memories, max_new_tokens=8)
    finally:
        handle.remove()
    assert altered == original
    assert calls
    assert all(torch.equal(parameter, before[name]) for name, parameter in reader.model.named_parameters())
    assert all(parameter.grad is None for parameter in reader.model.parameters())
    assert len(calls) <= 8 * 2


def test_generation_uses_shared_normalized_exact_match():
    assert normalized_exact_match("The Office!", "the office")
    assert not normalized_exact_match("office", "home")


def test_generation_preflights_context_and_frozen_reader(reader):
    tokens, memories = _rows(reader)
    reader.model.config.max_position_embeddings = 8
    with pytest.raises(ValueError, match="truncation"):
        generate(reader, tokens, memories, max_new_tokens=8)
    reader.model.config.max_position_embeddings = 40
    reader.model.train()
    with pytest.raises(ValueError, match="evaluation mode"):
        generate(reader, tokens, memories)
    reader.model.eval()
    next(reader.model.parameters()).requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        generate(reader, tokens, memories)


def test_generation_rejects_a_training_child(reader):
    tokens, memories = _rows(reader)
    child = next(module for module in reader.model.modules() if module is not reader.model)
    child.train()
    assert not reader.model.training
    with pytest.raises(ValueError, match="evaluation mode"):
        generate(reader, tokens, memories)


def test_base_encoder_restores_the_frozen_adapted_reader(reader):
    pytest.importorskip("peft")
    attach_reader_lora(reader, rank=2, checkpointing=False)
    configure_read_adapter(reader, trainable=False)
    tokens, memories = _rows(reader)
    adapted = generate(reader, tokens, memories, max_new_tokens=2)
    with reader.model.disable_adapter():
        pass
    # Document the PEFT behavior that motivates the helper.
    assert any(parameter.requires_grad for parameter in reader.model.parameters())
    configure_read_adapter(reader, trainable=False)
    def disabled_layers():
        flags = [module.disable_adapters for module in reader.model.modules()
                 if isinstance(getattr(module, "disable_adapters", None), bool)]
        assert flags
        return flags

    with base_encoder(reader):
        assert all(disabled_layers())
        features = frozen_history_features(reader, [3, 4])
    assert features.shape == (2, reader.model.config.hidden_size)
    assert not any(parameter.requires_grad for parameter in reader.model.parameters())
    assert not any(disabled_layers())
    assert generate(reader, tokens, memories, max_new_tokens=2) == adapted


def test_generation_rejects_mismatched_or_invalid_memory(reader):
    tokens, memories = _rows(reader)
    with pytest.raises(ValueError, match="one memory"):
        generate(reader, tokens[:-1], memories)
    with pytest.raises(ValueError, match="reader_width"):
        generate(reader, tokens, [torch.zeros(1, 8), *memories[1:]])
    with pytest.raises(ValueError, match="finite"):
        bad = [memories[0].clone(), memories[1].clone(), memories[2].clone()]
        bad[0][0, 0] = float("nan")
        generate(reader, tokens, bad)
