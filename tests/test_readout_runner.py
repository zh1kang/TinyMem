"""Before-only execution with a real tiny random Qwen, not model-quality evidence."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from test_memory_updates import episode
from test_update_runner import WordTokenizer
from tinymem.memory.readout_interface import OneShotEncoder, ReadoutBridge
from tinymem.research.pretrained import PretrainedReader
from tinymem.research.readout_runner import encode_before, train_readout_step


@pytest.fixture
def tiny_reader():
    transformers = pytest.importorskip("transformers")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(
        vocab_size=256, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=2048, eos_token_id=0, attention_dropout=0.0,
    )).requires_grad_(False).eval()
    try:
        yield PretrainedReader(model, WordTokenizer())
    finally:
        torch.set_num_threads(previous)


def test_encoding_preserves_native_before_tokens_only(tiny_reader, monkeypatch):
    source = episode()
    from tinymem.research import readout_runner
    original = readout_runner.encode_memory_example
    seen = []

    def observe(reader, case):
        seen.append(case.case_id)
        return original(reader, case)

    monkeypatch.setattr(readout_runner, "encode_memory_example", observe)
    encoded = encode_before(tiny_reader, source)
    assert seen == [case.case_id for case in source.before]
    assert encoded.history_ids == tuple(tiny_reader.tokenizer.encode("\n\n".join(source.initial_chunks) + "\n\n"))
    assert len(encoded.queries) == 10
    assert not hasattr(encoded, "branches") and not hasattr(encoded, "episode")
    assert [q.case_id for q in encoded.queries] == seen
    tiny_reader.model.config.max_position_embeddings = len(encoded.history_ids)
    with pytest.raises(ValueError, match="context"):
        encode_before(tiny_reader, source)


@pytest.mark.parametrize("kind", ["affine", "gelu"])
def test_before_training_updates_both_modules_and_not_reader(tiny_reader, kind):
    encoded = encode_before(tiny_reader, episode())
    encoder, bridge = OneShotEncoder(16), ReadoutBridge(16, kind)
    initial = [deepcopy(module.state_dict()) for module in (encoder, bridge, tiny_reader.model)]
    parameters = list(encoder.parameters()) + list(bridge.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    for _ in range(2):
        result = train_readout_step(tiny_reader, encoder, bridge, encoded, optimizer)
        assert result["persistent_bytes"] == 66
        assert result["write_states"] == 1
        assert result["supervised_tokens"] == sum(len(q.answer_ids) for q in encoded.queries)
        assert result["gradient_norm"] > 0
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    for module, before in zip((encoder, bridge), initial[:2], strict=True):
        assert any(not torch.equal(value, before[name]) for name, value in module.state_dict().items())
    assert all(torch.equal(value, initial[2][name]) for name, value in tiny_reader.model.state_dict().items())
    assert all(p.grad is None for p in tiny_reader.model.parameters())
    with pytest.raises(ValueError, match="optimizer"):
        train_readout_step(tiny_reader, encoder, bridge, encoded, torch.optim.AdamW(encoder.parameters()))


def test_invalid_before_labels_are_rejected(tiny_reader):
    source = episode()
    bad = replace(source, before=(replace(source.before[0], answer="unknown"), *source.before[1:]))
    with pytest.raises(ValueError, match="replay"):
        encode_before(tiny_reader, bad)
