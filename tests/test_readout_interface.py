from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from tinymem.memory.readout_interface import OneShotEncoder, ReadoutBridge, check_readout_state
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.readout_interface import encode_readout_history
from tinymem.research.prefix_reader import generate_prefix_answer, prefix_answer_loss
from tinymem.research.pretrained import PretrainedReader


def test_paired_parameters_and_actual_production_counts():
    arms = []
    for kind in ("affine", "gelu"):
        torch.manual_seed(1337)
        arms.append((OneShotEncoder(2048), ReadoutBridge(2048, kind)))
    assert sum(p.numel() for p in arms[0][0].parameters()) == 135752
    assert sum(p.numel() for p in arms[0][1].parameters()) == 67872
    for left, right in zip(*arms, strict=True):
        assert left.state_dict().keys() == right.state_dict().keys()
        for name, value in left.state_dict().items():
            assert torch.equal(value, right.state_dict()[name])
    state = arms[0][0](torch.randn(1, 5, 2048), torch.ones(1, 5, dtype=torch.bool))
    assert not torch.equal(arms[0][1](state), arms[1][1](state))


@pytest.mark.parametrize("kind", ["affine", "gelu"])
def test_explicit_attention_reference_matches_values_and_gradients(kind):
    torch.manual_seed(7)
    encoder, bridge = OneShotEncoder(16), ReadoutBridge(16, kind)
    other_encoder, other_bridge = deepcopy(encoder), deepcopy(bridge)
    hidden = torch.randn(1, 4, 16, requires_grad=True)
    other_hidden = hidden.detach().clone().requires_grad_()
    valid = torch.tensor([[True, False, True, True]])
    actual = bridge(encoder(hidden, valid))
    tokens = other_hidden[0, [0, 2, 3]]
    vectors = []
    for query in other_encoder.queries:
        logits = torch.stack([torch.dot(query, token) / 4 for token in tokens])
        pooled = sum(weight * token for weight, token in zip(logits.softmax(0), tokens, strict=True))
        code = other_encoder.output_projection(F.gelu(other_encoder.input_projection(pooled))).tanh()
        projected = other_bridge.input_projection(code)
        vectors.append(other_bridge.output_projection(F.gelu(projected) if kind == "gelu" else projected))
    expected = torch.stack(vectors)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(hidden.grad, other_hidden.grad)
    for left, right in zip(list(encoder.parameters()) + list(bridge.parameters()),
                           list(other_encoder.parameters()) + list(other_bridge.parameters()), strict=True):
        torch.testing.assert_close(left.grad, right.grad)


def test_padding_empty_history_and_storage_ownership():
    encoder, bridge = OneShotEncoder(16), ReadoutBridge(16, "affine")
    hidden = torch.randn(1, 4, 16)
    hidden[:, 1] = float("nan")
    hidden.requires_grad_()
    valid = torch.tensor([[True, False, True, True]])
    state = encoder(hidden, valid)
    assert state.nbytes == 66
    assert set(vars(state)) == {"values", "valid"}
    bridge(state).sum().backward()
    assert torch.isfinite(hidden.grad).all() and hidden.grad[:, 1].count_nonzero() == 0
    empty = encoder(torch.full((1, 4, 16), float("nan")), torch.zeros_like(valid))
    assert empty.values.count_nonzero() == 0 and not empty.valid.any()
    assert bridge(empty).shape == (0, 16)
    # A view with the right logical shape still retains another history's bytes.
    backing = torch.zeros(2, 2, 8)
    with pytest.raises(ValueError, match="66 bytes"):
        check_readout_state(LatentSlotState(backing[:1], torch.ones(1, 2, dtype=torch.bool)))


@pytest.mark.parametrize("width", [True, 0, -1, 1.5])
def test_invalid_dimensions(width):
    with pytest.raises(ValueError, match="reader_width"):
        OneShotEncoder(width)
    with pytest.raises(ValueError, match="reader_width"):
        ReadoutBridge(width, "affine")


def test_invalid_features_and_states():
    encoder = OneShotEncoder(16)
    hidden, valid = torch.zeros(1, 3, 16), torch.ones(1, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match="kind"):
        ReadoutBridge(16, "unknown")
    with pytest.raises(ValueError, match="shape"):
        encoder(hidden.expand(2, -1, -1), valid.expand(2, -1))
    with pytest.raises(ValueError, match="boolean"):
        encoder(hidden, valid.float())
    with pytest.raises(TypeError, match="FP32"):
        encoder(hidden.double(), valid)
    with pytest.raises(ValueError, match="finite"):
        encoder(hidden + float("inf"), valid)
    state = encoder(hidden, valid)
    with pytest.raises(ValueError, match="bounded"):
        ReadoutBridge(16, "gelu")(LatentSlotState(state.values + 2, state.valid))
    with pytest.raises(TypeError, match="FP32"):
        ReadoutBridge(16, "gelu").double()(state)


@pytest.fixture
def reader():
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(
        {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "hello": 3, "world": 4, "answer": 5}, unk_token="[UNK]"))
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]")
    torch.manual_seed(6)
    config = transformers.Qwen3Config(vocab_size=6, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
                                     num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=32)
    model = transformers.Qwen3ForCausalLM(config).eval().requires_grad_(False)
    model.generation_config.eos_token_id = 2
    return PretrainedReader(model, tokenizer)


@pytest.mark.parametrize("kind", ["affine", "gelu"])
def test_tiny_training_and_fresh_reader_reload_need_only_state(reader, kind, tmp_path):
    safetensors = pytest.importorskip("safetensors.torch")
    encoder, bridge = OneShotEncoder(16), ReadoutBridge(16, kind)
    frozen = deepcopy(reader.model.state_dict())
    initial = deepcopy(encoder.state_dict())
    initial_bridge = deepcopy(bridge.state_dict())
    history, before, answer = (torch.tensor(ids) for ids in ([3, 4, 3, 4], [3], [4, 2]))
    questions = (torch.tensor([4, 5]), torch.tensor([3, 5]))
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(bridge.parameters()), lr=0.001)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        state = encode_readout_history(reader, encoder, history)
        memory = bridge(state)
        loss = torch.stack([prefix_answer_loss(reader, before, memory, query, answer) for query in questions]).mean()
        loss.backward()
        for p in list(encoder.parameters()) + list(bridge.parameters()):
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        optimizer.step()
    assert any(not torch.equal(value, initial[name]) for name, value in encoder.state_dict().items())
    assert any(not torch.equal(value, initial_bridge[name]) for name, value in bridge.state_dict().items())
    assert all(p.grad is None for p in reader.model.parameters())
    assert all(torch.equal(value, frozen[name]) for name, value in reader.model.state_dict().items())
    with torch.no_grad():
        state = encode_readout_history(reader, encoder, history)
        assert state.nbytes == 66 and state.values.grad_fn is None
        saved = state.values.clone()
        expected = [generate_prefix_answer(reader, before, bridge(state), query, max_new_tokens=3) for query in questions]
        assert torch.equal(saved, state.values)
        assert torch.equal(state.values, encode_readout_history(reader, encoder, history).values)
    safetensors.save_file({"values": state.values, "valid": state.valid}, str(tmp_path / "state.safetensors"))
    safetensors.save_file(bridge.state_dict(), str(tmp_path / "bridge.safetensors"))
    fresh_reader = deepcopy(reader)
    del history, encoder, state, bridge, memory, optimizer, loss
    payload = safetensors.load_file(str(tmp_path / "state.safetensors"))
    restored = LatentSlotState(payload["values"], payload["valid"])
    # File headers are artifact overhead, not part of the runtime tensor payload.
    assert restored.nbytes == 66 and torch.equal(restored.values, saved)
    restored_bridge = ReadoutBridge(16, kind)
    restored_bridge.load_state_dict(safetensors.load_file(str(tmp_path / "bridge.safetensors")))
    actual = [generate_prefix_answer(fresh_reader, before, restored_bridge(restored), query, max_new_tokens=3) for query in questions]
    assert actual == expected


def test_feature_extraction_rejects_unfrozen_reader_and_invalid_history(reader):
    encoder = OneShotEncoder(16)
    for ids in (torch.tensor([], dtype=torch.long), torch.tensor([3.0]), torch.tensor([[3]])):
        with pytest.raises(ValueError, match="integer vector"):
            encode_readout_history(reader, encoder, ids)
    with pytest.raises(ValueError, match="vocabulary"):
        encode_readout_history(reader, encoder, torch.tensor([6]))
    with pytest.raises(ValueError, match="truncation"):
        encode_readout_history(reader, encoder, torch.ones(33, dtype=torch.long))
    reader.model.train()
    with pytest.raises(ValueError, match="evaluation"):
        encode_readout_history(reader, encoder, torch.tensor([3]))
    reader.model.eval()
    next(reader.model.parameters()).requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        encode_readout_history(reader, encoder, torch.tensor([3]))
