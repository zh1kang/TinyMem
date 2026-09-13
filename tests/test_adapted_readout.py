"""Actual tiny-Qwen checks for a separate compression-aware training path."""
from copy import deepcopy

import pytest
import torch

from test_readout_runner import tiny_reader
from tinymem.memory.readout_interface import OneShotEncoder, ReadoutBridge
from tinymem.research import adapted_readout
from tinymem.research.reader_adaptation import attach_reader_lora
from tinymem.research.readout_runner import ReadoutQuery


def test_adapted_step_matches_full_logit_reference_and_preserves_base(tiny_reader):
    attach_reader_lora(tiny_reader, rank=2, checkpointing=False)
    tiny_reader.model.requires_grad_(False).eval()
    hidden = adapted_readout.frozen_history_features(tiny_reader, (3, 4, 5, 6))
    before = deepcopy(tiny_reader.model.state_dict())
    adapters = adapted_readout.configure_read_adapter(tiny_reader, trainable=True)
    encoder, bridge = OneShotEncoder(16), ReadoutBridge(16, "affine")
    queries = (ReadoutQuery("a", "update_known", "kitchen", (7, 8), (9, 0)),
               ReadoutQuery("b", "update_missing", "unknown", (8, 7), (10, 11, 0)))
    expected_reader, expected_encoder, expected_bridge = deepcopy(tiny_reader), deepcopy(encoder), deepcopy(bridge)
    features = hidden.unsqueeze(0)
    state = expected_encoder(features, torch.ones(features.shape[:2], dtype=torch.bool))
    memory = expected_bridge(state)
    embedding = expected_reader.model.get_input_embeddings()
    losses = []
    for q in queries:
        prompt = torch.cat((embedding(torch.tensor([1, 2])), memory, embedding(torch.tensor(q.after_ids))))
        inputs = torch.cat((prompt, embedding(torch.tensor(q.answer_ids[:-1])))).unsqueeze(0)
        logits = expected_reader.model(inputs_embeds=inputs, use_cache=False).logits[0]
        targets = torch.tensor(q.answer_ids)
        losses.append(torch.nn.functional.cross_entropy(logits[-len(targets):].float(), targets))
    reference = torch.stack(losses).mean()
    reference.backward()
    expected = list(expected_encoder.parameters()) + list(expected_bridge.parameters()) + [p for p in expected_reader.model.parameters() if p.requires_grad]
    parameters = list(encoder.parameters()) + list(bridge.parameters()) + list(adapters)
    expected_norm = torch.nn.utils.clip_grad_norm_(expected, 1.0)
    result = adapted_readout.train_adapted_step(tiny_reader, encoder, bridge, hidden, (1, 2), queries,
        torch.optim.SGD(parameters, lr=0.01), adapter_parameters=adapters)
    assert result['answer_ce'] == pytest.approx(float(reference.detach()))
    assert result['gradient_norm'] == pytest.approx(float(expected_norm))
    for actual, wanted in zip(parameters, expected, strict=True):
        torch.testing.assert_close(actual.grad, wanted.grad, rtol=1e-5, atol=1e-7)
    changed = [name for name, p in tiny_reader.model.named_parameters() if not torch.equal(p, before[name])]
    assert changed and all('lora_' in name for name in changed)
    assert all(p.grad is None for p in tiny_reader.model.parameters() if not p.requires_grad)
    assert hidden.device.type == 'cpu' and not hidden.requires_grad
    assert result['persistent_bytes'] == 66
    with pytest.raises(ValueError, match='frozen'):
        adapted_readout.frozen_history_features(tiny_reader, (3, 4))


def test_cached_features_match_original_writer_and_serialize_without_history(tiny_reader):
    from safetensors.torch import load, save
    from tinymem.memory.recurrent_slots import LatentSlotState
    from tinymem.research.readout_interface import encode_readout_history
    from tinymem.research.readout_read import read_state_answer

    encoder, bridge = OneShotEncoder(16), ReadoutBridge(16, 'affine')
    hidden = adapted_readout.frozen_history_features(tiny_reader, (3, 4, 5))
    state = encoder(hidden.unsqueeze(0), torch.ones(1, 3, dtype=torch.bool))
    original = encode_readout_history(tiny_reader, encoder, torch.tensor([3, 4, 5]))
    torch.testing.assert_close(state.values, original.values, rtol=0, atol=0)
    values, valid = state.values.detach().clone(), state.valid.clone()
    decoded = load(save({'values': values, 'valid': valid}))
    restored = LatentSlotState(decoded['values'], decoded['valid'])
    assert restored.nbytes == 66
    frozen = LatentSlotState(values, valid)
    for question in ([7, 8], [8, 7]):
        inputs = (tiny_reader, bridge)
        kwargs = dict(before_ids=torch.tensor([1, 2]), question_ids=torch.tensor(question), max_new_tokens=2)
        assert read_state_answer(*inputs, frozen, **kwargs) == read_state_answer(*inputs, restored, **kwargs)
    assert torch.equal(restored.values, values)


def test_frozen_control_and_invalid_ownership(tiny_reader):
    attach_reader_lora(tiny_reader, rank=2, checkpointing=False)
    tiny_reader.model.requires_grad_(False).eval()
    hidden = adapted_readout.frozen_history_features(tiny_reader, (3, 4, 5))
    adapters = adapted_readout.configure_read_adapter(tiny_reader, trainable=False)
    assert adapters == ()
    encoder, bridge = OneShotEncoder(16), ReadoutBridge(16, 'affine')
    queries = (ReadoutQuery('a', 'update_known', 'kitchen', (7, 8), (9, 0)),)
    parameters = list(encoder.parameters()) + list(bridge.parameters())
    optimizer = torch.optim.SGD(parameters, lr=0.01)
    before = deepcopy(tiny_reader.model.state_dict())
    adapted_readout.train_adapted_step(tiny_reader, encoder, bridge, hidden, (1, 2), queries,
        optimizer, adapter_parameters=adapters)
    assert all(torch.equal(before[k], v) for k, v in tiny_reader.model.state_dict().items())
    with pytest.raises(ValueError, match='optimizer'):
        adapted_readout.train_adapted_step(tiny_reader, encoder, bridge, hidden, (1, 2), queries,
            torch.optim.SGD(encoder.parameters(), lr=0.01), adapter_parameters=adapters)
    tiny_reader.model.get_input_embeddings().weight.requires_grad_(True)
    with pytest.raises(ValueError, match='ownership'):
        adapted_readout.train_adapted_step(tiny_reader, encoder, bridge, hidden, (1, 2), queries,
            optimizer, adapter_parameters=adapters)


def test_adapted_checkpoint_reloads_and_frozen_feature_snapshot_is_independent(tiny_reader, tmp_path):
    from peft import PeftModel, set_peft_model_state_dict
    from safetensors.torch import load_file
    from tinymem.research.pretrained import PretrainedReader
    from tinymem.research.prefix_reader import prefix_answer_loss

    base = deepcopy(tiny_reader.model)
    attach_reader_lora(tiny_reader, rank=2, checkpointing=False)
    tiny_reader.model.requires_grad_(False).eval()
    feature_reader = deepcopy(tiny_reader)
    feature_before = deepcopy(feature_reader.model.state_dict())
    hidden = adapted_readout.frozen_history_features(feature_reader, (3, 4, 5))
    adapters = adapted_readout.configure_read_adapter(tiny_reader, trainable=True)
    encoder, bridge = OneShotEncoder(16), ReadoutBridge(16, 'affine')
    queries = (ReadoutQuery('a', 'update_known', 'kitchen', (7, 8), (9, 0)),)
    optimizer = torch.optim.SGD([*encoder.parameters(), *bridge.parameters(), *adapters], lr=0.1)
    adapted_readout.train_adapted_step(tiny_reader, encoder, bridge, hidden, (1, 2), queries,
                                     optimizer, adapter_parameters=adapters)
    assert all(torch.equal(feature_before[k], v) for k, v in feature_reader.model.state_dict().items())
    tiny_reader.model.save_pretrained(tmp_path, save_embedding_layers=False)
    trained = deepcopy(tiny_reader.model.state_dict())
    with torch.no_grad():
        for parameter in adapters:
            parameter.zero_()
    set_peft_model_state_dict(tiny_reader.model, load_file(str(tmp_path / 'adapter_model.safetensors')), adapter_name='default')
    assert all(torch.equal(trained[k], v) for k, v in tiny_reader.model.state_dict().items())
    restored = PretrainedReader(PeftModel.from_pretrained(base, tmp_path).eval(), tiny_reader.tokenizer)
    memory = bridge(encoder(hidden.unsqueeze(0), torch.ones(1, 3, dtype=torch.bool))).detach()
    args = (torch.tensor([1, 2]), memory, torch.tensor([7, 8]), torch.tensor([9, 0]))
    torch.testing.assert_close(prefix_answer_loss(tiny_reader, *args), prefix_answer_loss(restored, *args), rtol=0, atol=0)
