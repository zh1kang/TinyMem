"""Actual tiny-reader gradients for the complete phase-two write/read path."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from tinymem.memory.delta_slots import DeltaSlotWriter
from tinymem.memory.query_pool_slots import QueryPoolSlotWriter
from tinymem.research.adapted_readout import configure_read_adapter
from tinymem.research.delta_fact_readout import SlotReadout
from tinymem.research.delta_fact_training import Endpoint, TrainingExample, train_batch
from tinymem.research.reader_adaptation import attach_reader_lora
from tinymem.research.adapted_readout import ReadoutQuery


def example():
    return TrainingExample("training-example", (1, 2),
                           (torch.randn(3, 16), torch.randn(2, 16), torch.randn(4, 16)),
                           (Endpoint(1, (ReadoutQuery("a", "known", "a", (3, 4), (5, 0)),)),
                            Endpoint(3, (ReadoutQuery("b", "known", "b", (4, 3), (6, 7, 0)),))))


@pytest.mark.parametrize("kind", ["gated", "delta"])
@pytest.mark.parametrize("width", [8, 32])
def test_full_recurrence_and_lora_gradients_match_unoptimized_full_logits(tiny_reader, kind, width):
    attach_reader_lora(tiny_reader, rank=2, checkpointing=False)
    adapters = configure_read_adapter(tiny_reader, trainable=True)
    writer = (QueryPoolSlotWriter(16, width, 2) if kind == "gated"
              else DeltaSlotWriter(16, width, key_width=4 if width == 8 else 8))
    bridge = SlotReadout(memory_width=width, reader_width=16)
    source = example()
    base_before = {name: p.detach().clone() for name, p in tiny_reader.model.named_parameters()
                   if not p.requires_grad}
    ref_reader, ref_writer, ref_bridge = deepcopy((tiny_reader, writer, bridge))
    state = ref_writer.empty(1)
    states = {}
    for index, features in enumerate(source.features, start=1):
        state = ref_writer(state, features.unsqueeze(0), torch.ones(1, len(features), dtype=torch.bool))
        states[index] = state
    embed = ref_reader.model.get_input_embeddings()
    losses = []
    for endpoint in source.endpoints:
        memory = ref_bridge(states[endpoint.after_write])[0]
        for query in endpoint.queries:
            inputs = torch.cat((embed(torch.tensor(source.before_ids)), memory,
                                embed(torch.tensor(query.after_ids)),
                                embed(torch.tensor(query.answer_ids[:-1])))).unsqueeze(0)
            logits = ref_reader.model(inputs_embeds=inputs, use_cache=False).logits[0]
            losses.append(F.cross_entropy(logits[-len(query.answer_ids):].float(),
                                          torch.tensor(query.answer_ids)))
    reference_loss = torch.stack(losses).mean()
    reference_loss.backward()
    expected = [*ref_writer.parameters(), *ref_bridge.parameters(),
                *(p for p in ref_reader.model.parameters() if p.requires_grad)]
    expected_norm = torch.nn.utils.clip_grad_norm_(expected, 1.0)
    parameters = [*writer.parameters(), *bridge.parameters(), *adapters]
    initial = [p.detach().clone() for p in parameters]
    metrics = train_batch(tiny_reader, writer, bridge, (source,),
                          torch.optim.SGD(parameters, lr=0.01), adapter_parameters=adapters)
    assert metrics["answer_ce"] == pytest.approx(float(reference_loss.detach()), abs=1e-6)
    assert metrics["gradient_norm"] == pytest.approx(float(expected_norm), rel=1e-5)
    assert metrics["persistent_bytes"] == 2 * width * 4 + 2
    for actual, wanted in zip(parameters, expected, strict=True):
        torch.testing.assert_close(actual.grad, wanted.grad, rtol=3e-4, atol=2e-7)
    assert any(not torch.equal(old, new) for old, new in zip(initial, parameters, strict=True))
    for name, p in tiny_reader.model.named_parameters():
        if name in base_before:
            assert torch.equal(p, base_before[name]) and p.grad is None


def test_training_rejects_wrong_ownership_and_attached_features(tiny_reader):
    writer = QueryPoolSlotWriter(16, 8, 2)
    bridge = SlotReadout(memory_width=8, reader_width=16)
    source = example()
    with pytest.raises(ValueError, match="optimizer"):
        train_batch(tiny_reader, writer, bridge, (source,), torch.optim.AdamW(writer.parameters()))
    source.features[0].requires_grad_(True)
    optimizer = torch.optim.AdamW([*writer.parameters(), *bridge.parameters()])
    with pytest.raises(ValueError, match="detached"):
        train_batch(tiny_reader, writer, bridge, (source,), optimizer)


def test_training_rejects_heldout_without_mutating_parameters(tiny_reader):
    writer = QueryPoolSlotWriter(16, 8, 2)
    bridge = SlotReadout(memory_width=8, reader_width=16)
    parameters = [*writer.parameters(), *bridge.parameters()]
    before = [p.detach().clone() for p in parameters]
    optimizer = torch.optim.AdamW(parameters)
    with pytest.raises(ValueError, match="training examples only"):
        train_batch(tiny_reader, writer, bridge, (replace(example(), split="test"),), optimizer)
    assert all(torch.equal(p, old) for p, old in zip(parameters, before, strict=True))
    assert not optimizer.state


def test_batch_weights_episodes_equally_when_query_counts_differ(tiny_reader):
    writer = QueryPoolSlotWriter(16, 8, 2)
    bridge = SlotReadout(memory_width=8, reader_width=16)
    parameters = [*writer.parameters(), *bridge.parameters()]
    optimizer = torch.optim.SGD(parameters, lr=0)
    long = example()
    short = replace(example(), endpoints=(Endpoint(3, long.endpoints[0].queries),))
    individual = [train_batch(tiny_reader, writer, bridge, (row,), optimizer)["answer_ce"]
                  for row in (long, short)]
    combined = train_batch(tiny_reader, writer, bridge, (long, short), optimizer)
    assert combined["answer_sequences"] == 3
    assert combined["answer_ce"] == pytest.approx(sum(individual) / 2, abs=1e-6)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
@pytest.mark.parametrize("kind", ["gated", "delta"])
def test_training_uses_deterministic_mps_backward(tiny_reader, kind):
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        tiny_reader.model.to("mps")
        attach_reader_lora(tiny_reader, rank=2, checkpointing=False)
        adapters = configure_read_adapter(tiny_reader, trainable=True)
        writer = (QueryPoolSlotWriter(16, 8, 2) if kind == "gated"
                  else DeltaSlotWriter(16, 8, key_width=4)).to("mps")
        bridge = SlotReadout(memory_width=8, reader_width=16).to("mps")
        optimizer = torch.optim.SGD([*writer.parameters(), *bridge.parameters(), *adapters], lr=0.01)
        metrics = train_batch(tiny_reader, writer, bridge, (example(),), optimizer,
                              adapter_parameters=adapters)
        assert metrics["persistent_bytes"] == 66
        assert metrics["gradient_norm"] > 0
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn)
