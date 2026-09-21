from dataclasses import replace

import pytest
import torch

from tinymem.reader.adapter import configure_read_adapter
from tinymem.studies.delta.data import build_dataset
from tinymem.studies.delta.readout import SlotReadout
from tinymem.studies.oracle.training import (
    encode_oracle_episode,
    oracle_validation_losses,
    train_oracle_batch,
)
from tinymem.reader.lora import attach_reader_lora
from tinymem.studies.oracle.state import oracle_records


def _example(reader, split="train"):
    dataset = build_dataset(seed=31, train_prefixes=16, validation_prefixes=16, test_prefixes=16)
    episode = dataset.train[0] if split == "train" else dataset.validation[0]
    return encode_oracle_episode(reader, replace(episode, split=split))


def test_oracle_training_updates_only_bridge_and_lora(tiny_reader):
    source = _example(tiny_reader)
    attach_reader_lora(tiny_reader, rank=2, checkpointing=False)
    adapters = configure_read_adapter(tiny_reader, trainable=True)
    bridge = SlotReadout(memory_width=32, reader_width=16)
    bridge_before = {name: value.detach().clone() for name, value in bridge.state_dict().items()}
    frozen_before = {
        name: value.detach().clone()
        for name, value in tiny_reader.model.named_parameters()
        if not value.requires_grad
    }
    parameters = [*bridge.parameters(), *adapters]
    metric = train_oracle_batch(
        tiny_reader, bridge, (source,), torch.optim.SGD(parameters, lr=0.01),
        adapter_parameters=adapters,
    )

    assert metric["persistent_bytes"] == 258
    assert metric["answer_sequences"] == sum(len(endpoint.queries) for endpoint in source.endpoints)
    assert torch.isfinite(torch.tensor(metric["answer_ce"]))
    assert metric["bridge_gradient_norm"] >= 0
    assert metric["adapter_gradient_norm"] >= 0
    assert any(not torch.equal(value, bridge_before[name]) for name, value in bridge.state_dict().items())
    for name, value in tiny_reader.model.named_parameters():
        if name in frozen_before:
            assert torch.equal(value, frozen_before[name])
            assert value.grad is None


def test_oracle_training_rejects_nontraining_split_without_optimizer_state(tiny_reader):
    source = _example(tiny_reader)
    source = replace(source, split="test")
    attach_reader_lora(tiny_reader, rank=2, checkpointing=False)
    adapters = configure_read_adapter(tiny_reader, trainable=True)
    bridge = SlotReadout(memory_width=32, reader_width=16)
    optimizer = torch.optim.AdamW([*bridge.parameters(), *adapters])
    with pytest.raises(ValueError, match="training examples only"):
        train_oracle_batch(tiny_reader, bridge, (source,), optimizer, adapter_parameters=adapters)
    assert not optimizer.state


def test_oracle_encoder_rejects_wrong_supplied_state_and_keeps_question_tokens_fixed(tiny_reader):
    dataset = build_dataset(seed=31, train_prefixes=16, validation_prefixes=16, test_prefixes=16)
    episode = next(row for row in dataset.train if row.condition == "correction")
    example = encode_oracle_episode(tiny_reader, episode)
    assert [q.after_ids for q in example.endpoints[0].queries] == [q.after_ids for q in example.endpoints[1].queries]
    records = list(oracle_records((episode,)))
    row = records[0]
    changed = row.values.clone()
    changed[0, 0, 0] *= -1
    records[0] = replace(row, values=changed, truth=(1 - row.truth[0], *row.truth[1:]))
    with pytest.raises(ValueError, match="episode truth"):
        encode_oracle_episode(tiny_reader, episode, records)


def test_oracle_validation_reports_each_condition(tiny_reader):
    dataset = build_dataset(seed=37, train_prefixes=16, validation_prefixes=16, test_prefixes=16)
    episodes = tuple(row for row in dataset.validation if row.prefix_id.endswith("p0000"))
    encoded_reader = tiny_reader
    attach_reader_lora(encoded_reader, rank=2, checkpointing=False)
    configure_read_adapter(encoded_reader, trainable=False)
    bridge = SlotReadout(memory_width=32, reader_width=16).requires_grad_(False).eval()
    result = oracle_validation_losses(encoded_reader, bridge, episodes)
    assert set(result) == {"no_write", "repeat", "correction", "balanced"}
    assert result["no_write"]["episodes"] == 1
    assert result["balanced"]["episodes"] == 1
    assert result["repeat"]["episodes"] == 4
    assert result["correction"]["episodes"] == 4
    assert all(row["episode_mean_ce"] >= 0 for row in result.values())
