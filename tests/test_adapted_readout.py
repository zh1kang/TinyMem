"""Tiny-Qwen checks for fixed history features and read-adapter ownership."""
from copy import deepcopy

import pytest
import torch

from tinymem.research import adapted_readout
from tinymem.research.reader_adaptation import attach_reader_lora


def test_history_features_are_detached_cpu_copies_from_a_frozen_reader(tiny_reader):
    hidden = adapted_readout.frozen_history_features(tiny_reader, (3, 4, 5, 6))
    assert hidden.shape == (4, tiny_reader.model.config.hidden_size)
    assert hidden.device.type == "cpu" and hidden.dtype == torch.float32 and not hidden.requires_grad
    torch.testing.assert_close(hidden, adapted_readout.frozen_history_features(tiny_reader, (3, 4, 5, 6)), rtol=0, atol=0)
    with pytest.raises(ValueError, match="valid native token IDs"):
        adapted_readout.frozen_history_features(tiny_reader, ())
    with pytest.raises(ValueError, match="valid native token IDs"):
        adapted_readout.frozen_history_features(tiny_reader, (3, tiny_reader.model.config.vocab_size))
    tiny_reader.model.config.max_position_embeddings = 3
    with pytest.raises(ValueError, match="no truncation"):
        adapted_readout.frozen_history_features(tiny_reader, (3, 4, 5, 6))
    tiny_reader.model.get_input_embeddings().weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        adapted_readout.frozen_history_features(tiny_reader, (3, 4))


def test_read_adapter_owns_exactly_the_lora_parameters_in_each_arm(tiny_reader):
    with pytest.raises(ValueError, match="PEFT reader"):
        adapted_readout.configure_read_adapter(tiny_reader, trainable=True)
    attach_reader_lora(tiny_reader, rank=2, checkpointing=False)
    before = deepcopy(tiny_reader.model.state_dict())
    trainable = adapted_readout.configure_read_adapter(tiny_reader, trainable=True)
    assert trainable and all(p.requires_grad for p in trainable)
    names = {name for name, p in tiny_reader.model.named_parameters() if p.requires_grad}
    assert names and all("lora_" in name for name in names)
    assert len(names) == len(trainable)
    assert not any(m.training for m in tiny_reader.model.modules())
    frozen = adapted_readout.configure_read_adapter(tiny_reader, trainable=False)
    assert frozen == ()
    assert not any(p.requires_grad for p in tiny_reader.model.parameters())
    assert all(torch.equal(before[k], v) for k, v in tiny_reader.model.state_dict().items())
    with pytest.raises(ValueError, match="boolean"):
        adapted_readout.configure_read_adapter(tiny_reader, trainable=1)
