from copy import deepcopy
import inspect

import pytest
import torch

from tinymem.memory.readout_interface import OneShotEncoder, ReadoutBridge
from tinymem.research.readout_controls import controlled_state
from tinymem.research.readout_interface import encode_readout_history
from tinymem.research.readout_read import read_state_answer
from test_readout_interface import reader


@pytest.mark.parametrize("kind", ["affine", "gelu"])
def test_reads_are_order_independent_and_cache_free(reader, kind):
    encoder, bridge = OneShotEncoder(16), ReadoutBridge(16, kind).eval()
    state = controlled_state(encode_readout_history(reader, encoder, torch.tensor([3, 4, 3])), "normal")
    saved = state.values.clone()
    frozen = deepcopy(reader.model.state_dict())
    before = torch.tensor([3])
    questions = [torch.tensor([4, 5]), torch.tensor([3, 5])]
    calls = []

    def capture(module, args, kwargs):
        assert kwargs["use_cache"] is False
        assert kwargs.get("past_key_values") is None
        length = kwargs["inputs_embeds"].shape[1]
        assert torch.equal(kwargs["position_ids"], torch.arange(length).unsqueeze(0))
        calls.append(length)

    handle = reader.model.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        expected = [read_state_answer(reader, bridge, state, before, q, max_new_tokens=3) for q in questions]
        actual = [read_state_answer(reader, bridge, state, before, q, max_new_tokens=3) for q in reversed(questions)]
        assert actual == list(reversed(expected))
        for control, positions in (("normal", 2), ("zero", 2), ("no_memory", 0)):
            result = read_state_answer(reader, bridge, controlled_state(state, control), before, questions[0], max_new_tokens=3)
            assert result["memory_positions"] == positions
            assert result["input_positions"] == 3 + positions
    finally:
        handle.remove()
    assert calls
    assert torch.equal(state.values, saved)
    assert all(torch.equal(value, frozen[name]) for name, value in reader.model.state_dict().items())
    assert all(p.grad is None for p in reader.model.parameters())
    assert set(inspect.signature(read_state_answer).parameters) == {
        "reader", "bridge", "state", "before_ids", "question_ids", "max_new_tokens"}


def test_read_rejects_training_graph_and_unfrozen_reader(reader):
    encoder, bridge = OneShotEncoder(16), ReadoutBridge(16, "affine").eval()
    state = encode_readout_history(reader, encoder, torch.tensor([3, 4]))
    before, question = torch.tensor([3]), torch.tensor([4, 5])
    with pytest.raises(ValueError, match="detached"):
        read_state_answer(reader, bridge, state, before, question)
    state = controlled_state(state, "normal")
    reader.model.train()
    with pytest.raises(ValueError, match="evaluation"):
        read_state_answer(reader, bridge, state, before, question)
    reader.model.eval()
    next(reader.model.parameters()).requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        read_state_answer(reader, bridge, state, before, question)
