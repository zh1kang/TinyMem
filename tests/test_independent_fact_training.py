"""Behavioral checks for the independent fact training step."""

from copy import deepcopy

import pytest
import torch

from test_readout_runner import tiny_reader
from tinymem.memory.readout_interface import ReadoutBridge
from tinymem.research.independent_fact_data import oracle_state
from tinymem.research.reader_adaptation import attach_reader_lora
from tinymem.research.readout_runner import ReadoutQuery


def _queries() -> tuple[ReadoutQuery, ...]:
    return tuple(
        ReadoutQuery(f"query-{index}", "update_known" if index < 4 else "update_missing",
                     "answer", (7 + index, 8), (9 + index, 0))
        for index in range(6)
    )


def _full_logit_reference(reader, bridge, state, before_ids, queries):
    embedding = reader.model.get_input_embeddings()
    memory = bridge(state)
    losses = []
    for query in queries:
        prompt = torch.cat((embedding(torch.tensor(before_ids)), memory,
                            embedding(torch.tensor(query.after_ids))))
        inputs = torch.cat((prompt, embedding(torch.tensor(query.answer_ids[:-1])))).unsqueeze(0)
        logits = reader.model(inputs_embeds=inputs, use_cache=False).logits[0]
        targets = torch.tensor(query.answer_ids)
        losses.append(torch.nn.functional.cross_entropy(logits[-len(targets):].float(), targets))
    return torch.stack(losses).mean()


def test_fact_step_matches_full_logit_reference_and_updates_only_declared_adapters(tiny_reader):
    from tinymem.research.adapted_readout import configure_read_adapter
    from tinymem.research.independent_fact_training import train_fact_step

    attach_reader_lora(tiny_reader, rank=2, checkpointing=False)
    tiny_reader.model.requires_grad_(False).eval()
    adapters = configure_read_adapter(tiny_reader, trainable=True)
    bridge = ReadoutBridge(16, "affine")
    state = oracle_state(5, torch.device("cpu"))
    state_before = state.values.clone(), state.valid.clone()
    queries = _queries()
    before_ids = (1, 2)
    reader_before = deepcopy(tiny_reader.model.state_dict())
    bridge_before = deepcopy(bridge.state_dict())

    expected_reader, expected_bridge = deepcopy(tiny_reader), deepcopy(bridge)
    reference = _full_logit_reference(expected_reader, expected_bridge, state, before_ids, queries)
    reference.backward()
    expected_parameters = [*expected_bridge.parameters(),
                           *(p for p in expected_reader.model.parameters() if p.requires_grad)]
    expected_norm = torch.nn.utils.clip_grad_norm_(expected_parameters, 1.0)

    parameters = [*bridge.parameters(), *adapters]
    result = train_fact_step(tiny_reader, bridge, state, 5, before_ids, queries,
                             torch.optim.SGD(parameters, lr=0.01),
                             adapter_parameters=adapters)

    assert result["answer_ce"] == pytest.approx(float(reference.detach()))
    assert result["gradient_norm"] == pytest.approx(float(expected_norm))
    assert result["persistent_bytes"] == 66
    assert result["write_states"] == 0
    assert result["supervised_tokens"] == sum(len(query.answer_ids) for query in queries)
    assert result["code"] == 5
    assert result["bridge_gradient_norm"] > 0
    assert result["adapter_gradient_norm"] > 0
    assert set(result) == {"answer_ce", "gradient_norm", "persistent_bytes", "write_states",
                           "supervised_tokens", "bridge_gradient_norm", "adapter_gradient_norm", "code"}
    for actual, expected in zip(parameters, expected_parameters, strict=True):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-7)
    assert torch.equal(state.values, state_before[0])
    assert torch.equal(state.valid, state_before[1])
    changed = [name for name, parameter in tiny_reader.model.named_parameters()
               if not torch.equal(parameter, reader_before[name])]
    assert changed and all("lora_" in name for name in changed)
    assert any(not torch.equal(value, bridge_before[name])
               for name, value in bridge.state_dict().items())
    assert all(parameter.grad is None for parameter in tiny_reader.model.parameters()
               if not parameter.requires_grad)


class _CountingBridge(ReadoutBridge):
    def __init__(self):
        super().__init__(16, "affine")
        self.calls = 0

    def forward(self, state):
        self.calls += 1
        return super().forward(state)


def test_fact_step_projects_shared_state_once_for_all_six_queries(tiny_reader):
    from tinymem.research.independent_fact_training import train_fact_step

    bridge = _CountingBridge()
    queries = _queries()
    parameters = tuple(bridge.parameters())
    result = train_fact_step(tiny_reader, bridge, oracle_state(0, torch.device("cpu")), 0,
                             (1, 2), queries, torch.optim.SGD(parameters, lr=0.01),
                             adapter_parameters=())
    assert bridge.calls == 1
    assert result["adapter_gradient_norm"] == 0


@pytest.mark.parametrize("code", [True, -1, 16, 1.0])
def test_fact_step_rejects_invalid_code(tiny_reader, code):
    from tinymem.research.independent_fact_training import train_fact_step

    bridge = ReadoutBridge(16, "affine")
    with pytest.raises(ValueError, match="code"):
        train_fact_step(tiny_reader, bridge, oracle_state(0, torch.device("cpu")), code,
                        (1, 2), _queries(), torch.optim.SGD(bridge.parameters(), lr=0.01),
                        adapter_parameters=())


def test_fact_step_rejects_state_for_a_different_code(tiny_reader):
    from tinymem.research.independent_fact_training import train_fact_step

    bridge = ReadoutBridge(16, "affine")
    with pytest.raises(ValueError, match="state"):
        train_fact_step(tiny_reader, bridge, oracle_state(0, torch.device("cpu")), 1,
                        (1, 2), _queries(), torch.optim.SGD(bridge.parameters(), lr=0.01),
                        adapter_parameters=())


def test_fact_step_rejects_optimizer_that_omits_declared_adapter(tiny_reader):
    from tinymem.research.adapted_readout import configure_read_adapter
    from tinymem.research.independent_fact_training import train_fact_step

    attach_reader_lora(tiny_reader, rank=2, checkpointing=False)
    tiny_reader.model.requires_grad_(False).eval()
    adapters = configure_read_adapter(tiny_reader, trainable=True)
    bridge = ReadoutBridge(16, "affine")
    with pytest.raises(ValueError, match="optimizer"):
        train_fact_step(tiny_reader, bridge, oracle_state(0, torch.device("cpu")), 0,
                        (1, 2), _queries(), torch.optim.SGD(bridge.parameters(), lr=0.01),
                        adapter_parameters=adapters)


def test_fact_step_rejects_unexpected_frozen_reader_gradient(tiny_reader):
    from tinymem.research.independent_fact_training import train_fact_step

    frozen = next(tiny_reader.model.parameters())
    frozen.grad = torch.ones_like(frozen)
    bridge = ReadoutBridge(16, "affine")
    try:
        with pytest.raises(ValueError, match="frozen"):
            train_fact_step(tiny_reader, bridge, oracle_state(0, torch.device("cpu")), 0,
                            (1, 2), _queries(), torch.optim.SGD(bridge.parameters(), lr=0.01),
                            adapter_parameters=())
    finally:
        frozen.grad = None
