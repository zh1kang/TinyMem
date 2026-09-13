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
        ReadoutQuery(
            f"query-{index}",
            "update_known" if index < 4 else "update_missing",
            "answer",
            (7 + index, 8),
            (9 + index, 0),
        )
        for index in range(6)
    )


def _full_logit_reference(reader, bridge, state, before_ids, queries):
    embedding = reader.model.get_input_embeddings()
    memory = bridge(state)
    losses = []
    for query in queries:
        prompt = torch.cat((embedding(torch.tensor(before_ids)), memory, embedding(torch.tensor(query.after_ids))))
        inputs = torch.cat((prompt, embedding(torch.tensor(query.answer_ids[:-1])))).unsqueeze(0)
        logits = reader.model(inputs_embeds=inputs, use_cache=False).logits[0]
        targets = torch.tensor(query.answer_ids)
        losses.append(torch.nn.functional.cross_entropy(logits[-len(targets):].float(), targets))
    return torch.stack(losses).mean()


def test_placement_states_have_exact_layout_and_byte_budget():
    from tinymem.research.independent_fact_placement import placement_state

    for layout in ("control", "separate_fact0"):
        states = [placement_state(code, torch.device("cpu"), layout) for code in range(16)]
        assert all(state.nbytes == 66 for state in states)
        assert all(torch.equal(state.valid, torch.ones(1, 2, dtype=torch.bool)) for state in states)
        assert len({state.values.numpy().tobytes() for state in states}) == 16
        for code, state in enumerate(states):
            expected = torch.zeros(1, 2, 8)
            if layout == "control":
                expected[0, 0, :4] = torch.tensor([1 if code & (1 << bit) else -1 for bit in range(4)])
            else:
                expected[0, 0, 0] = 1 if code & 1 else -1
                expected[0, 1, 1:4] = torch.tensor([1 if code & (1 << bit) else -1 for bit in range(1, 4)])
            assert torch.equal(state.values, expected)


@pytest.mark.parametrize("layout", ["control", "separate_fact0"])
def test_placement_state_rejects_wrong_layout_code_and_unused_coordinates(layout):
    from tinymem.research.independent_fact_placement import check_placement_state, placement_state

    state = placement_state(5, torch.device("cpu"), layout)
    with pytest.raises(ValueError, match="layout"):
        placement_state(5, torch.device("cpu"), "wrong")
    with pytest.raises(ValueError, match="layout"):
        check_placement_state(state, 5, "wrong")
    with pytest.raises(ValueError, match="code"):
        placement_state(True, torch.device("cpu"), layout)
    with pytest.raises(ValueError, match="code"):
        check_placement_state(state, 16, layout)
    altered = deepcopy(state)
    altered.values[0, 0, 7] = 0.25
    with pytest.raises(ValueError, match="state"):
        check_placement_state(altered, 5, layout)


def test_treatment_matches_independent_full_forward_loss_and_gradients(tiny_reader):
    from tinymem.research.independent_fact_placement import placement_state, train_placement_step

    queries = _queries()
    state = placement_state(5, torch.device("cpu"), "separate_fact0")
    bridge = ReadoutBridge(16, "affine")
    expected_bridge = deepcopy(bridge)
    expected_reader = deepcopy(tiny_reader)
    expected = _full_logit_reference(expected_reader, expected_bridge, state, (1, 2), queries)
    expected.backward()
    expected_norm = torch.nn.utils.clip_grad_norm_(tuple(expected_bridge.parameters()), 1.0)

    result = train_placement_step(
        tiny_reader,
        bridge,
        state,
        5,
        (1, 2),
        queries,
        torch.optim.SGD(bridge.parameters(), lr=0.01),
        adapter_parameters=(),
        layout="separate_fact0",
    )

    assert result["answer_ce"] == pytest.approx(float(expected.detach()))
    assert result["gradient_norm"] == pytest.approx(float(expected_norm))
    assert result["persistent_bytes"] == 66
    assert result["write_states"] == 0
    assert result["supervised_tokens"] == sum(len(query.answer_ids) for query in queries)
    assert result["code"] == 5
    assert result["bridge_gradient_norm"] > 0
    assert result["adapter_gradient_norm"] == 0
    for actual, wanted in zip(bridge.parameters(), expected_bridge.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, wanted.grad, rtol=1e-5, atol=1e-7)


def test_treatment_matches_full_forward_with_lora_and_updates_only_lora(tiny_reader):
    from tinymem.research.adapted_readout import configure_read_adapter
    from tinymem.research.independent_fact_placement import placement_state, train_placement_step

    attach_reader_lora(tiny_reader, rank=2, checkpointing=False)
    tiny_reader.model.requires_grad_(False).eval()
    adapters = configure_read_adapter(tiny_reader, trainable=True)
    bridge = ReadoutBridge(16, "affine")
    state = placement_state(5, torch.device("cpu"), "separate_fact0")
    queries = _queries()
    before_ids = (1, 2)
    reader_before = deepcopy(tiny_reader.model.state_dict())
    expected_reader, expected_bridge = deepcopy(tiny_reader), deepcopy(bridge)
    reference = _full_logit_reference(expected_reader, expected_bridge, state, before_ids, queries)
    reference.backward()
    expected_adapters = tuple(p for p in expected_reader.model.parameters() if p.requires_grad)
    expected_parameters = (*expected_bridge.parameters(), *expected_adapters)
    expected_norm = torch.nn.utils.clip_grad_norm_(expected_parameters, 1.0)

    parameters = (*bridge.parameters(), *adapters)
    result = train_placement_step(
        tiny_reader, bridge, state, 5, before_ids, queries,
        torch.optim.SGD(parameters, lr=0.01), adapter_parameters=adapters,
        layout="separate_fact0",
    )

    assert result["answer_ce"] == pytest.approx(float(reference.detach()))
    assert result["gradient_norm"] == pytest.approx(float(expected_norm))
    assert result["adapter_gradient_norm"] > 0
    for actual, wanted in zip(parameters, expected_parameters, strict=True):
        torch.testing.assert_close(actual.grad, wanted.grad, rtol=1e-5, atol=1e-7)
    changed = [name for name, parameter in tiny_reader.model.named_parameters()
               if not torch.equal(parameter, reader_before[name])]
    assert changed and all("lora_" in name for name in changed)
    assert all(torch.equal(parameter, reader_before[name])
               for name, parameter in tiny_reader.model.named_parameters()
               if "lora_" not in name)
    assert all(parameter.grad is None for parameter in tiny_reader.model.parameters()
               if not parameter.requires_grad)


def test_control_delegates_and_matches_frozen_training_step(tiny_reader):
    from tinymem.research.independent_fact_placement import placement_state, train_placement_step
    from tinymem.research.independent_fact_training import train_fact_step

    queries = _queries()
    reader_a, reader_b = deepcopy(tiny_reader), deepcopy(tiny_reader)
    bridge_a, bridge_b = ReadoutBridge(16, "affine"), ReadoutBridge(16, "affine")
    bridge_b.load_state_dict(deepcopy(bridge_a.state_dict()))
    state_a = placement_state(5, torch.device("cpu"), "control")
    state_b = oracle_state(5, torch.device("cpu"))
    result_a = train_placement_step(
        reader_a, bridge_a, state_a, 5, (1, 2), queries,
        torch.optim.AdamW(bridge_a.parameters(), lr=0.01), adapter_parameters=(), layout="control",
    )
    result_b = train_fact_step(
        reader_b, bridge_b, state_b, 5, (1, 2), queries,
        torch.optim.AdamW(bridge_b.parameters(), lr=0.01), adapter_parameters=(),
    )
    assert result_a == result_b
    for actual, wanted in zip(bridge_a.parameters(), bridge_b.parameters(), strict=True):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)


def test_control_with_lora_matches_frozen_training_step_exactly(tiny_reader):
    from tinymem.research.adapted_readout import configure_read_adapter
    from tinymem.research.independent_fact_placement import placement_state, train_placement_step
    from tinymem.research.independent_fact_training import train_fact_step

    reader_a = deepcopy(tiny_reader)
    attach_reader_lora(reader_a, rank=2, checkpointing=False)
    reader_a.model.requires_grad_(False).eval()
    reader_b = deepcopy(reader_a)
    adapters_a = configure_read_adapter(reader_a, trainable=True)
    adapters_b = configure_read_adapter(reader_b, trainable=True)
    bridge_a, bridge_b = ReadoutBridge(16, "affine"), ReadoutBridge(16, "affine")
    bridge_b.load_state_dict(deepcopy(bridge_a.state_dict()))
    queries = _queries()
    result_a = train_placement_step(
        reader_a, bridge_a, placement_state(5, torch.device("cpu"), "control"), 5,
        (1, 2), queries, torch.optim.AdamW((*bridge_a.parameters(), *adapters_a), lr=0.01),
        adapter_parameters=adapters_a, layout="control",
    )
    result_b = train_fact_step(
        reader_b, bridge_b, oracle_state(5, torch.device("cpu"),), 5,
        (1, 2), queries, torch.optim.AdamW((*bridge_b.parameters(), *adapters_b), lr=0.01),
        adapter_parameters=adapters_b,
    )
    assert result_a == result_b
    for actual, wanted in zip(bridge_a.parameters(), bridge_b.parameters(), strict=True):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    for actual, wanted in zip(adapters_a, adapters_b, strict=True):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)


def test_treatment_preserves_reader_and_state(tiny_reader):
    from tinymem.research.independent_fact_placement import placement_state, train_placement_step

    bridge = ReadoutBridge(16, "affine")
    state = placement_state(3, torch.device("cpu"), "separate_fact0")
    state_values, state_valid = state.values.clone(), state.valid.clone()
    reader_before = deepcopy(tiny_reader.model.state_dict())
    train_placement_step(
        tiny_reader,
        bridge,
        state,
        3,
        (1, 2),
        _queries(),
        torch.optim.SGD(bridge.parameters(), lr=0.01),
        adapter_parameters=(),
        layout="separate_fact0",
    )
    assert all(torch.equal(value, reader_before[name]) for name, value in tiny_reader.model.state_dict().items())
    assert torch.equal(state.values, state_values)
    assert torch.equal(state.valid, state_valid)


def test_treatment_rejects_optimizer_omitting_lora(tiny_reader):
    from tinymem.research.adapted_readout import configure_read_adapter
    from tinymem.research.independent_fact_placement import placement_state, train_placement_step

    attach_reader_lora(tiny_reader, rank=2, checkpointing=False)
    tiny_reader.model.requires_grad_(False).eval()
    adapters = configure_read_adapter(tiny_reader, trainable=True)
    bridge = ReadoutBridge(16, "affine")
    with pytest.raises(ValueError, match="optimizer"):
        train_placement_step(
            tiny_reader, bridge, placement_state(0, torch.device("cpu"), "separate_fact0"), 0,
            (1, 2), _queries(), torch.optim.SGD(bridge.parameters(), lr=0.01),
            adapter_parameters=adapters, layout="separate_fact0",
        )


def test_treatment_rejects_unexpected_frozen_reader_gradient(tiny_reader):
    from tinymem.research.adapted_readout import configure_read_adapter
    from tinymem.research.independent_fact_placement import placement_state, train_placement_step

    attach_reader_lora(tiny_reader, rank=2, checkpointing=False)
    tiny_reader.model.requires_grad_(False).eval()
    adapters = configure_read_adapter(tiny_reader, trainable=True)
    frozen = next(parameter for parameter in tiny_reader.model.parameters() if not parameter.requires_grad)
    frozen.grad = torch.ones_like(frozen)
    bridge = ReadoutBridge(16, "affine")
    try:
        with pytest.raises(ValueError, match="frozen"):
            train_placement_step(
                tiny_reader, bridge, placement_state(0, torch.device("cpu"), "separate_fact0"), 0,
                (1, 2), _queries(), torch.optim.SGD((*bridge.parameters(), *adapters), lr=0.01),
                adapter_parameters=adapters, layout="separate_fact0",
            )
    finally:
        frozen.grad = None


def test_treatment_rejects_control_state_and_invalid_validity(tiny_reader):
    from tinymem.research.independent_fact_placement import (
        check_placement_state, placement_state, train_placement_step,
    )

    bridge = ReadoutBridge(16, "affine")
    control = placement_state(0, torch.device("cpu"), "control")
    with pytest.raises(ValueError, match="state"):
        train_placement_step(
            tiny_reader, bridge, control, 0, (1, 2), _queries(),
            torch.optim.SGD(bridge.parameters(), lr=0.01), adapter_parameters=(),
            layout="separate_fact0",
        )
    invalid = placement_state(0, torch.device("cpu"), "separate_fact0")
    invalid.valid[0, 1] = False
    with pytest.raises(ValueError, match="state"):
        check_placement_state(invalid, 0, "separate_fact0")
