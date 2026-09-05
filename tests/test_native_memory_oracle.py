from copy import deepcopy

import pytest
import torch

from tinymem.data.reader_gate import ReaderCase
from tinymem.research.memory_prompt import NativeMemoryExample
from tinymem.research.native_memory_oracle import QA1_LOCATIONS, NativeMemoryOracle, select_oracle_cases
from tinymem.research.prefix_reader import prefix_answer_loss
from tinymem.research.pretrained import PretrainedReader
from tinymem.research.recurrent_memory import NativeRecurrentMemory


def test_oracle_selection_is_balanced_shortest_and_order_independent():
    cases, examples = [], []
    for answer in QA1_LOCATIONS:
        for suffix, length in (("z", 3), ("b", 2), ("a", 2)):
            key = f"{answer}-{suffix}"
            cases.append(ReaderCase(key, "babi_qa1", key, key, "where?", answer))
            examples.append(NativeMemoryExample(key, (1,), (2,) * length, (3,), (4, 5)))
    selected = select_oracle_cases(cases, examples)
    assert [cases[index].case_id for index in selected] == [f"{answer}-a" for answer in QA1_LOCATIONS]
    assert [cases[::-1][index].case_id for index in select_oracle_cases(cases[::-1], examples[::-1])] == [cases[index].case_id for index in selected]
    with pytest.raises(ValueError, match="align"):
        select_oracle_cases(cases, examples[::-1])
    with pytest.raises(ValueError, match="missing"):
        select_oracle_cases(cases[3:], examples[3:])
    with pytest.raises(ValueError, match="unique"):
        select_oracle_cases(cases + cases[:1], examples + examples[:1])
    selected_cases = [cases[index] for index in selected]
    selected_examples = [examples[index] for index in selected]
    selected_cases[0] = ReaderCase(selected_cases[0].case_id, "babi_qa1", selected_cases[1].history_id, "different", "where?", QA1_LOCATIONS[0])
    with pytest.raises(ValueError, match="distinct history_id"):
        select_oracle_cases(selected_cases, selected_examples)


def test_oracle_payload_is_bounded_independent_and_matches_native_readout():
    oracle = NativeMemoryOracle(6, 16)
    state = oracle.state(2)
    assert state.values.shape == (1, 2, 8) and state.nbytes == 66
    assert state.valid.all() and state.values.abs().max() < 1
    assert state.values.untyped_storage().data_ptr() != oracle.codes.untyped_storage().data_ptr()
    native = NativeRecurrentMemory(16, memory_width=8, slots=2, segment_length=8)
    native.read_projection.load_state_dict(oracle.read_projection.state_dict())
    torch.testing.assert_close(oracle(2), native.memory_vectors(state), rtol=0, atol=0)
    oracle(2).square().sum().backward()
    assert oracle.codes.grad[2].abs().sum() > 0
    assert oracle.codes.grad[:2].count_nonzero() == oracle.codes.grad[3:].count_nonzero() == 0
    for index in (-1, 6, True, 0.5):
        with pytest.raises(ValueError, match="index"):
            oracle(index)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_oracle_answer_fit_updates_codes_and_projection_not_reader(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(31)
    model = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(
        vocab_size=16, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=32,
    )).to(device).requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    reader = PretrainedReader(model, None)
    oracle = NativeMemoryOracle(2, 16).to(device)
    initial = deepcopy(oracle.state_dict())
    frozen = deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(oracle.parameters(), lr=0.01)
    before, after = torch.tensor([1, 3], device=device), torch.tensor([4, 5], device=device)
    answers = [torch.tensor([token, 2], device=device) for token in (6, 7)]
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    observed = []

    def check_cache(module, args, kwargs, output):
        observed.append(kwargs["use_cache"])
        assert output.past_key_values is None

    hook = model.get_decoder().register_forward_hook(check_cache, with_kwargs=True)
    try:
        losses = []
        for _ in range(12):
            optimizer.zero_grad(set_to_none=True)
            loss = sum(prefix_answer_loss(reader, before, oracle(index), after, answer) for index, answer in enumerate(answers)) / len(answers)
            losses.append(float(loss.detach()))
            loss.backward()
            assert torch.isfinite(oracle.codes.grad).all() and (oracle.codes.grad.abs().sum(dim=(1, 2)) > 0).all()
            optimizer.step()
        assert losses[-1] < losses[0]
        assert observed and not any(observed)
        assert all(parameter.grad is None for parameter in model.parameters())
        for name, value in model.state_dict().items():
            assert torch.equal(value, frozen[name])
        for name, value in oracle.state_dict().items():
            assert not torch.equal(value, initial[name])
    finally:
        hook.remove()
        torch.use_deterministic_algorithms(deterministic)


@pytest.mark.parametrize("kwargs", [{"cases": 0}, {"reader_width": False}, {"slots": -1}, {"memory_width": 1.5}])
def test_oracle_rejects_invalid_dimensions(kwargs):
    arguments = {"cases": 6, "reader_width": 16, **kwargs}
    with pytest.raises(ValueError, match="positive integer"):
        NativeMemoryOracle(**arguments)
