"""A literal bit, exact optimizer ownership, and real tiny-Qwen gradient checks."""
from copy import deepcopy

import pytest
import torch

from test_readout_runner import tiny_reader
from tinymem.memory.readout_interface import ReadoutBridge
from tinymem.research.adapted_readout import configure_read_adapter
from tinymem.research.oracle_readout import oracle_variant_state, train_oracle_step
from tinymem.research.reader_adaptation import attach_reader_lora
from tinymem.research.readout_runner import ReadoutQuery


def test_oracle_has_only_two_immutable_owned_payloads():
    left, right = [oracle_variant_state(v, torch.device('cpu')) for v in ('a', 'b')]
    assert left.nbytes == right.nbytes == 66
    assert left.values.dtype == torch.float32 and left.valid.dtype == torch.bool
    assert left.valid.all() and right.valid.all()
    assert left.values.count_nonzero() == right.values.count_nonzero() == 1
    assert left.values[0, 0, 0] == -1 and right.values[0, 0, 0] == 1
    assert torch.equal(left.values, -right.values)
    assert not left.values.requires_grad and not right.values.requires_grad
    another = oracle_variant_state('a', torch.device('cpu'))
    left.values[0, 0, 0] = 0
    assert another.values[0, 0, 0] == -1
    with pytest.raises(ValueError):
        oracle_variant_state('history-id', torch.device('cpu'))


@pytest.mark.parametrize('trainable', [False, True])
def test_oracle_step_matches_full_logit_reference_and_has_no_writer(tiny_reader, trainable):
    attach_reader_lora(tiny_reader, rank=2, checkpointing=False)
    tiny_reader.model.requires_grad_(False).eval()
    adapters = configure_read_adapter(tiny_reader, trainable=trainable)
    bridge = ReadoutBridge(16, 'affine')
    state = oracle_variant_state('b', torch.device('cpu'))
    before = deepcopy(tiny_reader.model.state_dict())
    state_before = state.values.clone()
    queries = (ReadoutQuery('known', 'update_known', 'kitchen', (7, 8), (9, 0)),
               ReadoutQuery('absent', 'update_missing', 'unknown', (8, 7), (10, 11, 0)))
    expected_reader, expected_bridge = deepcopy(tiny_reader), deepcopy(bridge)
    embedding = expected_reader.model.get_input_embeddings()
    memory = expected_bridge(state)
    losses = []
    for q in queries:
        prompt = torch.cat((embedding(torch.tensor([1, 2])), memory, embedding(torch.tensor(q.after_ids))))
        inputs = torch.cat((prompt, embedding(torch.tensor(q.answer_ids[:-1])))).unsqueeze(0)
        logits = expected_reader.model(inputs_embeds=inputs, use_cache=False).logits[0]
        losses.append(torch.nn.functional.cross_entropy(logits[-len(q.answer_ids):].float(), torch.tensor(q.answer_ids)))
    reference = torch.stack(losses).mean()
    reference.backward()
    expected = [*expected_bridge.parameters(), *(p for p in expected_reader.model.parameters() if p.requires_grad)]
    norm = torch.nn.utils.clip_grad_norm_(expected, 1.0)
    parameters = [*bridge.parameters(), *adapters]
    metric = train_oracle_step(tiny_reader, bridge, state, (1, 2), queries,
                              torch.optim.SGD(parameters, lr=.01), adapter_parameters=adapters)
    assert metric['answer_ce'] == pytest.approx(float(reference.detach()))
    assert metric['gradient_norm'] == pytest.approx(float(norm))
    assert metric['write_states'] == 0 and metric['persistent_bytes'] == 66
    for actual, wanted in zip(parameters, expected, strict=True):
        torch.testing.assert_close(actual.grad, wanted.grad, rtol=1e-5, atol=1e-7)
    assert torch.equal(state.values, state_before) and state.values.grad is None
    changed = [name for name, p in tiny_reader.model.named_parameters() if not torch.equal(p, before[name])]
    assert bool(changed) == trainable and all('lora_' in name for name in changed)
    assert all(p.grad is None for p in tiny_reader.model.parameters() if not p.requires_grad)
    with pytest.raises(ValueError, match='optimizer'):
        train_oracle_step(tiny_reader, bridge, state, (1, 2), queries,
                          torch.optim.SGD([next(bridge.parameters())], lr=.01), adapter_parameters=adapters)


@pytest.mark.parametrize('change', ['gradient', 'validity', 'extra_value', 'wrong_bit'])
def test_oracle_rejects_invalid_payload(change):
    from tinymem.research.oracle_readout import check_oracle_state

    state = oracle_variant_state('a', torch.device('cpu'))
    if change == 'gradient':
        state.values.requires_grad_(True)
    elif change == 'validity':
        state.valid[0, 1] = False
    elif change == 'extra_value':
        state.values[0, 1, 1] = .5
    else:
        state.values[0, 0, 0] = 0
    with pytest.raises(ValueError):
        check_oracle_state(state)


def test_known_bit_absent_and_retention_are_separate_gates(tiny_reader):
    from test_memory_updates import episode
    from test_paired_readout_evaluation import records_for
    from scripts.fit_oracle_readout import final_decision
    from tinymem.research.oracle_readout import summarize_oracle
    from tinymem.research.paired_readout_data import build_pairs, encode_pairs

    rows = encode_pairs(tiny_reader, build_pairs([episode(i) for i in range(4)]))
    records = records_for(rows)
    good = summarize_oracle(records, rows)
    assert good['known_bit_readout_passed'] and good['binary_qa_passed'] and good['full_text_passed']
    assert final_decision({'frozen': good, 'adapted': good}) == 'oracle_read_path_ready_for_separate_design'
    for record in records:
        if record['condition'] == 'full_text':
            record['prediction'] = 'wrong'
    retention = summarize_oracle(records, rows)
    assert retention['known_bit_readout_passed'] and retention['binary_qa_passed'] and not retention['full_text_passed']
    assert final_decision({'frozen': good, 'adapted': retention}) == 'known_bit_readable_full_text_retention_failed'
    for record in records:
        if record['condition'] == 'normal' and record['category'] == 'update_missing':
            record['prediction'] = 'wrong'
    absent = summarize_oracle(records, rows)
    assert absent['known_bit_readout_passed'] and not absent['binary_qa_passed']
    assert final_decision({'frozen': good, 'adapted': absent}) == 'known_bit_readable_absent_qa_failed'
    for record in records:
        if record['condition'] == 'normal':
            record['prediction'] = 'wrong'
    failed = summarize_oracle(records, rows)
    assert not failed['known_bit_readout_passed']
    assert final_decision({'frozen': good, 'adapted': failed}) == 'stop_this_oracle_readout_configuration'
