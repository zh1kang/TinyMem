"""Tiny random Qwen exercises production code, not scientific model quality."""
from copy import deepcopy
from dataclasses import replace
import re

import pytest
import torch

from test_memory_updates import episode
from tinymem.research.pretrained import PretrainedReader
from tinymem.research.prefix_reader import prefix_answer_loss
from tinymem.research.update_runner import (
    BASELINES, CONTROLS, NEURAL_METHODS, STAGES, check_state, competence_gate,
    encode_update, evaluate_update_episode, new_update_writer, retention_policies,
    retention_states, train_update_step, training_schedule, training_vocabulary,
    update_answer_loss, write_update_states,
)


class WordTokenizer:
    """Reversible local test vocabulary; no pretrained weights or tokenizer."""
    eos_token_id = 0

    def __init__(self):
        self.pieces = [""]
        self.ids = {}

    def encode(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        result = []
        for piece in re.findall(r"\w+|[^\w]", text):
            if piece not in self.ids:
                self.ids[piece] = len(self.pieces)
                self.pieces.append(piece)
            result.append(self.ids[piece])
        return result

    def decode(self, ids, *, skip_special_tokens=False):
        return "".join(self.pieces[int(i)] if int(i) < len(self.pieces) else "?" for i in ids)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and add_generation_prompt and not enable_thinking
        return f"<system>{messages[0]['content']}<user>{messages[1]['content']}<assistant>"


@pytest.fixture
def tiny():
    transformers = pytest.importorskip("transformers")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(
        vocab_size=256, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=2048, eos_token_id=0, attention_dropout=0.0,
    )).requires_grad_(False).eval()
    reader = PretrainedReader(model, WordTokenizer())
    try:
        yield reader, encode_update(reader, episode())
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("method", NEURAL_METHODS)
@pytest.mark.parametrize("checkpointing", [False, True])
def test_actual_training_and_independent_fork_gradients(tiny, method, checkpointing):
    reader, encoded = tiny
    if checkpointing:
        reader.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reader.model.train()
    writer = new_update_writer(reader, method, 17)
    independent = deepcopy(writer)
    loss, trace = update_answer_loss(reader, writer, encoded, trace_gradients=True)
    loss.backward()
    # Independently replay all four prefixes instead of sharing their graph.
    expected = []
    for index, queries in enumerate(encoded.queries):
        state = independent.writer.empty(1)
        chunks = encoded.initial_chunks + (() if index == 0 else (encoded.events[index - 1],))
        for chunk in chunks:
            state = independent.write(reader, state, torch.tensor(chunk))
        memory = independent.memory_vectors(state)
        expected.extend(prefix_answer_loss(reader, torch.tensor(q.before_ids), memory,
                        torch.tensor(q.after_ids), torch.tensor(q.answer_ids)) for q in queries)
    reference = torch.stack(expected).mean()
    reference.backward()
    torch.testing.assert_close(loss, reference)
    for actual, other in zip(writer.parameters(), independent.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, other.grad, atol=2e-6, rtol=2e-4)
    assert len(trace) == 7 and all(s.values.grad.norm() > 0 for s in trace)
    before = deepcopy(writer.state_dict())
    frozen = deepcopy(reader.model.state_dict())
    result = train_update_step(reader, writer, encoded, torch.optim.AdamW(writer.parameters(), lr=1e-3),
                               trace_gradients=True)
    assert result["persistent_bytes"] == 66 and result["write_states"] == 7
    assert len(result["state_gradient_norms"]) == 7
    assert result["logical_forward_tokens"] > result["supervised_tokens"] > 40
    assert any(not torch.equal(before[key], value) for key, value in writer.state_dict().items())
    assert all(torch.equal(frozen[key], value) for key, value in reader.model.state_dict().items())
    assert all(p.grad is None for p in reader.model.parameters())


@pytest.mark.parametrize("method", NEURAL_METHODS)
def test_after_only_loss_reaches_all_four_prefix_writes(tiny, method):
    reader, encoded = tiny
    writer = new_update_writer(reader, method, 23)
    states, trace = write_update_states(reader, writer, encoded, trace_gradients=True)
    query = encoded.queries[3][0]
    prefix_answer_loss(reader, torch.tensor(query.before_ids), writer.memory_vectors(states[3]),
                       torch.tensor(query.after_ids), torch.tensor(query.answer_ids)).backward()
    assert all(s.values.grad is not None and s.values.grad.norm() > 0 for s in (*trace[:4], trace[6]))
    assert trace[4].values.grad is None and trace[5].values.grad is None


@pytest.mark.parametrize("method", (*NEURAL_METHODS, *BASELINES, *CONTROLS))
def test_real_generation_all_methods_shared_state_and_strict_budget(tiny, method, monkeypatch):
    reader, encoded = tiny
    writer = new_update_writer(reader, method, 17) if method in NEURAL_METHODS else None
    policy = retention_policies(reader, training_vocabulary([encoded])).get(method)
    writes = []
    memories = []
    from tinymem.research import update_runner
    generate = update_runner.generate_prefix_answer

    def observe_read(shared_reader, before, memory, after, **kwargs):
        saved = memory.clone()
        result = generate(shared_reader, before, memory, after, **kwargs)
        torch.testing.assert_close(memory, saved)
        memories.append(memory)
        return result

    monkeypatch.setattr(update_runner, "generate_prefix_answer", observe_read)
    if writer:
        original = writer.write
        def observe(shared_reader, state, ids):
            writes.append((state, ids.tolist()))
            return original(shared_reader, state, ids)
        monkeypatch.setattr(writer, "write", observe)
    if method == "fingerprint":
        def forbidden(*args, **kwargs):
            pytest.fail("fingerprint lookup must not use Qwen")
        monkeypatch.setattr(reader.model, "forward", forbidden)
    result = evaluate_update_episode(reader, encoded, method, writer=writer, policy=policy)
    assert len(result["predictions"]) == 40
    if method != "fingerprint":
        assert len(memories) == 40
        for index in range(4):
            assert all(memory is memories[index * 10] for memory in memories[index * 10:(index + 1) * 10])
    assert set(result["metrics"]["rates"]) and result["episode_id"] == encoded.episode.episode_id
    if writer:
        assert len(writes) == 7
        assert all(state is writes[4][0] for state, _ in writes[4:])
        assert [ids for _, ids in writes] == list(map(list, (*encoded.initial_chunks, *encoded.events)))
    if method in (*BASELINES, *NEURAL_METHODS):
        assert 0 < result["persistent_bytes"] <= 66
        assert set(result["states"]) == set(STAGES)
        if writer:
            assert result["persistent_bytes"] == 66
        if method != "fingerprint":
            for index in range(4):
                rows = result["predictions"][index * 10:(index + 1) * 10]
                assert len({row["memory_positions"] for row in rows}) == 1
                if writer:
                    assert rows[0]["memory_positions"] == 2
    else:
        assert result["persistent_bytes"] == (None if method == "full_context" else 0)
    repeated = evaluate_update_episode(reader, encoded, method, writer=writer, policy=policy)
    assert repeated == result


def test_retention_forks_do_not_mutate_before_state(tiny):
    reader, encoded = tiny
    for method, policy in retention_policies(reader, training_vocabulary([encoded])).items():
        states = retention_states(reader, method, policy, encoded)
        before = states[0].payload.clone()
        assert len(states) == 4 and all(s.nbytes <= 66 for s in states)
        assert len({s.payload.data_ptr() for s in states}) == 4
        again = retention_states(reader, method, policy, encoded)
        torch.testing.assert_close(states[0].payload, before)
        for first, second in zip(states, again, strict=True):
            torch.testing.assert_close(first.payload, second.payload)


def oracle_record(row, method="full_context"):
    return {"episode_id": row.episode_id, "method": method, "metrics": {"rates": "untrusted"},
            "predictions": [{"case_id": case.case_id, "prediction": case.answer}
                            for cases in (row.before, *(b.queries for b in row.branches)) for case in cases]}


def test_gate_rescores_raw_outputs_and_requires_every_expected_history():
    row = episode()
    record = oracle_record(row)
    assert competence_gate([record], [row], reader_gate=True)["passed"]
    record["predictions"][10]["prediction"] = "wrong"
    assert not competence_gate([record], [row], reader_gate=True)["passed"]
    record["method"] = "query_pool"
    assert competence_gate([record], [row], reader_gate=False)["passed"]
    record["predictions"][0]["prediction"] = "wrong"
    assert competence_gate([record], [row], reader_gate=False)["interpretation"] == "competence_limited"
    for records, episodes in (([], [row]), ([record, record], [row]), ([record], [row, episode(1)])):
        with pytest.raises(ValueError, match="expected"):
            competence_gate(records, episodes, reader_gate=False)
    record["predictions"].pop()
    with pytest.raises(ValueError):
        competence_gate([record], [row], reader_gate=False)


def test_schedule_vocabulary_and_invalid_training_ownership(tiny):
    reader, encoded = tiny
    schedule = training_schedule(5, 12, 17)
    assert schedule == training_schedule(5, 12, 17)
    assert sorted(schedule[:5]) == sorted(schedule[5:10]) == list(range(5))
    for args in ((True, 2, 0), (1, 0, 0), (1, 2, -1)):
        with pytest.raises(ValueError):
            training_schedule(*args)
    vocab = training_vocabulary([encoded])
    assert vocab == sorted(set(token for chunk in (*encoded.initial_chunks, *encoded.events) for token in chunk))
    assert reader.tokenizer.ids["unknown"] not in vocab
    writer = new_update_writer(reader, "query_pool", 17)
    with pytest.raises(ValueError, match="exactly"):
        train_update_step(reader, writer, encoded, torch.optim.AdamW(writer.writer.parameters()))
    next(writer.parameters()).requires_grad_(False)
    with pytest.raises(ValueError, match="trainable"):
        train_update_step(reader, writer, encoded, torch.optim.AdamW(writer.parameters()))
    state = writer.writer.empty(1)
    with pytest.raises(ValueError, match="bounded"):
        check_state(replace(state, values=torch.full_like(state.values, float("nan"))))


def test_encode_preflights_full_context_without_filtering(tiny):
    reader, encoded = tiny
    record = encoded.token_record()
    assert len(record["events"]) == 3 and len(record["answer_ids"]) == 4
    reader.model.config.max_position_embeddings = len(encoded.queries[0][0].before_ids) + 80
    with pytest.raises(ValueError, match="context"):
        encode_update(reader, encoded.episode)
