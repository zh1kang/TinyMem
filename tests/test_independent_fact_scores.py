from copy import deepcopy

import pytest
import torch

from test_readout_runner import tiny_reader
from tinymem.research.independent_fact_scores import prefix_answer_log_probs


def _inputs(reader):
    device = reader.model.device
    before = torch.tensor([1, 2], device=device)
    history = torch.tensor([4, 5, 6], device=device)
    after = torch.tensor([7, 8], device=device)
    answer = torch.tensor([11, 12, reader.tokenizer.eos_token_id], device=device)
    memory = reader.model.get_input_embeddings()(history)
    return before, history, memory, after, answer


def _independent_full_forward(reader, before, history, after, answer):
    input_ids = torch.cat((before, history, after, answer[:-1])).unsqueeze(0)
    with torch.inference_mode():
        logits = reader.model(input_ids=input_ids, use_cache=False).logits[0, -answer.numel():].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs.gather(1, answer.long().unsqueeze(1)).squeeze(1)


def test_scores_match_independent_full_forward_logits_and_are_reproducible(tiny_reader):
    before, history, memory, after, answer = _inputs(tiny_reader)
    expected = _independent_full_forward(tiny_reader, before, history, after, answer)

    actual = prefix_answer_log_probs(tiny_reader, before, memory, after, answer)

    torch.testing.assert_close(torch.tensor(actual["token_log_probs"]), expected, rtol=0, atol=0)
    assert actual["answer_ids"] == answer.tolist()
    assert actual["sequence_log_probability"] == sum(actual["token_log_probs"])
    assert actual["mean_answer_ce"] == -sum(actual["token_log_probs"]) / len(actual["token_log_probs"])


def test_scores_support_variable_lengths_and_report_sum_and_mean(tiny_reader):
    before, history, memory, after, _ = _inputs(tiny_reader)
    eos = tiny_reader.tokenizer.eos_token_id

    short = prefix_answer_log_probs(
        tiny_reader, before, memory, after, torch.tensor([11, eos], device=before.device)
    )
    long = prefix_answer_log_probs(
        tiny_reader, before, memory, after, torch.tensor([11, 12, eos], device=before.device)
    )

    for result in (short, long):
        contributions = result["token_log_probs"]
        assert result["sequence_log_probability"] == sum(contributions)
        assert result["mean_answer_ce"] == -sum(contributions) / len(contributions)
    assert len(short["token_log_probs"]) == 2
    assert len(long["token_log_probs"]) == 3


@pytest.mark.parametrize(
    "answer, message",
    [([11], "at least two"), ([11, 12], "EOS")],
)
def test_scores_reject_incomplete_answers(tiny_reader, answer, message):
    before, _, memory, after, _ = _inputs(tiny_reader)
    with pytest.raises(ValueError, match=message):
        prefix_answer_log_probs(tiny_reader, before, memory, after, torch.tensor(answer, device=before.device))


def test_scores_require_frozen_evaluation_reader(tiny_reader):
    before, _, memory, after, answer = _inputs(tiny_reader)
    tiny_reader.model.train()
    try:
        with pytest.raises(ValueError, match="evaluation"):
            prefix_answer_log_probs(tiny_reader, before, memory, after, answer)
    finally:
        tiny_reader.model.eval()

    parameter = next(tiny_reader.model.parameters())
    parameter.requires_grad_(True)
    try:
        with pytest.raises(ValueError, match="frozen"):
            prefix_answer_log_probs(tiny_reader, before, memory, after, answer)
    finally:
        tiny_reader.model.requires_grad_(False)


def test_scores_do_not_mutate_reader_or_memory(tiny_reader):
    before, _, memory, after, answer = _inputs(tiny_reader)
    reader_before = deepcopy(tiny_reader.model.state_dict())
    memory_before = memory.clone()

    prefix_answer_log_probs(tiny_reader, before, memory, after, answer)

    assert all(torch.equal(value, reader_before[name]) for name, value in tiny_reader.model.state_dict().items())
    assert torch.equal(memory, memory_before)
    assert all(parameter.grad is None for parameter in tiny_reader.model.parameters())
