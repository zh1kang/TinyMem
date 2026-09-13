import pytest
import torch

from test_readout_runner import tiny_reader
from test_independent_fact_scores import _independent_full_forward, _inputs


def test_empty_answer_end_token_matches_full_forward(tiny_reader):
    from tinymem.research.independent_fact_placement_scores import score_complete_answer
    before, history, memory, after, _ = _inputs(tiny_reader)
    answer = torch.tensor([tiny_reader.tokenizer.eos_token_id])
    expected = _independent_full_forward(tiny_reader, before, history, after, answer)
    actual = score_complete_answer(tiny_reader, before, memory, after, answer)
    torch.testing.assert_close(torch.tensor(actual['token_log_probs']), expected, rtol=0, atol=0)
    assert actual['sequence_log_probability'] == sum(actual['token_log_probs'])


def test_auxiliary_scores_keep_generated_tokens_and_mark_appended_end(tiny_reader):
    from tinymem.research.independent_fact_placement_scores import candidate_sequences
    eos = tiny_reader.tokenizer.eos_token_id
    fixed = {'office': [11, eos]}
    assert candidate_sequences(fixed, [11, eos], 'office', eos) == [('office','fixed_candidate',False,[11,eos])]
    truncated = candidate_sequences(fixed, [12,13], 'other', eos)
    assert truncated[-1] == ('other','saved_greedy_completion',True,[12,13,eos])
    empty = candidate_sequences(fixed, [eos], '', eos)
    assert empty[-1] == ('','saved_greedy_completion',False,[eos])


def test_end_only_scoring_requires_frozen_reader(tiny_reader):
    from tinymem.research.independent_fact_placement_scores import score_complete_answer
    before, _, memory, after, _ = _inputs(tiny_reader)
    tiny_reader.model.train()
    with pytest.raises(ValueError,match='evaluation'):
        score_complete_answer(tiny_reader,before,memory,after,torch.tensor([0]))
