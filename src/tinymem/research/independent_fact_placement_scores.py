"""Complete-answer scoring, including empty and unterminated saved outputs."""

import torch

from tinymem.research.independent_fact_scores import prefix_answer_log_probs
from tinymem.research.prefix_reader import _check_ids, _forward, _prompt_embeddings


def candidate_sequences(fixed, generated_ids, prediction, eos):
    records = [(label, 'fixed_candidate', False, ids) for label, ids in fixed.items()]
    if not generated_ids:
        raise ValueError('saved generation must contain at least one token')
    if generated_ids not in fixed.values():
        appended = generated_ids[-1] != eos
        ids = [*generated_ids, eos] if appended else list(generated_ids)
        records.append((prediction, 'saved_greedy_completion', appended, ids))
    return records


@torch.inference_mode()
def score_complete_answer(reader, before, memory, after, answer):
    if answer.numel() != 1:
        return prefix_answer_log_probs(reader, before, memory, after, answer)
    if any(module.training for module in reader.model.modules()):
        raise ValueError('complete scoring requires evaluation mode')
    if any(p.requires_grad or p.grad is not None for p in reader.model.parameters()):
        raise ValueError('complete scoring requires a frozen reader')
    _check_ids(reader, answer, 'answer_ids')
    if int(answer[0]) != reader.tokenizer.eos_token_id:
        raise ValueError('complete answer must end in native EOS')
    prompt = _prompt_embeddings(reader, before, memory, after)
    # A single-position projection can round differently from the full forward pass.
    logits = _forward(reader, prompt, 0)[0, -1:].float()
    score = float(torch.log_softmax(logits, dim=-1)[0, int(answer[0])])
    return {'answer_ids':answer.tolist(), 'token_log_probs':[score],
            'sequence_log_probability':score, 'mean_answer_ce':-score}
