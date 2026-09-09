"""Before-state evaluation with fresh reads and fixed whole-state controls."""

from collections.abc import Sequence

import torch

from tinymem.data.opaque_qa1 import ROOMS
from tinymem.evaluation.longmemeval import normalized_answer
from tinymem.memory.readout_interface import OneShotEncoder, ReadoutBridge, STATE_BYTES
from tinymem.research.prefix_reader import generate_prefix_answer, prefix_answer_loss
from tinymem.research.pretrained import PretrainedReader
from tinymem.research.readout_controls import controlled_state, state_donors
from tinymem.research.readout_interface import encode_readout_history
from tinymem.research.readout_read import read_state_answer
from tinymem.research.readout_runner import EncodedBefore, ReadoutQuery


def _score_answer(reader, before, memory, after, query: ReadoutQuery, generated: dict) -> dict:
    """Score after generation; never supply gold tokens to the generation call."""
    answer = torch.tensor(query.answer_ids, device=reader.model.device)
    loss = prefix_answer_loss(reader, before, memory, after, answer)
    first = prefix_answer_loss(reader, before, memory, after, answer[:1])
    stopping = prefix_answer_loss(reader, before, memory, torch.cat((after, answer[:-1])), answer[-1:])
    if not torch.isfinite(torch.stack((loss, first, stopping))).all():
        raise ValueError("nonfinite evaluation answer loss")
    value = normalized_answer(generated["prediction"])
    return {
        **generated, "case_id": query.case_id, "category": query.category, "answer": query.answer,
        "correct": value == normalized_answer(query.answer),
        "known_false_abstention": query.category == "update_known" and value == "unknown",
        "invalid_output": value not in (*ROOMS, "unknown"),
        "answer_tokens": answer.numel(), "answer_ce": float(loss),
        "first_answer_ce": float(first), "stopping_ce": float(stopping),
    }


@torch.inference_mode()
def evaluate_full_text(
    reader: PretrainedReader, rows: Sequence[EncodedBefore], *, max_new_tokens: int = 8,
) -> list[dict[str, object]]:
    """Use every native history token; retain failures rather than filtering them.

    This capability reference does not have the learned state's byte budget.
    Validate the entire input before any reader forward pass.
    """
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    if any(module.training for module in reader.model.modules()):
        raise ValueError("reader must be in evaluation mode")
    if any(p.requires_grad or p.grad is not None for p in reader.model.parameters()):
        raise ValueError("reader must be frozen with no parameter gradients")
    if not rows:
        raise ValueError("nonempty histories are required")
    histories, cases = set(), set()
    vocabulary = reader.model.config.vocab_size
    context = reader.model.config.max_position_embeddings
    for row in rows:
        if not isinstance(row.history_id, str) or not row.history_id.strip() or row.history_id in histories:
            raise ValueError("history identities must be nonempty and unique")
        histories.add(row.history_id)
        if not row.queries:
            raise ValueError("each history requires nonempty queries")
        for query in row.queries:
            if not isinstance(query.case_id, str) or not query.case_id.strip() or query.case_id in cases:
                raise ValueError("query identities must be nonempty and unique")
            cases.add(query.case_id)
            for ids in (row.before_ids, row.history_ids, query.after_ids, query.answer_ids):
                if not isinstance(ids, tuple) or not ids or any(type(token) is not int or not 0 <= token < vocabulary for token in ids):
                    raise ValueError("native token fragments must be nonempty integer tuples within vocabulary")
            if len(query.answer_ids) < 2:
                raise ValueError("answer tokens must include an answer and stopping token")
            positions = len(row.before_ids) + len(row.history_ids) + len(query.after_ids)
            if positions + max(max_new_tokens, len(query.answer_ids) - 1) > context:
                raise ValueError("full-text read exceeds reader context; no filtering allowed")
    device = reader.model.device
    memory = reader.model.get_input_embeddings().weight.new_empty((0, reader.model.get_input_embeddings().embedding_dim))
    results = []
    for row in rows:
        before = torch.tensor(row.before_ids + row.history_ids, device=device)
        for query in row.queries:
            after = torch.tensor(query.after_ids, device=device)
            generated = generate_prefix_answer(reader, before, memory, after, max_new_tokens=max_new_tokens)
            results.append({
                **_score_answer(reader, before, memory, after, query, generated),
                "history_id": row.history_id, "condition": "full_text",
                "persistent_bytes": None, "history_tokens": len(row.history_ids),
                "native_envelope_tokens": len(row.before_ids) + len(query.after_ids),
                "temporary_vector_bytes": 0, "memory_vector_norms": [],
            })
    return results


@torch.inference_mode()
def evaluate_readout(
    reader: PretrainedReader, encoder: OneShotEncoder, bridge: ReadoutBridge,
    rows: Sequence[EncodedBefore], *, max_new_tokens: int = 8,
) -> list[dict[str, object]]:
    """Write each history once; gold labels never enter the generation call.

    Runtime belongs to the enclosing run, not these order-independent records.
    The state table is transient evaluation storage for the fixed derangement.
    Each read receives only one independently owned 66-byte state.
    """
    donors = state_donors([row.history_id for row in rows])
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    for row in rows:
        if not row.queries or len({q.case_id for q in row.queries}) != len(row.queries):
            raise ValueError("each history requires nonempty unique queries")
        for query in row.queries:
            if len(query.answer_ids) < 2:
                raise ValueError("answer tokens must include an answer and stopping token")
            if len(row.before_ids) + 2 + len(query.after_ids) + max(max_new_tokens, len(query.answer_ids) - 1) > reader.model.config.max_position_embeddings:
                raise ValueError("read exceeds reader context; no filtering allowed")
    encoder.eval()
    bridge.eval()
    device = reader.model.device
    states = {
        row.history_id: controlled_state(encode_readout_history(
            reader, encoder, torch.tensor(row.history_ids, device=device),
        ), "normal")
        for row in rows
    }
    results = []
    for row in rows:
        before = torch.tensor(row.before_ids, device=device)
        for condition in ("normal", "zero", "no_memory", "shuffled"):
            donor = donors[row.history_id] if condition == "shuffled" else row.history_id
            state = controlled_state(states[donor], "normal" if condition == "shuffled" else condition)
            memory = bridge(state)
            for query in row.queries:
                after = torch.tensor(query.after_ids, device=device)
                generated = read_state_answer(
                    reader, bridge, state, before, after, max_new_tokens=max_new_tokens,
                )
                results.append({
                    **_score_answer(reader, before, memory, after, query, generated),
                    "history_id": row.history_id,
                    "condition": condition, "donor_history_id": donor,
                    "persistent_bytes": STATE_BYTES,
                    "temporary_vector_bytes": memory.numel() * memory.element_size(),
                    "memory_vector_norms": memory.norm(dim=-1).tolist(),
                })
    return results
