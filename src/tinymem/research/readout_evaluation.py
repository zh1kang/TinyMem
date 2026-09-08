"""Before-state evaluation with fresh reads and fixed whole-state controls."""

from collections.abc import Sequence

import torch

from tinymem.data.opaque_qa1 import ROOMS
from tinymem.evaluation.longmemeval import normalized_answer
from tinymem.memory.readout_interface import OneShotEncoder, ReadoutBridge, STATE_BYTES
from tinymem.research.prefix_reader import prefix_answer_loss
from tinymem.research.pretrained import PretrainedReader
from tinymem.research.readout_controls import controlled_state, state_donors
from tinymem.research.readout_interface import encode_readout_history
from tinymem.research.readout_read import read_state_answer
from tinymem.research.readout_runner import EncodedBefore


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
                # Teacher forcing is a separate measurement, never a generation input.
                answer = torch.tensor(query.answer_ids, device=device)
                loss = prefix_answer_loss(reader, before, memory, after, answer)
                first = prefix_answer_loss(reader, before, memory, after, answer[:1])
                stopping = prefix_answer_loss(reader, before, memory, torch.cat((after, answer[:-1])), answer[-1:])
                if not torch.isfinite(torch.stack((loss, first, stopping))).all():
                    raise ValueError("nonfinite evaluation answer loss")
                value = normalized_answer(generated["prediction"])
                results.append({
                    **generated, "history_id": row.history_id, "case_id": query.case_id,
                    "category": query.category, "answer": query.answer,
                    "condition": condition, "donor_history_id": donor,
                    "correct": value == normalized_answer(query.answer),
                    "known_false_abstention": query.category == "update_known" and value == "unknown",
                    "invalid_output": value not in (*ROOMS, "unknown"),
                    "answer_tokens": answer.numel(), "answer_ce": float(loss),
                    "first_answer_ce": float(first), "stopping_ce": float(stopping),
                    "persistent_bytes": STATE_BYTES,
                    "temporary_vector_bytes": memory.numel() * memory.element_size(),
                    "memory_vector_norms": memory.norm(dim=-1).tolist(),
                })
    return results
