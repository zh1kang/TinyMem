"""Before-state encoding and single-history training for paired readout arms."""

from dataclasses import dataclass

import torch

from tinymem.data.memory_updates import UpdateEpisode, validate_update_episode
from tinymem.memory.readout_interface import OneShotEncoder, ReadoutBridge, STATE_BYTES
from tinymem.research.memory_prompt import encode_memory_example
from tinymem.research.prefix_reader import prefix_answer_loss
from tinymem.research.pretrained import PretrainedReader
from tinymem.research.readout_interface import encode_readout_history
from tinymem.research.update_encoding import encode_update_chunks


@dataclass(frozen=True)
class ReadoutQuery:
    case_id: str
    category: str
    answer: str
    after_ids: tuple[int, ...]
    answer_ids: tuple[int, ...]


@dataclass(frozen=True)
class EncodedBefore:
    history_id: str
    source_group_ids: tuple[str, str]
    source_case_ids: tuple[str, str]
    source_context_sha256: tuple[str, str]
    before_ids: tuple[int, ...]
    history_ids: tuple[int, ...]
    queries: tuple[ReadoutQuery, ...]


def encode_before(reader: PretrainedReader, episode: UpdateEpisode) -> EncodedBefore:
    """Validate serialized labels but tokenize only the original before state."""
    validate_update_episode(episode)
    chunks = encode_update_chunks(reader, episode.initial_chunks, max_chunk_tokens=512)
    history = tuple(token for chunk in chunks for token in chunk)
    native = tuple(encode_memory_example(reader, case) for case in episode.before)
    opening = native[0].before_ids
    for query in native:
        if query.history_ids != history or query.before_ids != opening:
            raise ValueError("before queries must share the exact native history and opening")
        # Include the full-text capability read, not just the shorter soft prefix.
        positions = len(opening) + max(len(history), 2) + len(query.after_ids)
        if positions + max(8, len(query.answer_ids) - 1) > reader.model.config.max_position_embeddings:
            raise ValueError("full-context control exceeds reader context; no filtering allowed")
    return EncodedBefore(
        episode.episode_id, episode.source_group_ids, episode.source_case_ids,
        episode.source_context_sha256, opening, history,
        tuple(ReadoutQuery(q.case_id, case.category, case.answer, q.after_ids, q.answer_ids)
              for case, q in zip(episode.before, native, strict=True)),
    )


def train_readout_step(
    reader: PretrainedReader, encoder: OneShotEncoder, bridge: ReadoutBridge,
    encoded: EncodedBefore, optimizer: torch.optim.Optimizer,
) -> dict[str, float | int]:
    """One query-blind write, mean query CE, and one joint optimizer update."""
    parameters = list(encoder.parameters()) + list(bridge.parameters())
    owned = {id(p) for p in parameters}
    optimized = [p for group in optimizer.param_groups for p in group["params"]]
    if len(optimized) != len(owned) or {id(p) for p in optimized} != owned:
        raise ValueError("optimizer must own exactly the encoder and bridge parameters")
    if any(not p.requires_grad for p in parameters):
        raise ValueError("all encoder and bridge parameters must be trainable")
    if not encoded.queries:
        raise ValueError("a history must have supervised queries")
    encoder.train()
    bridge.train()
    optimizer.zero_grad(set_to_none=True)
    device = reader.model.device
    state = encode_readout_history(reader, encoder, torch.tensor(encoded.history_ids, device=device))
    memory = bridge(state)
    before = torch.tensor(encoded.before_ids, device=device)
    loss = torch.stack([
        prefix_answer_loss(reader, before, memory, torch.tensor(q.after_ids, device=device),
                           torch.tensor(q.answer_ids, device=device))
        for q in encoded.queries
    ]).mean()
    if not torch.isfinite(loss):
        raise ValueError("nonfinite answer loss")
    loss.backward()
    if any(p.grad is None for p in parameters):
        raise ValueError("all encoder and bridge parameters must receive gradients")
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    if any(p.grad is not None for p in reader.model.parameters()):
        raise ValueError("reader-gradient ownership violation")
    result = {
        "answer_ce": float(loss.detach()), "gradient_norm": float(norm),
        "supervised_tokens": sum(len(q.answer_ids) for q in encoded.queries),
        "logical_forward_tokens": len(encoded.history_ids) + sum(
            len(encoded.before_ids) + memory.shape[0] + len(q.after_ids) + len(q.answer_ids) - 1
            for q in encoded.queries),
        "write_states": 1, "persistent_bytes": STATE_BYTES,
    }
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in parameters):
        raise ValueError("optimizer produced nonfinite encoder or bridge parameters")
    return result
