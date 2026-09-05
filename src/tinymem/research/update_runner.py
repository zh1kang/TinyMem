"""Single-budget update training and reading with the existing memory modules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import random

import torch

from tinymem.data.memory_updates import UpdateEpisode, validate_update_episode
from tinymem.evaluation.memory_updates import CountRate, UpdatePrediction, score_update_episode
from tinymem.memory.fingerprint_facts import FingerprintFactRetention
from tinymem.memory.latest_fact_tokens import append_latest_fact_sentence
from tinymem.memory.packed_tokens import PackedTokenRetention, PackedTokenState
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.memory.template_facts import TemplateFactRetention
from tinymem.memory.vocabulary_tokens import VocabularyTokenRetention
from tinymem.research.memory_prompt import NativeMemoryExample, encode_memory_example
from tinymem.research.prefix_reader import generate_prefix_answer, prefix_answer_loss
from tinymem.research.pretrained import PretrainedReader
from tinymem.research.recurrent_memory import NativeRecurrentMemory
from tinymem.research.update_encoding import encode_update_chunks


BUDGET = 66
STAGES = ("before", "addition", "repetition", "correction")
NEURAL_METHODS = ("query_pool", "mean_pool")
BASELINES = ("recent_native", "recent_vocabulary", "latest_vocabulary", "latest_template", "fingerprint")
CONTROLS = ("full_context", "no_memory")


@dataclass(frozen=True)
class EncodedUpdate:
    episode: UpdateEpisode
    initial_chunks: tuple[tuple[int, ...], ...]
    events: tuple[tuple[int, ...], ...]
    queries: tuple[tuple[NativeMemoryExample, ...], ...]

    def token_record(self) -> dict:
        return {
            "episode_id": self.episode.episode_id, "initial_chunks": self.initial_chunks, "events": self.events,
            "before_ids": self.queries[0][0].before_ids,
            "after_ids": [query.after_ids for query in self.queries[0]],
            "answer_ids": [[query.answer_ids for query in stage] for stage in self.queries],
        }


def encode_update(reader: PretrainedReader, episode: UpdateEpisode) -> EncodedUpdate:
    validate_update_episode(episode)
    initial = encode_update_chunks(reader, episode.initial_chunks, max_chunk_tokens=512)
    events, queries = [], []
    cases = (episode.before, *(branch.queries for branch in episode.branches))
    for index, stage in enumerate(cases):
        chunks = initial
        if index:
            chunks = encode_update_chunks(reader, (*episode.initial_chunks, episode.branches[index - 1].event), max_chunk_tokens=512)
            if chunks[:4] != initial:
                raise ValueError("event changed the shared write prefix")
            events.append(chunks[-1])
        encoded = tuple(encode_memory_example(reader, case) for case in stage)
        flattened = tuple(token for chunk in chunks for token in chunk)
        if any(query.history_ids != flattened for query in encoded):
            raise ValueError("write chunks do not match the native query history")
        if any(query.before_ids != encoded[0].before_ids for query in encoded):
            raise ValueError("queries must share a native opening")
        if index and [(q.before_ids, q.after_ids) for q in encoded] != [(q.before_ids, q.after_ids) for q in queries[0]]:
            raise ValueError("the event changed a query envelope")
        # Preflight both the soft-code read and the full-context capability check.
        for query in encoded:
            if len(query.before_ids) + len(query.history_ids) + len(query.after_ids) + max(8, len(query.answer_ids) - 1) > reader.model.config.max_position_embeddings:
                raise ValueError("full-context control exceeds the reader context; no filtering allowed")
        queries.append(encoded)
    return EncodedUpdate(episode, initial, tuple(events), tuple(queries))


def new_update_writer(reader: PretrainedReader, method: str, seed: int) -> NativeRecurrentMemory:
    if method not in NEURAL_METHODS or type(seed) is not int or seed < 0:
        raise ValueError("expected a declared neural method and nonnegative integer seed")
    torch.manual_seed(seed)
    width = reader.model.get_input_embeddings().embedding_dim
    writer = NativeRecurrentMemory(width, memory_width=8, slots=2, segment_length=512,
                                   writer_kind=method, aggregation_width=64 if method == "query_pool" else 82).to(reader.model.device)
    projection = torch.empty(width, 8).uniform_(-1 / math.sqrt(8), 1 / math.sqrt(8),
                                               generator=torch.Generator(device="cpu").manual_seed(seed))
    with torch.no_grad():
        writer.read_projection.weight.copy_(projection.to(reader.model.device))
    check_state(writer.writer.empty(1))
    return writer


def check_state(state: LatentSlotState | PackedTokenState) -> None:
    if isinstance(state, LatentSlotState):
        if state.values.shape != (1, 2, 8) or state.valid.shape != (1, 2) or state.values.dtype != torch.float32:
            raise ValueError("learned state must be two width-eight FP32 slots")
        if state.nbytes != BUDGET:
            raise ValueError("learned state must own exactly 66 persistent bytes")
        if not torch.isfinite(state.values).all() or (state.values.abs() > 1).any():
            raise ValueError("learned state must contain finite bounded values")
    elif not isinstance(state, PackedTokenState) or not 0 < state.nbytes <= BUDGET:
        raise ValueError("packed state exceeds the persistent-byte budget")


def write_update_states(
    reader: PretrainedReader, writer: NativeRecurrentMemory, encoded: EncodedUpdate,
    *, trace_gradients: bool = False,
) -> tuple[tuple[LatentSlotState, ...], list[LatentSlotState]]:
    """Four common writes, then three independent one-event forks, never queries."""
    state = writer.writer.empty(1)
    trace = []
    for chunk in encoded.initial_chunks:
        state = writer.write(reader, state, torch.tensor(chunk, device=reader.model.device))
        check_state(state)
        if trace_gradients:
            state.values.retain_grad()
        trace.append(state)
    states = [state]
    for event in encoded.events:
        updated = writer.write(reader, state, torch.tensor(event, device=reader.model.device))
        check_state(updated)
        if trace_gradients:
            updated.values.retain_grad()
        trace.append(updated)
        states.append(updated)
    return tuple(states), trace


def update_answer_loss(
    reader: PretrainedReader, writer: NativeRecurrentMemory, encoded: EncodedUpdate,
    *, trace_gradients: bool = False,
) -> tuple[torch.Tensor, list[LatentSlotState]]:
    states, trace = write_update_states(reader, writer, encoded, trace_gradients=trace_gradients)
    stages = []
    for state, examples in zip(states, encoded.queries, strict=True):
        memory = writer.memory_vectors(state)
        losses = []
        for example in examples:
            before, after, answer = (torch.tensor(ids, device=reader.model.device)
                                     for ids in (example.before_ids, example.after_ids, example.answer_ids))
            losses.append(prefix_answer_loss(reader, before, memory, after, answer))
        stages.append(torch.stack(losses).mean())
    return torch.stack(stages).mean(), trace


def training_schedule(worlds: int, steps: int, seed: int) -> list[int]:
    if any(type(value) is not int for value in (worlds, steps, seed)) or min(worlds, steps) <= 0 or seed < 0:
        raise ValueError("positive world/step counts and a nonnegative integer seed are required")
    rng, schedule = random.Random(seed), []
    while len(schedule) < steps:
        indices = list(range(worlds))
        rng.shuffle(indices)
        schedule.extend(indices)
    return schedule[:steps]


def train_update_step(reader, writer, encoded, optimizer, *, trace_gradients=False) -> dict:
    """The production step, also exercised with tiny random Qwen in tests."""
    owned = {id(parameter) for parameter in writer.parameters()}
    optimized = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if {id(parameter) for parameter in optimized} != owned or len(optimized) != len(owned):
        raise ValueError("optimizer must own exactly the writer/projection parameters")
    if any(not parameter.requires_grad for parameter in writer.parameters()):
        raise ValueError("all writer/projection parameters must be trainable")
    if any(parameter.requires_grad or parameter.grad is not None for parameter in reader.model.parameters()):
        raise ValueError("reader must be frozen with no accumulated parameter gradients")
    optimizer.zero_grad(set_to_none=True)
    loss, trace = update_answer_loss(reader, writer, encoded, trace_gradients=trace_gradients)
    if not torch.isfinite(loss):
        raise ValueError("nonfinite answer loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(writer.parameters(), 1.0, error_if_nonfinite=True)
    if any(parameter.grad is not None for parameter in reader.model.parameters()):
        raise ValueError("reader-gradient ownership violation")
    result = {"answer_ce": float(loss.detach()), "gradient_norm": float(norm),
              "supervised_tokens": sum(len(query.answer_ids) for stage in encoded.queries for query in stage),
              "logical_forward_tokens": sum(len(chunk) for chunk in encoded.initial_chunks) + sum(map(len, encoded.events))
                  + sum(int(state.valid.sum()) for state in trace[:3]) + 3 * int(trace[3].valid.sum())
                  + sum(len(query.before_ids) + int(state.valid.sum()) + len(query.after_ids) + len(query.answer_ids) - 1
                        for state, stage in zip((trace[3], *trace[4:]), encoded.queries, strict=True) for query in stage),
              "write_states": len(trace), "persistent_bytes": BUDGET}
    if trace_gradients:
        gradients = [None if state.values.grad is None else float(state.values.grad.norm()) for state in trace]
        if len(gradients) != 7 or any(value is None or not math.isfinite(value) or value <= 0 for value in gradients):
            raise ValueError("all four prefix and three event states require nonzero finite gradients")
        result["state_gradient_norms"] = gradients
    optimizer.step()
    if any(not torch.isfinite(parameter).all() for parameter in writer.parameters()):
        raise ValueError("optimizer produced nonfinite writer parameters")
    return result


def training_vocabulary(encoded: Sequence[EncodedUpdate]) -> list[int]:
    return sorted({token for row in encoded for chunk in (*row.initial_chunks, *row.events) for token in chunk})


def retention_policies(reader, vocabulary: Sequence[int]) -> dict:
    vocab_size = reader.model.config.vocab_size
    capacity = BUDGET * 8 // max(1, (vocab_size - 1).bit_length())
    while PackedTokenRetention(capacity, vocab_size).payload_bytes > BUDGET:
        capacity -= 1
    return {"recent_native": PackedTokenRetention(capacity, vocab_size),
            "recent_vocabulary": VocabularyTokenRetention(BUDGET, vocab_size, vocabulary),
            "latest_vocabulary": VocabularyTokenRetention(BUDGET, vocab_size, vocabulary),
            "latest_template": TemplateFactRetention(BUDGET), "fingerprint": FingerprintFactRetention(BUDGET)}


def write_retention_state(reader, method, policy, state, text, ids):
    if method in ("recent_native", "recent_vocabulary"):
        tokens = torch.tensor([ids], dtype=torch.long)
        state = policy.append(state, tokens, torch.ones_like(tokens, dtype=torch.bool))
    else:
        for sentence in text.splitlines():
            state = (append_latest_fact_sentence(policy, reader.tokenizer, state, sentence)
                     if method == "latest_vocabulary" else policy.append_sentence(state, sentence))
    check_state(state)
    return state


def retention_states(reader, method, policy, encoded) -> tuple[PackedTokenState, ...]:
    state = policy.empty(1, device="cpu") if method in BASELINES[:3] else policy.empty()
    for text, ids in zip(encoded.episode.initial_chunks, encoded.initial_chunks, strict=True):
        state = write_retention_state(reader, method, policy, state, text, ids)
    return (state, *(write_retention_state(reader, method, policy, state, branch.event, ids)
                     for branch, ids in zip(encoded.episode.branches, encoded.events, strict=True)))


@torch.inference_mode()
def evaluate_update_episode(
    reader: PretrainedReader, encoded: EncodedUpdate, method: str,
    *, writer: NativeRecurrentMemory | None = None, policy=None,
) -> dict:
    if reader.model.training or any(parameter.requires_grad for parameter in reader.model.parameters()):
        raise ValueError("evaluation requires an eval-mode frozen reader")
    if method in NEURAL_METHODS:
        if writer is None or writer.writer_kind != method or policy is not None:
            raise ValueError("neural evaluation requires its independently trained writer")
        states, _ = write_update_states(reader, writer, encoded)
    elif method in BASELINES:
        if policy is None or writer is not None:
            raise ValueError("bounded reference requires its retention policy, not a neural writer")
        states = retention_states(reader, method, policy, encoded)
    elif method in CONTROLS and writer is None and policy is None:
        states = (None,) * 4
    else:
        raise ValueError("unsupported method or unexpected state owner")
    predictions, records, snapshots = [], [], {}
    embedding, device = reader.model.get_input_embeddings(), reader.model.device
    for stage, examples, state in zip(STAGES, encoded.queries, states, strict=True):
        if state is not None:
            check_state(state)
            snapshots[stage] = ({"values": state.values.tolist(), "valid": state.valid.tolist(), "nbytes": state.nbytes}
                                if isinstance(state, LatentSlotState) else {"payload": state.payload.tolist(), "nbytes": state.nbytes})
        if method in NEURAL_METHODS:
            memory = writer.memory_vectors(state)
        elif method == "fingerprint":
            memory = None
        else:
            if method == "full_context":
                ids = examples[0].history_ids
            elif method == "no_memory":
                ids = ()
            elif method == "latest_template":
                ids = reader.tokenizer.encode(policy.text(state), add_special_tokens=False)
            else:
                tokens, valid = policy.materialize(state, pad_id=0)
                ids = tokens[0, valid[0]].tolist()
            memory = embedding(torch.tensor(ids, dtype=torch.long, device=device))
        for index, example in enumerate(examples):
            if method == "fingerprint":
                generated = {"prediction": policy.lookup(state, encoded.episode.entities[index]), "decoder": "handcrafted_lookup"}
                ce = None
            else:
                before, after, answer = (torch.tensor(ids, device=device) for ids in (example.before_ids, example.after_ids, example.answer_ids))
                generated = generate_prefix_answer(reader, before, memory, after, max_new_tokens=8)
                ce = float(prefix_answer_loss(reader, before, memory, after, answer))
                if not math.isfinite(ce):
                    raise ValueError("evaluation produced nonfinite answer CE")
            predictions.append(UpdatePrediction(example.case_id, generated["prediction"]))
            records.append({"case_id": example.case_id, "stage": stage, **generated, "answer_ce": ce})
    return {"episode_id": encoded.episode.episode_id, "method": method, "predictions": records,
            "metrics": score_update_episode(encoded.episode, predictions).to_dict(), "states": snapshots,
            "persistent_bytes": None if method == "full_context" else (0 if method == "no_memory" else max(state.nbytes for state in states)),
            "reader_kind": "handcrafted" if method == "fingerprint" else "shared_frozen_reader"}


def competence_gate(records: Sequence[Mapping], episodes: Sequence[UpdateEpisode], *, reader_gate: bool) -> dict:
    """Rescore complete predictions against authoritative data, not cached rates."""
    expected = {episode.episode_id: episode for episode in episodes}
    if (not expected or len(expected) != len(episodes) or len(records) != len(expected)
            or {row["episode_id"] for row in records} != set(expected)):
        raise ValueError("gate requires exactly the expected distinct nonempty histories")
    expected_method = "full_context" if reader_gate else records[0]["method"]
    if any(row["method"] != expected_method for row in records) or (not reader_gate and expected_method not in NEURAL_METHODS):
        raise ValueError("gate results must come from one appropriate method")
    scored = [score_update_episode(expected[row["episode_id"]],
              [UpdatePrediction(item["case_id"], item["prediction"]) for item in row["predictions"]])
              for row in records]
    rates = {}
    for stage in STAGES if reader_gate else ("before",):
        for category in ("known_accuracy", "absent_accuracy"):
            key = f"{stage}.{category}"
            parts = [result.rates[key] for result in scored]
            rates[key] = CountRate(sum(part.numerator for part in parts), sum(part.denominator for part in parts))
    passed = all(rate.denominator > 0 and rate.numerator * 100 >= 95 * rate.denominator for rate in rates.values())
    return {"threshold": 0.95, "passed": passed, "histories": len(records),
            "rates": {key: rate.to_dict() for key, rate in rates.items()},
            "interpretation": ("qualified_text_reader" if reader_gate else "competent_before_state") if passed
                              else ("reader_not_qualified" if reader_gate else "competence_limited")}
