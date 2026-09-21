"""Read-side fitting on official QA1 questions and query-blind correct states."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from tinymem.data.cases import ReaderCase
from tinymem.data.schema import ReasoningExample
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.studies.qa1.data import Vocabulary, replay_locations, story_id, validate_example
from tinymem.studies.qa1.state import oracle_locations, oracle_state
from tinymem.reader.memory_prompt import NativeMemoryExample, encode_memory_example
from tinymem.reader.prefix import prefix_answer_losses


@dataclass(frozen=True)
class EncodedQuestion:
    source: ReasoningExample
    native: NativeMemoryExample
    state: LatentSlotState


def encode_question(reader, source: ReasoningExample, vocabulary: Vocabulary) -> EncodedQuestion:
    validate_example(source, vocabulary)
    history = source.context.splitlines()
    state = oracle_state(history, vocabulary)
    if oracle_locations(state, vocabulary) != replay_locations(history, vocabulary):
        raise ValueError('matrix state disagrees with independent replay')
    case = ReaderCase(source.source_example_id, 'babi_qa1', story_id(source),
                      source.context, source.question, source.answer)
    return EncodedQuestion(source, encode_memory_example(reader, case), state)


def train_batch(reader, bridge, examples: Sequence[EncodedQuestion], optimizer, *, adapters: tuple,
                vocabulary: Vocabulary, clip_norm: float) -> dict:
    if not examples or any(e.source.split != 'train' for e in examples):
        raise ValueError('optimization accepts training questions only')
    if any(m.training for m in reader.model.modules()):
        raise ValueError('reader must remain in evaluation mode')
    owned = {id(p) for p in adapters}
    parameters = (*tuple(bridge.parameters()), *adapters)
    optimized = [p for group in optimizer.param_groups for p in group['params']]
    if (owned != {id(p) for p in reader.model.parameters() if p.requires_grad}
            or len(owned) != len(adapters)
            or len({id(p) for p in parameters}) != len(parameters)
            or any(not p.requires_grad for p in parameters)
            or len(optimized) != len(parameters)
            or {id(p) for p in optimized} != {id(p) for p in parameters}):
        raise ValueError('optimizer must own exactly the bridge and declared reader adapters')
    for example in examples:
        oracle_locations(example.state, vocabulary)
    optimizer.zero_grad(set_to_none=True)
    reads = []
    device = reader.model.device
    for example in examples:
        native = example.native
        state = LatentSlotState(example.state.values.to(device), example.state.valid.to(device))
        reads.append((torch.tensor(native.before_ids, device=device), bridge(state)[0],
                      torch.tensor(native.after_ids, device=device),
                      torch.tensor(native.answer_ids, device=device)))
    loss = prefix_answer_losses(reader, reads).mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError('nonfinite answer loss')
    loss.backward()
    if any(p.grad is not None for p in reader.model.parameters() if id(p) not in owned):
        raise ValueError('frozen reader received gradients')
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
        raise ValueError('missing or nonfinite training gradient')
    norm = torch.nn.utils.clip_grad_norm_(parameters, clip_norm, error_if_nonfinite=True)
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in parameters):
        raise ValueError('optimizer produced nonfinite parameters')
    return {'answer_ce': float(loss.detach()), 'gradient_norm': float(norm),
            'questions': len(examples), 'persistent_bytes': 258}
