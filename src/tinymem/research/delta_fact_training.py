"""Train phase-two writers and compatible read interfaces on frozen features."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from tinymem.memory.delta_slots import DeltaSlotWriter
from tinymem.memory.query_pool_slots import QueryPoolSlotWriter
from tinymem.research.delta_fact_readout import SlotReadout
from tinymem.research.prefix_reader import prefix_answer_losses
from tinymem.research.pretrained import PretrainedReader
from tinymem.research.adapted_readout import ReadoutQuery


Writer = QueryPoolSlotWriter | DeltaSlotWriter


@dataclass(frozen=True)
class Endpoint:
    after_write: int
    queries: tuple[ReadoutQuery, ...]


@dataclass(frozen=True)
class TrainingExample:
    episode_id: str
    before_ids: tuple[int, ...]
    features: tuple[torch.Tensor, ...]
    endpoints: tuple[Endpoint, ...]
    split: str = "train"


def _validate_example(example: TrainingExample, reader_width: int) -> None:
    if example.split != "train":
        raise ValueError("optimization accepts training examples only")
    if not example.episode_id or not example.features or not example.before_ids or not example.endpoints:
        raise ValueError("training examples require an identity, features, prompt, and endpoints")
    for hidden in example.features:
        if (hidden.ndim != 2 or hidden.shape[0] == 0 or hidden.shape[1] != reader_width
                or hidden.dtype != torch.float32 or hidden.device.type != "cpu"
                or hidden.requires_grad or hidden.grad_fn is not None):
            raise ValueError("features must be detached CPU FP32 token matrices of the reader width")
        if not torch.isfinite(hidden).all():
            raise ValueError("features must be finite")
    endpoints = [e.after_write for e in example.endpoints]
    if (any(type(i) is not int or not 1 <= i <= len(example.features) for i in endpoints)
            or endpoints != sorted(set(endpoints)) or any(not e.queries for e in example.endpoints)):
        raise ValueError("endpoints must be distinct ordered write positions with queries")


def train_batch(
    reader: PretrainedReader, writer: Writer, bridge: SlotReadout,
    examples: Sequence[TrainingExample], optimizer: torch.optim.Optimizer, *,
    adapter_parameters: tuple[nn.Parameter, ...] = (),
) -> dict[str, float | int]:
    """Average answer CE equally over episodes; keep every write attached."""
    if not examples:
        raise ValueError("a training batch must contain examples")
    if writer.slots != 2 or writer.memory_width != bridge.memory_width:
        raise ValueError("writer and bridge state shapes differ")
    if writer.reader_width != bridge.reader_width:
        raise ValueError("writer and bridge reader widths differ")
    if any(module.training for module in reader.model.modules()):
        raise ValueError("reader must remain in evaluation mode during adapter training")
    reader_owned = {id(p) for p in adapter_parameters}
    if len(reader_owned) != len(adapter_parameters) or reader_owned != {
        id(p) for p in reader.model.parameters() if p.requires_grad
    }:
        raise ValueError("reader adapter ownership differs from the declared parameters")
    parameters = [*writer.parameters(), *bridge.parameters(), *adapter_parameters]
    optimized = [p for group in optimizer.param_groups for p in group["params"]]
    if (len({id(p) for p in parameters}) != len(parameters)
            or len(optimized) != len(parameters)
            or {id(p) for p in parameters} != {id(p) for p in optimized}
            or any(not p.requires_grad for p in parameters)):
        raise ValueError("optimizer must own exactly the writer, bridge, and declared adapter")
    if any(p.device != reader.model.device for p in parameters):
        raise ValueError("all training modules must share the reader device")
    frozen_parameters = [p for p in reader.model.parameters() if id(p) not in reader_owned]
    if any(p.grad is not None for p in frozen_parameters):
        raise ValueError("frozen reader has unexpected gradients")
    for example in examples:
        _validate_example(example, writer.reader_width)
    optimizer.zero_grad(set_to_none=True)
    device = reader.model.device
    reads = []
    weights = []
    for example in examples:
        state = writer.empty(1)
        endpoints = {e.after_write: e for e in example.endpoints}
        count = sum(len(e.queries) for e in example.endpoints)
        before = torch.tensor(example.before_ids, device=device)
        for index, hidden in enumerate(example.features, start=1):
            features = hidden.to(device).unsqueeze(0)
            state = writer(state, features, torch.ones(features.shape[:2], device=device, dtype=torch.bool))
            if index not in endpoints:
                continue
            if not bool(state.valid.all()):
                raise ValueError("nonempty training writes must initialize both slots")
            # Both slots are valid here; Boolean indexing adds an unnecessary
            # scatter backward that MPS cannot execute with strict determinism.
            memory = bridge(state)[0]
            for query in endpoints[index].queries:
                reads.append((before, memory, torch.tensor(query.after_ids, device=device),
                              torch.tensor(query.answer_ids, device=device)))
                weights.append(1 / (len(examples) * count))
    losses = prefix_answer_losses(reader, reads)
    loss = (losses * losses.new_tensor(weights)).sum()
    if not torch.isfinite(loss):
        raise ValueError("nonfinite answer loss")
    loss.backward()
    if any(p.grad is None for p in parameters):
        raise ValueError("a declared training parameter did not receive a gradient")
    if any(p.grad is not None for p in frozen_parameters):
        raise ValueError("frozen reader gradient ownership violation")
    gradient_norms = {}
    for name, group in (("writer", tuple(writer.parameters())),
                        ("bridge", tuple(bridge.parameters())), ("adapter", adapter_parameters)):
        norm = torch.stack([p.grad.float().square().sum() for p in group]).sum().sqrt() if group else loss.new_zeros(())
        gradient_norms[name + "_gradient_norm"] = float(norm)
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in parameters):
        raise ValueError("optimizer produced nonfinite parameters")
    return {"answer_ce": float(loss.detach()), "gradient_norm": float(norm),
            "episodes": len(examples), "answer_sequences": len(reads),
            "write_calls": sum(len(e.features) for e in examples),
            "persistent_bytes": bridge.state_bytes, **gradient_norms}
