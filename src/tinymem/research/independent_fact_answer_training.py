"""Writer-only recurrent training from frozen answer readout losses."""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from tinymem.memory.query_pool_slots import QueryPoolSlotWriter
from tinymem.memory.readout_interface import ReadoutBridge
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.independent_fact_recurrent_training import (
    RecurrentBatch,
    _check_state,
    _validate_batch,
)
from tinymem.research.independent_fact_data import FactPrompt
from tinymem.research.pretrained import PretrainedReader
from tinymem.research.prefix_reader import prefix_answer_losses


def _validate_rows(rows: Sequence[FactPrompt]) -> dict[int, FactPrompt]:
    if not isinstance(rows, Sequence) or len(rows) != 16:
        raise ValueError("answer rows must contain all sixteen worlds")
    by_code: dict[int, FactPrompt] = {}
    for row in rows:
        code = getattr(row, "code", None)
        row_queries = getattr(row, "queries", None)
        if type(code) is not int or not 0 <= code < 16 or code in by_code:
            raise ValueError("answer rows must contain each world code exactly once")
        if not isinstance(row_queries, Sequence) or len(row_queries) != 6:
            raise ValueError("each answer row must contain six queries")
        by_code[code] = row
    if set(by_code) != set(range(16)):
        raise ValueError("answer rows must cover world codes zero through fifteen")
    return by_code


def _validate_modules(
    reader: PretrainedReader,
    bridge: ReadoutBridge,
    writer: QueryPoolSlotWriter,
    optimizer: torch.optim.Optimizer,
) -> tuple[tuple[nn.Parameter, ...], torch.device]:
    if any(module.training for module in reader.model.modules()):
        raise ValueError("reader must be in evaluation mode")
    reader_parameters = tuple(reader.model.parameters())
    if any(parameter.requires_grad or parameter.grad is not None for parameter in reader_parameters):
        raise ValueError("reader must be frozen")
    bridge_parameters = tuple(bridge.parameters())
    if any(parameter.requires_grad or parameter.grad is not None for parameter in bridge_parameters):
        raise ValueError("bridge must be frozen")
    reader_device = reader.model.device
    if any(parameter.device != reader_device for parameter in bridge_parameters):
        raise ValueError("reader and bridge must share a device")
    writer_parameters = tuple(writer.parameters())
    if not writer_parameters or any(
        parameter.device.type != "cpu"
        or parameter.dtype != torch.float32
        or not parameter.requires_grad
        for parameter in writer_parameters
    ):
        raise ValueError("writer must be trainable CPU FP32")
    optimized = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    owned = {id(parameter) for parameter in writer_parameters}
    if (
        len(owned) != len(writer_parameters)
        or len(optimized) != len(owned)
        or {id(parameter) for parameter in optimized} != owned
    ):
        raise ValueError("optimizer must own exactly all writer parameters")
    return writer_parameters, reader_device


def _answer_state_gradient(
    reader: PretrainedReader,
    bridge: ReadoutBridge,
    values: torch.Tensor,
    valid: torch.Tensor,
    index: int,
    row: FactPrompt,
    device: torch.device,
    query_microbatch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    single_values = values[index:index + 1].detach().to(device).clone().requires_grad_(True)
    single_valid = valid[index:index + 1].detach().to(device).clone()
    before = torch.tensor(row.before_ids, dtype=torch.long, device=device)
    loss_values = []
    single_gradient = torch.zeros_like(single_values)
    for start in range(0, len(row.queries), query_microbatch):
        # Rebuild the small bridge graph so each reader microbatch can be released.
        memory = bridge(LatentSlotState(single_values, single_valid))
        examples = [
            (before, memory,
             torch.tensor(query.after_ids, dtype=torch.long, device=device),
             torch.tensor(query.answer_ids, dtype=torch.long, device=device))
            for query in row.queries[start:start + query_microbatch]
        ]
        losses = prefix_answer_losses(reader, examples)
        chunk_loss = losses.sum() / len(row.queries)
        if not torch.isfinite(chunk_loss):
            raise ValueError("nonfinite answer loss")
        chunk_gradient = torch.autograd.grad(chunk_loss, single_values)[0]
        single_gradient.add_(chunk_gradient)
        loss_values.append(chunk_loss.detach())
    if not torch.isfinite(single_gradient).all():
        raise ValueError("nonfinite answer-state gradient")
    gradient = torch.zeros_like(values)
    gradient[index:index + 1] = single_gradient.to(values.device)
    return torch.stack(loss_values).sum(), gradient


def train_answer_recurrent_batch(
    writer: QueryPoolSlotWriter,
    batch: RecurrentBatch,
    optimizer: torch.optim.Optimizer,
    *,
    reader: PretrainedReader,
    bridge: ReadoutBridge,
    worlds: Sequence[FactPrompt],
    coordinate_weight: int | float,
    query_microbatch: int = 6,
) -> dict[str, float | int | bool | None]:
    """Train a CPU writer through frozen reader answer losses and four writes."""
    if coordinate_weight not in (0, 1):
        raise ValueError("coordinate_weight must be zero or one")
    if type(query_microbatch) is not int or not 1 <= query_microbatch <= 6:
        raise ValueError("query_microbatch must be an integer from one through six")
    if not hasattr(writer, "reader_width") or not hasattr(writer, "memory_width"):
        raise TypeError("writer must expose the QueryPoolSlotWriter interface")
    _validate_batch(batch, writer)
    rows_by_code = _validate_rows(worlds)
    parameters, reader_device = _validate_modules(reader, bridge, writer, optimizer)
    batch_size = batch.events.shape[0]
    snapshots = tuple(tensor.clone() for tensor in batch.__dict__.values() if isinstance(tensor, torch.Tensor))

    optimizer.zero_grad(set_to_none=True)
    initial = writer(writer.empty(16), batch.histories, batch.history_valid)
    _check_state(initial, 16, writer)
    state = LatentSlotState(
        initial.values.index_select(0, batch.initial_codes),
        initial.valid.index_select(0, batch.initial_codes),
    )
    updates: list[LatentSlotState] = []
    for step in range(4):
        state = writer(state, batch.events[:, step], batch.event_valid[:, step])
        _check_state(state, batch_size, writer)
        updates.append(state)

    initial_gradient = torch.zeros_like(initial.values)
    initial_losses: list[float] = []
    for code in range(16):
        loss, gradient = _answer_state_gradient(
            reader, bridge, initial.values, initial.valid, code, rows_by_code[code], reader_device,
            query_microbatch,
        )
        initial_losses.append(float(loss))
        initial_gradient.add_(gradient, alpha=0.5 / 16)

    update_gradients = [torch.zeros_like(state.values) for state in updates]
    update_losses: list[float] = []
    for step, update in enumerate(updates):
        for row in range(batch_size):
            code = int(batch.after_codes[row, step])
            loss, gradient = _answer_state_gradient(
                reader, bridge, update.values, update.valid, row, rows_by_code[code], reader_device,
                query_microbatch,
            )
            update_losses.append(float(loss))
            update_gradients[step].add_(gradient, alpha=0.5 / (batch_size * 4))

    answer_gradient_norm = torch.stack([
        initial_gradient.square().sum(), *(g.square().sum() for g in update_gradients),
    ]).sum().sqrt()
    initial_ce = sum(initial_losses) / len(initial_losses)
    update_ce = sum(update_losses) / len(update_losses)
    initial_mse = initial.values.new_zeros(())
    update_mse = initial.values.new_zeros(())
    if coordinate_weight:
        initial_mse = F.mse_loss(initial.values, batch.initial_targets)
        mse_initial_gradient = torch.autograd.grad(
            initial_mse, initial.values, retain_graph=True, allow_unused=False,
        )[0]
        initial_gradient.add_(mse_initial_gradient, alpha=0.5)
        update_mse_values = []
        for step, (update, accumulated) in enumerate(zip(updates, update_gradients, strict=True)):
            step_mse = F.mse_loss(update.values, batch.after_targets[:, step])
            update_mse_values.append(step_mse)
            step_gradient = torch.autograd.grad(
                step_mse, update.values, retain_graph=True, allow_unused=False,
            )[0]
            accumulated.add_(step_gradient, alpha=0.5 / 4)
        update_mse = torch.stack(update_mse_values).mean()

    objective_value = 0.5 * initial_ce + 0.5 * update_ce + float(coordinate_weight) * (
        0.5 * float(initial_mse.detach()) + 0.5 * float(update_mse.detach())
    )
    objective = initial.values.new_tensor(objective_value)
    if not torch.isfinite(objective):
        raise ValueError("nonfinite training objective")
    torch.autograd.backward(
        (initial.values, *(state.values for state in updates)),
        (initial_gradient, *update_gradients),
    )
    if any(parameter.grad is None for parameter in parameters):
        raise ValueError("all writer parameters must receive gradients")
    if any(not torch.isfinite(parameter.grad).all() for parameter in parameters):
        raise ValueError("nonfinite writer gradients")
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    if not torch.isfinite(norm):
        raise ValueError("nonfinite writer gradient norm")
    optimizer.step()
    if any(not torch.isfinite(parameter).all() for parameter in parameters):
        raise ValueError("optimizer produced nonfinite writer parameters")
    if any(not torch.equal(actual, snapshot) for actual, snapshot in zip(
        (tensor for tensor in batch.__dict__.values() if isinstance(tensor, torch.Tensor)),
        snapshots,
        strict=True,
    )):
        raise ValueError("recurrent answer batch was mutated")
    if any(parameter.grad is not None for parameter in reader.model.parameters()) or any(
        parameter.grad is not None for parameter in bridge.parameters()
    ):
        raise ValueError("frozen reader or bridge received gradients")
    update_supervised_tokens = sum(
        len(query.answer_ids)
        for row in range(batch_size)
        for step in range(4)
        for query in rows_by_code[int(batch.after_codes[row, step])].queries
    )
    return {
        "initial_answer_ce": initial_ce,
        "updated_answer_ce": update_ce,
        "answer_ce": 0.5 * initial_ce + 0.5 * update_ce,
        "initial_coordinate_mse": float(initial_mse.detach()) if coordinate_weight else None,
        "updated_coordinate_mse": float(update_mse.detach()) if coordinate_weight else None,
        "coordinate_mse": (0.5 * float(initial_mse.detach()) + 0.5 * float(update_mse.detach())) if coordinate_weight else None,
        "answer_state_gradient_norm": float(answer_gradient_norm),
        "objective": objective_value,
        "gradient_norm": float(norm),
        "gradient_clipped": bool(float(norm) > 1.0),
        "coordinate_weight": coordinate_weight,
        "history_examples": 16,
        "event_examples": batch_size * 4,
        "writer_examples": 16 + batch_size * 4,
        "answer_sequences": (16 + batch_size * 4) * 6,
        "supervised_tokens": sum(len(query.answer_ids) for row in worlds for query in row.queries)
        + update_supervised_tokens,
    }
