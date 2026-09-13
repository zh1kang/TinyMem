"""Updater-only answer training with a fixed learned after-state auxiliary."""
from collections.abc import Sequence
import math

import torch

from tinymem.memory.query_pool_slots import QueryPoolSlotWriter
from tinymem.memory.readout_interface import ReadoutBridge
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.independent_fact_answer_training import (
    _answer_state_gradient, _validate_modules, _validate_rows,
)
from tinymem.research.independent_fact_data import FactPrompt
from tinymem.research.independent_fact_learned_state import validate_learned_targets
from tinymem.research.independent_fact_recurrent_training import RecurrentBatch, _check_state, _validate_batch
from tinymem.research.pretrained import PretrainedReader


def train_joint_update_batch(
    writer: QueryPoolSlotWriter,
    batch: RecurrentBatch,
    optimizer: torch.optim.Optimizer,
    *,
    reader: PretrainedReader,
    bridge: ReadoutBridge,
    worlds: Sequence[FactPrompt],
    targets: LatentSlotState,
    state_weight: float,
    measure_only: bool = False,
) -> dict[str, float | int | bool]:
    """Use full recurrent gradients; measurement leaves weights and optimizer unchanged."""
    if type(state_weight) not in (int, float) or not math.isfinite(state_weight) or state_weight < 0:
        raise ValueError('state_weight must be finite and nonnegative')
    if type(measure_only) is not bool:
        raise TypeError('measure_only must be boolean')
    _validate_batch(batch, writer)
    validate_learned_targets(targets)
    rows = _validate_rows(worlds)
    parameters, device = _validate_modules(reader, bridge, writer, optimizer)
    snapshots = {k: v.clone() for k, v in batch.__dict__.items() if isinstance(v, torch.Tensor)}
    teacher_values, teacher_valid = targets.values.clone(), targets.valid.clone()
    optimizer.zero_grad(set_to_none=True)
    state = LatentSlotState(targets.values[batch.initial_codes], targets.valid[batch.initial_codes])
    states = []
    for step in range(4):
        state = writer(state, batch.events[:, step], batch.event_valid[:, step])
        _check_state(state, batch.events.shape[0], writer)
        states.append(state)
    ce_gradients = []
    losses = []
    count = batch.events.shape[0] * 4
    for step, state in enumerate(states):
        accumulated = torch.zeros_like(state.values)
        for row in range(batch.events.shape[0]):
            code = int(batch.after_codes[row, step])
            loss, gradient = _answer_state_gradient(
                reader, bridge, state.values, state.valid, row, rows[code], device, 6,
            )
            losses.append(float(loss))
            accumulated.add_(gradient, alpha=1 / count)
        ce_gradients.append(accumulated)
    mse = (torch.stack([s.values for s in states], dim=1) - targets.values[batch.after_codes]).square().mean()
    ce_parameters = torch.autograd.grad(
        tuple(s.values for s in states), parameters, grad_outputs=tuple(ce_gradients), retain_graph=True,
    )
    mse_parameters = torch.autograd.grad(mse, parameters)
    if any(not torch.isfinite(g).all() for g in (*ce_parameters, *mse_parameters)):
        raise ValueError('nonfinite updater gradient')
    ce_norm = torch.stack([g.square().sum() for g in ce_parameters]).sum().sqrt()
    mse_norm = torch.stack([g.square().sum() for g in mse_parameters]).sum().sqrt()
    dot = torch.stack([(a*b).sum() for a,b in zip(ce_parameters, mse_parameters, strict=True)]).sum()
    for parameter, ce_gradient, mse_gradient in zip(parameters, ce_parameters, mse_parameters, strict=True):
        parameter.grad = ce_gradient + state_weight * mse_gradient
    ce = sum(losses) / count
    objective = ce + state_weight * float(mse.detach())
    if not math.isfinite(objective):
        raise ValueError('nonfinite joint objective')
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    if not measure_only:
        optimizer.step()
    if any(not torch.isfinite(p).all() for p in parameters):
        raise ValueError('optimizer produced nonfinite parameters')
    if (not torch.equal(targets.values, teacher_values) or not torch.equal(targets.valid, teacher_valid)
            or any(not torch.equal(getattr(batch,k),v) for k,v in snapshots.items())):
        raise ValueError('training input or frozen teacher mutated')
    if any(p.grad is not None for p in (*reader.model.parameters(), *bridge.parameters())):
        raise ValueError('frozen reader or bridge received gradients')
    return {
        'answer_ce': ce, 'learned_state_mse': float(mse.detach()), 'state_weight': float(state_weight),
        'objective': objective, 'answer_parameter_gradient_norm': float(ce_norm),
        'state_parameter_gradient_norm': float(mse_norm), 'parameter_gradient_dot': float(dot),
        'gradient_norm': float(norm), 'gradient_clipped': bool(float(norm)>1),
        'initial_encoder_examples': 0, 'event_examples': count, 'answer_sequences': count*6,
        'target_values': count*16, 'optimizer_steps': int(not measure_only),
    }


def calibrate_state_weight(metrics: Sequence[dict[str, float | int | bool]]) -> float:
    """Set auxiliary RMS parameter-gradient norm to one quarter of answer RMS norm."""
    if len(metrics) != 16 or any(m['optimizer_steps'] != 0 for m in metrics):
        raise ValueError('calibration requires sixteen measurement-only training batches')
    ce = [float(m['answer_parameter_gradient_norm']) for m in metrics]
    mse = [float(m['state_parameter_gradient_norm']) for m in metrics]
    if any(not math.isfinite(v) or v < 0 for v in ce+mse):
        raise ValueError('calibration norms must be finite and nonnegative')
    ce_rms = math.sqrt(sum(v*v for v in ce)/16)
    mse_rms = math.sqrt(sum(v*v for v in mse)/16)
    if ce_rms == 0 or mse_rms == 0:
        raise ValueError('calibration requires nonzero answer and state gradients')
    weight = .25 * ce_rms / mse_rms
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError('calibrated weight must be finite and positive')
    return weight
