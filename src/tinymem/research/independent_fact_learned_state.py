"""Train updates toward fixed states learned by an answer-trained initializer."""

import torch
from torch.nn import functional as F

from tinymem.memory.query_pool_slots import QueryPoolSlotWriter
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.independent_fact_recurrent_training import (
    RecurrentBatch, _check_state, _validate_batch, _writer_parameters,
)


def validate_learned_targets(targets: LatentSlotState) -> None:
    """Require detached CPU states for every fixed world, without assigned coordinates."""
    if (targets.values.shape != (16, 2, 8) or targets.valid.shape != (16, 2)
            or targets.values.dtype != torch.float32 or targets.valid.dtype != torch.bool
            or targets.values.device.type != 'cpu' or targets.valid.device.type != 'cpu'
            or targets.values.requires_grad or targets.values.grad_fn is not None
            or not targets.valid.all() or not torch.isfinite(targets.values).all()
            or (targets.values.abs() > 1).any()):
        raise ValueError('learned targets must be sixteen detached finite CPU 66-byte states')


def train_learned_state_batch(
    writer: QueryPoolSlotWriter,
    batch: RecurrentBatch,
    optimizer: torch.optim.Optimizer,
    *,
    targets: LatentSlotState,
    arm: str,
) -> dict[str, float | int | bool]:
    """Match the fixed initializer's after-world states using reset or raw recurrence."""
    if arm not in ('reset', 'recurrent'):
        raise ValueError('arm must be reset or recurrent')
    if writer.queries.device.type != 'cpu':
        raise ValueError('updater must run on CPU')
    _validate_batch(batch, writer)
    validate_learned_targets(targets)
    parameters = _writer_parameters(writer, optimizer)
    snapshots = {key: value.clone() for key, value in batch.__dict__.items() if isinstance(value, torch.Tensor)}
    target_values = targets.values.clone(); target_valid = targets.valid.clone()
    optimizer.zero_grad(set_to_none=True)
    state = LatentSlotState(targets.values.index_select(0, batch.initial_codes),
                            targets.valid.index_select(0, batch.initial_codes))
    updates = []
    for step in range(4):
        if arm == 'reset':
            state = LatentSlotState(targets.values.index_select(0, batch.before_codes[:, step]),
                                    targets.valid.index_select(0, batch.before_codes[:, step]))
        state = writer(state, batch.events[:, step], batch.event_valid[:, step])
        _check_state(state, batch.events.shape[0], writer)
        updates.append(state.values)
    expected = targets.values[batch.after_codes]
    predicted = torch.stack(updates, dim=1)
    loss = F.mse_loss(predicted, expected)
    if not torch.isfinite(loss):
        raise ValueError('nonfinite learned-state loss')
    loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
        raise ValueError('all updater parameters must receive finite gradients')
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in parameters):
        raise ValueError('updater parameters became nonfinite')
    if (not torch.equal(targets.values, target_values) or not torch.equal(targets.valid, target_valid)
            or any(not torch.equal(getattr(batch, key), value) for key, value in snapshots.items())):
        raise ValueError('learned targets or training inputs changed')
    return {
        'learned_state_mse': float(loss.detach()), 'gradient_norm': float(norm),
        'gradient_clipped': bool(float(norm) > 1), 'event_examples': batch.events.shape[0] * 4,
        'initial_encoder_examples': 0, 'target_values': expected.numel(),
    }
