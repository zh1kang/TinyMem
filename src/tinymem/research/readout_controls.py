"""Detached readout interventions and a fixed whole-history donor mapping."""

from collections.abc import Sequence
from typing import Literal

import torch

from tinymem.memory.readout_interface import check_readout_state
from tinymem.memory.recurrent_slots import LatentSlotState


StateControl = Literal["normal", "zero", "no_memory"]


def controlled_state(state: LatentSlotState, control: StateControl) -> LatentSlotState:
    """Own a detached payload; zeroing values does not remove occupied slots."""
    check_readout_state(state)
    if control not in ("normal", "zero", "no_memory"):
        raise ValueError("unknown state control")
    values = state.values.detach().clone() if control == "normal" else torch.zeros_like(state.values)
    valid = torch.zeros_like(state.valid) if control == "no_memory" else state.valid.clone()
    result = LatentSlotState(values, valid)
    check_readout_state(result)
    return result


def state_donors(history_ids: Sequence[str]) -> dict[str, str]:
    """Cycle sorted identities without consulting histories, answers, or seeds."""
    if (not isinstance(history_ids, Sequence) or isinstance(history_ids, (str, bytes))
            or len(history_ids) < 2
            or any(not isinstance(identity, str) or not identity.strip() for identity in history_ids)):
        raise ValueError("at least two nonempty history identities are required")
    if len(set(history_ids)) != len(history_ids):
        raise ValueError("history identities must be unique")
    ordered = sorted(history_ids)
    return dict(zip(ordered, ordered[1:] + ordered[:1], strict=True))
