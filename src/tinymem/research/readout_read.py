"""Inference from a detached state, shared parameters, and one question only."""

import torch

from tinymem.memory.readout_interface import ReadoutBridge, check_readout_state
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.prefix_reader import generate_prefix_answer
from tinymem.research.pretrained import PretrainedReader


@torch.inference_mode()
def read_state_answer(
    reader: PretrainedReader, bridge: ReadoutBridge, state: LatentSlotState,
    before_ids: torch.Tensor, question_ids: torch.Tensor, *, max_new_tokens: int = 8,
) -> dict[str, object]:
    """Never accept history features, history tokens, gold answers, or other queries."""
    check_readout_state(state)
    if state.values.requires_grad or state.values.grad_fn is not None:
        raise ValueError("inference state must be detached from the training graph")
    if any(module.training for module in reader.model.modules()):
        raise ValueError("reader must be in evaluation mode")
    if any(p.requires_grad or p.grad is not None for p in reader.model.parameters()):
        raise ValueError("reader must be frozen with no parameter gradients")
    return generate_prefix_answer(
        reader, before_ids, bridge(state), question_ids, max_new_tokens=max_new_tokens,
    )
