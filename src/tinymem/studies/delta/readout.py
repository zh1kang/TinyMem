"""Width-specific phase-two reads without the phase-one bounded-state contract."""

import torch
from torch import nn

from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.reader.prefix import generate_prefix_answer
from tinymem.reader.pretrained import PretrainedReader


class SlotReadout(nn.Module):
    """Map two stored slots to reader embeddings with a shared affine bridge."""

    def __init__(self, *, memory_width: int, reader_width: int) -> None:
        super().__init__()
        for name, value in (("memory_width", memory_width), ("reader_width", reader_width)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.memory_width = memory_width
        self.reader_width = reader_width
        self.input_projection = nn.Linear(memory_width, 32)
        self.output_projection = nn.Linear(32, reader_width)

    @property
    def state_bytes(self) -> int:
        return 2 * self.memory_width * 4 + 2

    def forward(self, state: LatentSlotState) -> torch.Tensor:
        if state.values.shape[1:] != (2, self.memory_width):
            raise ValueError("state shape must match two slots and the declared memory width")
        if state.values.dtype != torch.float32 or any(p.dtype != torch.float32 for p in self.parameters()):
            raise TypeError("state values and bridge parameters must be FP32")
        if state.values.device != self.input_projection.weight.device:
            raise ValueError("state and bridge must share a device")
        values = state.values.masked_fill(~state.valid.unsqueeze(-1), 0)
        if not torch.isfinite(values).all():
            raise ValueError("valid state values must be finite")
        memory = self.output_projection(self.input_projection(values))
        memory = memory.masked_fill(~state.valid.unsqueeze(-1), 0)
        if not torch.isfinite(memory).all():
            raise ValueError("projected memory must be finite")
        return memory


def own_state(state: LatentSlotState, index: int) -> LatentSlotState:
    """Detach and copy one history so a view cannot retain other histories."""
    if type(index) is not int or not 0 <= index < state.values.shape[0]:
        raise ValueError("index must select a history in the state batch")
    return LatentSlotState(
        state.values[index:index + 1].detach().clone().contiguous(),
        state.valid[index:index + 1].detach().clone().contiguous(),
    )


@torch.inference_mode()
def read_answer(
    reader: PretrainedReader, bridge: SlotReadout, state: LatentSlotState,
    before_ids: torch.Tensor, question_ids: torch.Tensor, *, max_new_tokens: int = 8,
) -> dict[str, object]:
    """Accept one owned state and one question, without history or write access."""
    if state.values.shape[0] != 1 or state.nbytes != bridge.state_bytes:
        raise ValueError("inference state must own exactly one history's declared bytes")
    if state.values.requires_grad or state.values.grad_fn is not None:
        raise ValueError("inference state must be detached")
    for name, module in (("reader", reader.model), ("bridge", bridge)):
        if any(part.training for part in module.modules()):
            raise ValueError(f"{name} must be in evaluation mode")
        if any(p.requires_grad or p.grad is not None for p in module.parameters()):
            raise ValueError(f"{name} must be frozen without parameter gradients")
    memory = bridge(state)[0][state.valid[0]]
    return generate_prefix_answer(reader, before_ids, memory, question_ids,
                                  max_new_tokens=max_new_tokens)
