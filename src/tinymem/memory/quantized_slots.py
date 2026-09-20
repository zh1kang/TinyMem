"""Quantized recurrent slot memory with a fixed byte representation."""

from __future__ import annotations

from numbers import Integral

import torch
from torch import nn

_QUANTIZATION_MAX = 127


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _quantize_ste(values: torch.Tensor) -> torch.Tensor:
    clipped = values.clamp(-1.0, 1.0)
    rounded = torch.round(clipped * _QUANTIZATION_MAX) / _QUANTIZATION_MAX
    return rounded.detach() + (clipped - clipped.detach())


class QuantizedSlotMemory(nn.Module):
    """A fixed-size signed-int8 recurrent memory with differentiable writes.

    The recurrent state is always a plain ``[batch, slots, memory_width]``
    FP32 tensor.  Each write is represented on the storage grid
    ``round(clamp(state, -1, 1) * 127) / 127`` in the forward pass, while the
    straight-through estimator supplies gradients through that quantizer.
    """

    def __init__(
        self,
        reader_width: int,
        *,
        slots: int,
        memory_width: int = 32,
        hidden_width: int = 128,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.reader_width = _positive_int("reader_width", reader_width)
        self.slots = _positive_int("slots", slots)
        self.memory_width = _positive_int("memory_width", memory_width)
        self.hidden_width = _positive_int("hidden_width", hidden_width)
        self.heads = _positive_int("heads", heads)
        if self.hidden_width % self.heads:
            raise ValueError("hidden_width must be divisible by heads")

        self.slot_identity = nn.Parameter(torch.empty(self.slots, self.hidden_width))
        nn.init.normal_(self.slot_identity, mean=0.0, std=self.hidden_width**-0.5)
        self.token_projection = nn.Linear(self.reader_width, self.hidden_width)
        self.old_memory_projection = nn.Linear(self.memory_width, self.hidden_width)
        self.token_norm = nn.LayerNorm(self.hidden_width)
        self.cross_attention = nn.MultiheadAttention(
            self.hidden_width, self.heads, batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(self.hidden_width)
        self.self_attention = nn.MultiheadAttention(
            self.hidden_width, self.heads, batch_first=True,
        )
        self.self_norm = nn.LayerNorm(self.hidden_width)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_width, 4 * self.hidden_width),
            nn.GELU(),
            nn.Linear(4 * self.hidden_width, self.hidden_width),
        )
        self.proposal_projection = nn.Linear(self.hidden_width, self.memory_width)
        self.gate_projection = nn.Linear(
            self.memory_width + self.hidden_width, self.memory_width,
        )
        nn.init.zeros_(self.gate_projection.weight)
        nn.init.zeros_(self.gate_projection.bias)
        self.memory_projection = nn.Linear(self.memory_width, self.reader_width)

    @property
    def persistent_bytes(self) -> int:
        """Return the exact serialized payload size for one history."""
        return self.slots * self.memory_width

    def _validate_state(self, state: object) -> torch.Tensor:
        if not isinstance(state, torch.Tensor):
            raise TypeError("state must be a tensor")
        if state.ndim != 3 or state.shape[1:] != (self.slots, self.memory_width):
            raise ValueError(
                f"state must have shape [batch, {self.slots}, {self.memory_width}]"
            )
        parameter = self.slot_identity
        if state.dtype != torch.float32:
            raise TypeError("state must use FP32")
        if state.device != parameter.device:
            raise ValueError("state and memory must share a device")
        if not bool(torch.isfinite(state).all()):
            raise ValueError("state must be finite")
        if bool((state.abs() > 1).any()):
            raise ValueError("state values must lie in [-1, 1]")
        return state

    def _validate_hidden(self, hidden: object, valid: object) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(hidden, torch.Tensor) or not isinstance(valid, torch.Tensor):
            raise TypeError("hidden and valid must be tensors")
        if hidden.ndim != 3 or hidden.shape[0] <= 0 or hidden.shape[1] <= 0:
            raise ValueError("hidden must have shape [batch, nonempty_tokens, reader_width]")
        if hidden.shape[2] != self.reader_width:
            raise ValueError(f"hidden final dimension must be {self.reader_width}")
        if valid.shape != hidden.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("valid must be boolean with shape [batch, tokens]")
        if hidden.dtype != torch.float32:
            raise TypeError("hidden must use FP32")
        if hidden.device != self.slot_identity.device or valid.device != hidden.device:
            raise ValueError("hidden, valid, and memory must share a device")
        if not bool(torch.isfinite(hidden.masked_select(valid.unsqueeze(-1))).all()):
            raise ValueError("valid hidden values must be finite")
        return hidden, valid

    def empty(self, batch_size: int) -> torch.Tensor:
        """Return a zero state with all slots present and no metadata."""
        batch_size = _positive_int("batch_size", batch_size)
        return self.slot_identity.new_zeros(batch_size, self.slots, self.memory_width)

    def _write(self, state: torch.Tensor, hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        safe_hidden = hidden.masked_fill(~valid.unsqueeze(-1), 0.0)
        token_values = self.token_norm(self.token_projection(safe_hidden))
        old = self.old_memory_projection(state) + self.slot_identity.unsqueeze(0)
        queries = self.cross_norm(old)
        active = valid.any(dim=1)
        safe_valid = valid.clone()
        safe_valid[:, 0] |= ~active
        cross, _ = self.cross_attention(
            queries,
            token_values,
            token_values,
            key_padding_mask=~safe_valid,
            need_weights=False,
        )
        hidden_slots = old + cross
        self_input = self.self_norm(hidden_slots)
        self_update, _ = self.self_attention(
            self_input, self_input, self_input, need_weights=False,
        )
        hidden_slots = hidden_slots + self_update
        hidden_slots = hidden_slots + self.ffn(self.self_norm(hidden_slots))
        proposal = torch.tanh(self.proposal_projection(hidden_slots))
        gate = torch.sigmoid(self.gate_projection(torch.cat((state, hidden_slots), dim=-1)))
        proposed = state * (1.0 - gate) + proposal * gate
        quantized = _quantize_ste(proposed)
        return torch.where(active[:, None, None], quantized, state)

    def forward(self, state: torch.Tensor, hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Write one masked feature segment and return the quantized state."""
        state = self._validate_state(state)
        hidden, valid = self._validate_hidden(hidden, valid)
        if state.shape[0] != hidden.shape[0]:
            raise ValueError("state and hidden must have the same batch size")
        return self._write(state, hidden, valid)

    def memory_vectors(self, state: torch.Tensor) -> torch.Tensor:
        """Project stored slots into reader vectors without changing state."""
        state = self._validate_state(state)
        return self.memory_projection(state)

    def pack(self, state: torch.Tensor) -> bytes:
        """Serialize one state using its exact signed-int8 grid."""
        state = self._validate_state(state)
        if state.shape[0] != 1:
            raise ValueError("pack accepts exactly one history")
        scaled = state * _QUANTIZATION_MAX
        rounded = torch.round(scaled)
        if not bool(torch.allclose(scaled, rounded, atol=1e-5, rtol=0.0)):
            raise ValueError("state values must lie on the signed-int8 grid")
        if bool((rounded < -_QUANTIZATION_MAX).any()) or bool((rounded > _QUANTIZATION_MAX).any()):
            raise ValueError("state values are outside the signed-int8 range")
        return rounded.to(torch.int8).cpu().contiguous().numpy().tobytes()

    def unpack(self, payload: bytes | bytearray | memoryview, device: torch.device | str | None = None) -> torch.Tensor:
        """Decode exactly one signed-int8 state from a byte payload."""
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("payload must be bytes-like")
        if len(payload) != self.persistent_bytes:
            raise ValueError(f"payload must contain exactly {self.persistent_bytes} bytes")
        raw = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        signed = raw.view(torch.int8).reshape(1, self.slots, self.memory_width)
        if bool((signed == -128).any()):
            raise ValueError("payload contains reserved signed-int8 value -128")
        target_device = self.slot_identity.device if device is None else device
        result = signed.to(device=target_device, dtype=torch.float32) / _QUANTIZATION_MAX
        if not bool(torch.isfinite(result).all()):
            raise ValueError("decoded state must be finite")
        return result


__all__ = ["QuantizedSlotMemory"]
