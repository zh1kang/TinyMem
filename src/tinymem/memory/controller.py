"""Adaptive categorical write decisions for recurrent memory."""

from dataclasses import dataclass
import math
from numbers import Integral, Real

import torch
from torch import nn
from torch.nn import functional as F


KEEP_ACTION = 0
WRITE_ACTION = 1
CONTROLLER_ACTIONS = ("keep", "write")


@dataclass(frozen=True)
class WriteControllerOutput:
    """Hold categorical decisions and their differentiable representation."""

    logits: torch.Tensor
    probabilities: torch.Tensor
    assignments: torch.Tensor
    actions: torch.Tensor
    write_strength: torch.Tensor


class AdaptiveWriteController(nn.Module):
    """Choose whether to write from segment, memory, and surprise features."""

    def __init__(
        self,
        model_width: int,
        *,
        hidden_width: int | None = None,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if isinstance(model_width, bool) or not isinstance(model_width, Integral):
            raise TypeError("model_width must be an integer")
        if model_width <= 0:
            raise ValueError("model_width must be positive")
        if hidden_width is None:
            hidden_width = int(model_width)
        if isinstance(hidden_width, bool) or not isinstance(hidden_width, Integral):
            raise TypeError("hidden_width must be an integer or None")
        if hidden_width <= 0:
            raise ValueError("hidden_width must be positive")

        self.model_width = int(model_width)
        self.hidden_width = int(hidden_width)
        self.input_projection = nn.Linear(
            2 * self.model_width + 1,
            self.hidden_width,
        )
        self.action_projection = nn.Linear(
            self.hidden_width,
            len(CONTROLLER_ACTIONS),
        )
        nn.init.zeros_(self.action_projection.weight)
        with torch.no_grad():
            self.action_projection.bias.copy_(torch.tensor([0.0, 1.0]))
        self.register_buffer(
            "_temperature",
            torch.tensor(self._validate_temperature(temperature)),
        )

    @staticmethod
    def _validate_temperature(temperature: float) -> float:
        if isinstance(temperature, bool) or not isinstance(temperature, Real):
            raise TypeError("temperature must be a real number")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        return float(temperature)

    @property
    def temperature(self) -> float:
        """Return the current Gumbel-Softmax temperature."""
        return float(self._temperature.item())

    def set_temperature(self, temperature: float) -> None:
        """Update the non-learned temperature stored in checkpoints."""
        self._temperature.fill_(self._validate_temperature(temperature))

    def _validate_inputs(
        self,
        segment_state: torch.Tensor,
        memory_state: torch.Tensor,
        prediction_surprise: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        for name, tensor in (
            ("segment_state", segment_state),
            ("memory_state", memory_state),
            ("prediction_surprise", prediction_surprise),
            ("valid", valid),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
        expected_state_shape = (segment_state.shape[0], self.model_width)
        if segment_state.ndim != 2 or segment_state.shape != expected_state_shape:
            raise ValueError(
                f"segment_state must have shape [batch, {self.model_width}]"
            )
        if memory_state.shape != expected_state_shape:
            raise ValueError(f"memory_state must have shape {expected_state_shape}")
        if prediction_surprise.shape != (segment_state.shape[0], 1):
            raise ValueError(
                "prediction_surprise must have shape [batch, 1]"
            )
        if valid.shape != (segment_state.shape[0],):
            raise ValueError("valid must have shape [batch]")
        if segment_state.shape[0] == 0:
            raise ValueError("controller inputs must contain a batch row")
        if not all(
            tensor.is_floating_point()
            for tensor in (segment_state, memory_state, prediction_surprise)
        ):
            raise TypeError("controller feature tensors must be floating point")
        if valid.dtype != torch.bool:
            raise TypeError("valid must be a boolean tensor")
        tensors = (segment_state, memory_state, prediction_surprise, valid)
        if any(tensor.device != segment_state.device for tensor in tensors):
            raise ValueError("controller inputs must share a device")
        if any(
            tensor.dtype != segment_state.dtype
            for tensor in (memory_state, prediction_surprise)
        ):
            raise TypeError("controller feature tensors must share a dtype")
        if segment_state.device != self.input_projection.weight.device:
            raise ValueError("controller inputs and parameters must share a device")
        if segment_state.dtype != self.input_projection.weight.dtype:
            raise TypeError("controller inputs and parameters must share a dtype")

    def forward(
        self,
        segment_state: torch.Tensor,
        memory_state: torch.Tensor,
        prediction_surprise: torch.Tensor,
        valid: torch.Tensor,
    ) -> WriteControllerOutput:
        """Return hard keep/write actions with relaxed training gradients."""
        self._validate_inputs(
            segment_state,
            memory_state,
            prediction_surprise,
            valid,
        )
        features = torch.cat(
            (segment_state, memory_state, prediction_surprise),
            dim=-1,
        )
        hidden = F.silu(self.input_projection(features))
        logits = self.action_projection(hidden)
        probabilities = torch.softmax(logits / self.temperature, dim=-1)

        if self.training:
            dtype_info = torch.finfo(logits.dtype)
            uniform = torch.rand_like(logits).clamp(
                min=dtype_info.tiny,
                max=1.0 - dtype_info.eps,
            )
            gumbel = -torch.log(-torch.log(uniform))
            relaxed = torch.softmax(
                (logits + gumbel) / self.temperature,
                dim=-1,
            )
            actions = relaxed.argmax(dim=-1)
        else:
            relaxed = probabilities
            actions = logits.argmax(dim=-1)

        hard = F.one_hot(
            actions,
            num_classes=len(CONTROLLER_ACTIONS),
        ).to(dtype=logits.dtype)
        assignments = (
            hard - relaxed.detach() + relaxed
            if self.training
            else hard
        )
        keep = torch.zeros_like(assignments)
        keep[:, KEEP_ACTION] = 1
        assignments = torch.where(valid.unsqueeze(-1), assignments, keep)
        probabilities = torch.where(valid.unsqueeze(-1), probabilities, keep)
        actions = torch.where(
            valid,
            actions,
            torch.full_like(actions, KEEP_ACTION),
        )
        return WriteControllerOutput(
            logits=logits,
            probabilities=probabilities,
            assignments=assignments,
            actions=actions,
            write_strength=assignments[:, WRITE_ACTION : WRITE_ACTION + 1],
        )
