"""Discrete memory compression through a learned codebook."""

from dataclasses import dataclass
import math
from numbers import Integral, Real

import torch
from torch import nn

from tinymem.memory.codebook import GumbelSoftmaxCodebook
from tinymem.memory.continuous import (
    ContinuousMemoryCompressor,
    MultiSlotAttentionMemoryCompressor,
)


@dataclass(frozen=True)
class DiscreteCompressionOutput:
    """Hold quantized slots and the information needed to inspect them."""

    values: torch.Tensor
    valid: torch.Tensor
    indices: torch.Tensor
    assignments: torch.Tensor
    probabilities: torch.Tensor
    logits: torch.Tensor
    prequantized: torch.Tensor


class DiscreteMemoryCompressor(ContinuousMemoryCompressor):
    """Summarize expired states and quantize each summary to one code."""

    def __init__(
        self,
        model_width: int,
        *,
        codebook_size: int,
        summary_slots: int,
        temperature: float = 1.0,
        evaluation_mode: str = "hard",
    ) -> None:
        super().__init__(model_width)
        for name, value in (
            ("codebook_size", codebook_size),
            ("summary_slots", summary_slots),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        self.codebook_size = int(codebook_size)
        self.summary_slots = int(summary_slots)
        self.summarizer = MultiSlotAttentionMemoryCompressor(
            self.model_width,
            summary_slots=self.summary_slots,
        )
        self.logit_projection = nn.Linear(
            self.model_width,
            self.codebook_size,
        )
        self.codebook = GumbelSoftmaxCodebook(
            self.model_width,
            self.codebook_size,
            evaluation_mode=evaluation_mode,
        )
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
        """Return the current sampling temperature."""
        return float(self._temperature.item())

    def set_temperature(self, temperature: float) -> None:
        """Update the non-learned temperature saved in the state dictionary."""
        value = self._validate_temperature(temperature)
        self._temperature.fill_(value)

    def set_evaluation_mode(self, mode: str) -> None:
        """Select the code representation used only during evaluation."""
        self.codebook.set_evaluation_mode(mode)

    def compress(
        self,
        expired_hidden: torch.Tensor,
        expired_valid: torch.Tensor,
    ) -> DiscreteCompressionOutput:
        """Return quantized slots together with their code assignments."""
        prequantized, summary_valid = self.summarizer(
            expired_hidden,
            expired_valid,
        )
        logits = self.logit_projection(prequantized)
        quantized = self.codebook(
            logits,
            summary_valid,
            temperature=self.temperature,
        )
        return DiscreteCompressionOutput(
            values=quantized.values,
            valid=summary_valid,
            indices=quantized.indices,
            assignments=quantized.assignments,
            probabilities=quantized.probabilities,
            logits=logits,
            prequantized=prequantized,
        )

    def forward(
        self,
        expired_hidden: torch.Tensor,
        expired_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return discrete vectors through the shared compressor interface."""
        output = self.compress(expired_hidden, expired_valid)
        return output.values, output.valid
