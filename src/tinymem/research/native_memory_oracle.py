"""Privileged training-only codes that isolate the native memory readout."""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from tinymem.data.reader_gate import ReaderCase
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.memory_prompt import NativeMemoryExample


QA1_LOCATIONS = ("bathroom", "bedroom", "garden", "hallway", "kitchen", "office")


def select_oracle_cases(
    cases: Sequence[ReaderCase], examples: Sequence[NativeMemoryExample],
) -> list[int]:
    """Select one shortest qa1 history per answer, using training inputs only."""
    if len(cases) != len(examples) or any(case.case_id != example.case_id for case, example in zip(cases, examples, strict=True)):
        raise ValueError("cases and native examples must align")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("case IDs must be unique")
    selected = []
    for answer in QA1_LOCATIONS:
        candidates = [index for index, case in enumerate(cases) if case.category == "babi_qa1" and case.answer == answer]
        if not candidates:
            raise ValueError(f"missing qa1 answer class: {answer}")
        selected.append(min(candidates, key=lambda index: (len(examples[index].history_ids), cases[index].case_id)))
    for field in ("context", "history_id"):
        if len({getattr(cases[index], field) for index in selected}) != len(selected):
            raise ValueError(f"selected oracle cases must have distinct {field}")
    return selected


class NativeMemoryOracle(nn.Module):
    """Fit a separate code per known case, not an encoder for unseen histories."""

    def __init__(self, cases: int, reader_width: int, *, slots: int = 2, memory_width: int = 8) -> None:
        super().__init__()
        for name, value in (("cases", cases), ("reader_width", reader_width), ("slots", slots), ("memory_width", memory_width)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.codes = nn.Parameter(torch.empty(cases, slots, memory_width))
        nn.init.normal_(self.codes, std=0.1)
        self.read_projection = nn.Linear(memory_width, reader_width, bias=False)

    def state(self, index: int) -> LatentSlotState:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.codes):
            raise ValueError("index must identify one oracle case")
        # Slice before tanh so the state owns only one code's storage.
        values = self.codes[index:index + 1].tanh()
        valid = torch.ones(values.shape[:2], device=values.device, dtype=torch.bool)
        return LatentSlotState(values, valid)

    def forward(self, index: int) -> torch.Tensor:
        return self.read_projection(self.state(index).values[0])


class FixedProjectionMemoryOracle(nn.Module):
    """Fit bounded history codes while keeping their initial read projection fixed."""

    def __init__(self, codes: torch.Tensor, projection: torch.Tensor) -> None:
        super().__init__()
        if not isinstance(codes, torch.Tensor) or not isinstance(projection, torch.Tensor):
            raise TypeError("codes and projection must be tensors")
        if codes.ndim != 3 or min(codes.shape) <= 0 or not codes.is_floating_point():
            raise ValueError("codes must have floating [histories, slots, width] shape")
        if projection.ndim != 2 or min(projection.shape) <= 0 or projection.shape[1] != codes.shape[2]:
            raise ValueError("projection must have [reader_width, code_width] shape")
        if projection.dtype != codes.dtype or projection.device != codes.device:
            raise ValueError("codes and projection must share floating dtype and device")
        if not torch.isfinite(codes).all() or not torch.isfinite(projection).all():
            raise ValueError("codes and projection must be finite")
        if (codes.abs() > 1).any():
            raise ValueError("initial codes must be within [-1, 1]")
        self.codes = nn.Parameter(codes.detach().clone())
        self.register_buffer("projection", projection.detach().clone())

    def state(self, index: int) -> LatentSlotState:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.codes):
            raise ValueError("index must identify one oracle history")
        # Own one history's storage without cutting its gradient path.
        values = self.codes[index:index + 1].clone()
        valid = torch.ones(values.shape[:2], device=values.device, dtype=torch.bool)
        return LatentSlotState(values, valid)

    def forward(self, index: int) -> torch.Tensor:
        return F.linear(self.state(index).values[0], self.projection)

    @torch.no_grad()
    def project_codes_(self) -> None:
        """Apply the declared box constraint after an optimizer step."""
        if not torch.isfinite(self.codes).all():
            raise ValueError("cannot project nonfinite codes")
        self.codes.clamp_(-1, 1)
