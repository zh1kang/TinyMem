"""Privileged training-only codes that isolate the native memory readout."""

from collections.abc import Sequence

import torch
from torch import nn

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
