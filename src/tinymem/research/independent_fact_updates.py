"""Query-independent correction and repetition writer training cases."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from tinymem.memory.query_pool_slots import QueryPoolSlotWriter
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.independent_fact_data import ENTITIES, ROOM_PAIRS
from tinymem.research.independent_fact_placement import placement_state


@dataclass(frozen=True)
class UpdateCase:
    case_id: str
    before_code: int
    target_fact: int
    new_bit: int
    after_code: int
    kind: str
    split: str
    event_text: str


UpdateBatch = tuple[LatentSlotState, torch.Tensor, torch.Tensor, torch.Tensor]


def build_update_cases() -> tuple[UpdateCase, ...]:
    """Return the fixed 128 correction and repetition events in protocol order."""
    cases = []
    for code in range(16):
        for fact in range(4):
            old_bit = (code >> fact) & 1
            for bit in range(2):
                after_code = (code & ~(1 << fact)) | (bit << fact)
                cases.append(UpdateCase(
                    case_id=f"code-{code:02d}:fact-{fact}:value-{bit}",
                    before_code=code,
                    target_fact=fact,
                    new_bit=bit,
                    after_code=after_code,
                    kind="repetition" if old_bit == bit else "correction",
                    split="train" if (code & ~(1 << fact)).bit_count() % 2 == 0 else "heldout",
                    event_text=f"{ENTITIES[fact]} moved to the {ROOM_PAIRS[fact][bit]}.",
                ))
    return tuple(cases)


def _validate_case(case: UpdateCase) -> None:
    if not isinstance(case, UpdateCase):
        raise TypeError("cases must contain UpdateCase values")
    if type(case.before_code) is not int or not 0 <= case.before_code < 16:
        raise ValueError("before_code must be an integer from zero through fifteen")
    if type(case.target_fact) is not int or case.target_fact not in range(4):
        raise ValueError("target_fact must be zero through three")
    if type(case.new_bit) is not int or case.new_bit not in (0, 1):
        raise ValueError("new_bit must be zero or one")
    if (case.case_id != f"code-{case.before_code:02d}:fact-{case.target_fact}:value-{case.new_bit}"
            or case.event_text != f"{ENTITIES[case.target_fact]} moved to the {ROOM_PAIRS[case.target_fact][case.new_bit]}."):
        raise ValueError("case identity or event text differs")
    expected_after = (case.before_code & ~(1 << case.target_fact)) | (case.new_bit << case.target_fact)
    if case.after_code != expected_after:
        raise ValueError("after_code does not match the update")
    expected_kind = "repetition" if ((case.before_code >> case.target_fact) & 1) == case.new_bit else "correction"
    expected_split = "train" if (case.before_code & ~(1 << case.target_fact)).bit_count() % 2 == 0 else "heldout"
    if case.kind != expected_kind or case.split != expected_split:
        raise ValueError("update case metadata differs")


def _validate_features(
    features: Mapping[tuple[int, int], torch.Tensor], reader_width: int,
) -> dict[tuple[int, int], torch.Tensor]:
    expected = {(fact, bit) for fact in range(4) for bit in range(2)}
    if set(features) != expected:
        raise ValueError("features must cover every fact and value")
    result = {}
    for key, value in features.items():
        if (value.ndim != 2 or value.shape[0] == 0 or value.shape[1] != reader_width
                or value.device.type != "cpu" or value.dtype != torch.float32
                or value.requires_grad or not torch.isfinite(value).all()):
            raise ValueError("features must be detached CPU FP32 token matrices")
        result[key] = value.detach().clone()
    return result


def pack_update_batch(cases: Sequence[UpdateCase], features: Mapping[tuple[int, int], torch.Tensor]) -> UpdateBatch:
    """Pack update events into padded CPU tensors for a query-independent writer."""
    if not cases:
        raise ValueError("at least one update case is required")
    for case in cases:
        _validate_case(case)
    width = next(iter(features.values())).shape[1] if features else 0
    feature_copy = _validate_features(features, width)
    maximum = max(feature_copy[(case.target_fact, case.new_bit)].shape[0] for case in cases)
    hidden = torch.zeros(len(cases), maximum, width, dtype=torch.float32)
    valid = torch.zeros(len(cases), maximum, dtype=torch.bool)
    old_values, target_values = [], []
    for row, case in enumerate(cases):
        feature = feature_copy[(case.target_fact, case.new_bit)]
        hidden[row, : feature.shape[0]] = feature
        valid[row, : feature.shape[0]] = True
        old_values.append(placement_state(case.before_code, torch.device("cpu"), "separate_fact0"))
        target_values.append(placement_state(case.after_code, torch.device("cpu"), "separate_fact0").values[0])
    old = LatentSlotState(
        torch.stack([state.values[0] for state in old_values]),
        torch.stack([state.valid[0] for state in old_values]),
    )
    return old, hidden, valid, torch.stack(target_values)


def new_update_writer(reader_width: int, seed: int) -> QueryPoolSlotWriter:
    """Create a reproducibly initialized two-slot query-pool writer."""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        return QueryPoolSlotWriter(reader_width, 8, 2, hidden_width=64)


def _validate_batch(batch: UpdateBatch, writer: QueryPoolSlotWriter) -> None:
    if not isinstance(batch, tuple) or len(batch) != 4:
        raise TypeError("batch must contain old state, hidden, valid, and targets")
    old, hidden, valid, target = batch
    if (not isinstance(old, LatentSlotState) or hidden.ndim != 3 or valid.shape != hidden.shape[:2]
            or valid.dtype != torch.bool or target.shape != (hidden.shape[0], writer.slots, writer.memory_width)
            or any(tensor.device.type != "cpu" for tensor in (old.values, old.valid, hidden, valid, target))
            or any(tensor.dtype != torch.float32 for tensor in (old.values, hidden, target))
            or old.values.shape != target.shape or old.valid.shape != valid.shape[:1] + (writer.slots,)
            or not old.valid.all() or not torch.isfinite(hidden).all() or not torch.isfinite(target).all()):
        raise ValueError("batch tensors have incompatible shapes, dtypes, or devices")
    if old.values.requires_grad or hidden.requires_grad or target.requires_grad:
        raise ValueError("batch tensors must be detached")


def train_update_batch(
    writer: QueryPoolSlotWriter,
    batch: UpdateBatch,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Train the writer on a mean coordinate MSE for one packed batch."""
    _validate_batch(batch, writer)
    parameters = tuple(writer.parameters())
    optimized = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    owned = {id(parameter) for parameter in parameters}
    if (len(owned) != len(parameters) or len(optimized) != len(owned)
            or {id(parameter) for parameter in optimized} != owned
            or any(not parameter.requires_grad for parameter in parameters)):
        raise ValueError("optimizer must own exactly all writer parameters")
    old, hidden, valid, target = batch
    old_values, old_valid, hidden_copy, valid_copy, target_copy = (
        old.values.clone(), old.valid.clone(), hidden.clone(), valid.clone(), target.clone()
    )
    optimizer.zero_grad(set_to_none=True)
    output = writer(old, hidden, valid)
    loss = F.mse_loss(output.values, target)
    if not torch.isfinite(loss):
        raise ValueError("nonfinite state MSE")
    loss.backward()
    if any(parameter.grad is None for parameter in parameters):
        raise ValueError("all writer parameters must receive gradients")
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    if not torch.isfinite(norm):
        raise ValueError("nonfinite writer gradient norm")
    optimizer.step()
    if any(not torch.isfinite(parameter).all() for parameter in parameters):
        raise ValueError("optimizer produced nonfinite writer parameters")
    if (not torch.equal(old.values, old_values) or not torch.equal(old.valid, old_valid)
            or not torch.equal(hidden, hidden_copy) or not torch.equal(valid, valid_copy)
            or not torch.equal(target, target_copy)):
        raise ValueError("update batch was mutated")
    if not output.valid.all() or output.nbytes // output.values.shape[0] != 66:
        raise ValueError("writer output must retain two occupied 66-byte states")
    return {"state_mse": float(loss.detach()), "gradient_norm": float(norm)}
