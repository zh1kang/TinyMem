"""Diagnostics and byte accounting for discrete memory codes."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import math
from numbers import Integral

import torch

from tinymem.memory.discrete_compressor import DiscreteMemoryCompressor
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.training.continuous import collate_segmented_answer_supervision
from tinymem.training.controlled_qa import EncodedQAExample


@dataclass(frozen=True)
class CodebookDiagnostics:
    """Summarize hard usage, soft uncertainty, and quantization distortion."""

    assignments: int
    active_codes: int
    unused_codes: int
    unused_fraction: float
    most_common_code: int
    most_common_fraction: float
    hard_perplexity: float
    mean_soft_entropy: float
    aggregate_soft_perplexity: float
    quantization_mse: float
    counts: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible diagnostic values."""
        return asdict(self)


@dataclass(frozen=True)
class CodeAttributeDiagnostics:
    """Measure empirical dependence between codes and integer attributes."""

    assignments: int
    active_attributes: int
    mutual_information_nats: float
    joint_counts: tuple[tuple[int, ...], ...]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible diagnostic values."""
        return asdict(self)


@dataclass(frozen=True)
class DiscreteMemoryBudget:
    """Report logical and realized bytes for one example's memory state."""

    capacity: int
    code_bits: int
    logical_bytes: int
    tensor_bytes: int
    shared_codebook_bytes: int

    def to_dict(self) -> dict[str, int]:
        """Return JSON-compatible byte counts."""
        return asdict(self)


def _validate_code_tensors(
    indices: torch.Tensor,
    valid: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    codebook_size: int,
) -> None:
    for name, tensor in (
        ("indices", indices),
        ("valid", valid),
        ("probabilities", probabilities),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
    if isinstance(codebook_size, bool) or not isinstance(codebook_size, Integral):
        raise TypeError("codebook_size must be an integer")
    if codebook_size <= 1:
        raise ValueError("codebook_size must be greater than one")
    if indices.ndim != 2 or indices.numel() == 0:
        raise ValueError("indices must have nonempty shape [batch, assignments]")
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("indices must be an integer tensor")
    if valid.shape != indices.shape:
        raise ValueError(f"valid must have shape {indices.shape}")
    if valid.dtype != torch.bool:
        raise TypeError("valid must be a boolean tensor")
    expected_probabilities = (*indices.shape, int(codebook_size))
    if probabilities.shape != expected_probabilities:
        raise ValueError(
            f"probabilities must have shape {expected_probabilities}"
        )
    if not probabilities.is_floating_point():
        raise TypeError("probabilities must be floating point")
    if any(
        tensor.device != indices.device
        for tensor in (valid, probabilities)
    ):
        raise ValueError("all code diagnostic tensors must share a device")
    if not valid.any():
        raise ValueError("code diagnostics require a valid assignment")
    if ((indices[valid] < 0) | (indices[valid] >= codebook_size)).any():
        raise ValueError("valid code indices must be inside the codebook")
    if (indices[~valid] != -1).any():
        raise ValueError("invalid code indices must be -1")


def summarize_codebook(
    indices: torch.Tensor,
    valid: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    codebook_size: int,
    prequantized: torch.Tensor,
    quantized: torch.Tensor,
) -> CodebookDiagnostics:
    """Return collapse and reconstruction diagnostics for one code trace."""
    _validate_code_tensors(
        indices,
        valid,
        probabilities,
        codebook_size=codebook_size,
    )
    for name, tensor in (
        ("prequantized", prequantized),
        ("quantized", quantized),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim != 3 or tensor.shape[:2] != indices.shape:
            raise ValueError(
                f"{name} must have shape [batch, assignments, model_width]"
            )
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must be floating point")
        if tensor.device != indices.device:
            raise ValueError("all code diagnostic tensors must share a device")
    if prequantized.shape != quantized.shape:
        raise ValueError("prequantized and quantized must share a shape")

    selected_indices = indices[valid].to(dtype=torch.long)
    counts = torch.bincount(
        selected_indices,
        minlength=int(codebook_size),
    )
    total = int(selected_indices.numel())
    hard_distribution = counts.to(dtype=torch.float64) / total
    positive_hard = hard_distribution > 0
    hard_entropy = -(
        hard_distribution[positive_hard]
        * hard_distribution[positive_hard].log()
    ).sum()

    selected_probabilities = probabilities[valid].to(dtype=torch.float64)
    tiny = torch.finfo(selected_probabilities.dtype).tiny
    row_entropy = -(
        selected_probabilities
        * selected_probabilities.clamp_min(tiny).log()
    ).sum(dim=-1)
    aggregate_soft = selected_probabilities.mean(dim=0)
    aggregate_entropy = -(
        aggregate_soft * aggregate_soft.clamp_min(tiny).log()
    ).sum()
    squared_error = (
        prequantized[valid].to(dtype=torch.float64)
        - quantized[valid].to(dtype=torch.float64)
    ).square().mean()
    active_codes = int((counts > 0).sum())
    most_common_count, most_common_code = counts.max(dim=0)
    return CodebookDiagnostics(
        assignments=total,
        active_codes=active_codes,
        unused_codes=int(codebook_size) - active_codes,
        unused_fraction=(int(codebook_size) - active_codes) / int(codebook_size),
        most_common_code=int(most_common_code),
        most_common_fraction=int(most_common_count) / total,
        hard_perplexity=math.exp(float(hard_entropy)),
        mean_soft_entropy=float(row_entropy.mean()),
        aggregate_soft_perplexity=math.exp(float(aggregate_entropy)),
        quantization_mse=float(squared_error),
        counts=tuple(int(count) for count in counts.tolist()),
    )


def summarize_code_attributes(
    indices: torch.Tensor,
    valid: torch.Tensor,
    attributes: torch.Tensor,
    *,
    codebook_size: int,
    attribute_count: int,
) -> CodeAttributeDiagnostics:
    """Estimate mutual information from empirical code-attribute counts."""
    if not isinstance(attributes, torch.Tensor):
        raise TypeError("attributes must be a torch.Tensor")
    for name, value in (
        ("codebook_size", codebook_size),
        ("attribute_count", attribute_count),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        if value <= 1:
            raise ValueError(f"{name} must be greater than one")
    if indices.ndim != 2 or indices.numel() == 0:
        raise ValueError("indices must have nonempty shape [batch, assignments]")
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("indices must be an integer tensor")
    if valid.shape != indices.shape or valid.dtype != torch.bool:
        raise ValueError("valid must be a boolean tensor shaped like indices")
    if attributes.shape != indices.shape:
        raise ValueError("attributes must have the same shape as indices")
    if attributes.dtype not in (torch.int32, torch.int64):
        raise TypeError("attributes must be an integer tensor")
    if any(tensor.device != indices.device for tensor in (valid, attributes)):
        raise ValueError("all attribute diagnostic tensors must share a device")
    if not valid.any():
        raise ValueError("attribute diagnostics require a valid assignment")
    if ((indices[valid] < 0) | (indices[valid] >= codebook_size)).any():
        raise ValueError("valid code indices must be inside the codebook")
    if ((attributes[valid] < 0) | (attributes[valid] >= attribute_count)).any():
        raise ValueError("valid attributes must be inside the attribute range")

    selected_codes = indices[valid].to(dtype=torch.long)
    selected_attributes = attributes[valid].to(dtype=torch.long)
    flat_joint = selected_attributes * int(codebook_size) + selected_codes
    joint_counts = torch.bincount(
        flat_joint,
        minlength=int(attribute_count) * int(codebook_size),
    ).reshape(int(attribute_count), int(codebook_size))
    joint = joint_counts.to(dtype=torch.float64) / selected_codes.numel()
    attribute_marginal = joint.sum(dim=1, keepdim=True)
    code_marginal = joint.sum(dim=0, keepdim=True)
    expected = attribute_marginal * code_marginal
    observed = joint > 0
    mutual_information = (
        joint[observed] * (joint[observed] / expected[observed]).log()
    ).sum()
    return CodeAttributeDiagnostics(
        assignments=int(selected_codes.numel()),
        active_attributes=int((attribute_marginal.squeeze(1) > 0).sum()),
        mutual_information_nats=float(mutual_information),
        joint_counts=tuple(
            tuple(int(count) for count in row)
            for row in joint_counts.tolist()
        ),
    )


def continuous_memory_bytes(
    *,
    capacity: int,
    model_width: int,
    element_size: int,
) -> int:
    """Return realized value, validity, and position bytes per example."""
    for name, value in (
        ("capacity", capacity),
        ("model_width", model_width),
        ("element_size", element_size),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    return int(capacity) * (int(model_width) * int(element_size) + 1 + 8)


def discrete_memory_budget(
    *,
    capacity: int,
    codebook_size: int,
    model_width: int,
    element_size: int,
) -> DiscreteMemoryBudget:
    """Return logical, tensor, and shared-parameter byte counts."""
    for name, value in (
        ("capacity", capacity),
        ("codebook_size", codebook_size),
        ("model_width", model_width),
        ("element_size", element_size),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if codebook_size <= 1:
        raise ValueError("codebook_size must be greater than one")
    code_bits = math.ceil(math.log2(int(codebook_size)))
    logical_bits = int(capacity) * (code_bits + 1 + 64)
    return DiscreteMemoryBudget(
        capacity=int(capacity),
        code_bits=code_bits,
        logical_bytes=math.ceil(logical_bits / 8),
        tensor_bytes=int(capacity) * (8 + 1 + 8),
        shared_codebook_bytes=(
            int(codebook_size) * int(model_width) * int(element_size)
        ),
    )


def matched_discrete_capacity(continuous_bytes: int) -> int:
    """Return the largest int64-code capacity within a continuous budget."""
    if isinstance(continuous_bytes, bool) or not isinstance(
        continuous_bytes,
        Integral,
    ):
        raise TypeError("continuous_bytes must be an integer")
    if continuous_bytes < 17:
        raise ValueError("continuous_bytes must hold at least one code slot")
    return int(continuous_bytes) // 17


@torch.no_grad()
def evaluate_codebook_diagnostics(
    decoder: SegmentedContinuousDecoder,
    examples: Sequence[EncodedQAExample],
    *,
    batch_size: int,
    pad_id: int,
    device: torch.device | str,
) -> CodebookDiagnostics:
    """Aggregate code usage over an encoded evaluation collection."""
    if not isinstance(decoder, SegmentedContinuousDecoder):
        raise TypeError("decoder must be a SegmentedContinuousDecoder")
    if not isinstance(decoder.compressor, DiscreteMemoryCompressor):
        raise ValueError("codebook diagnostics require a discrete compressor")
    if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)):
        raise TypeError("examples must be a sequence")
    if not examples:
        raise ValueError("examples must be nonempty")
    if not all(isinstance(example, EncodedQAExample) for example in examples):
        raise TypeError("examples must contain EncodedQAExample values")
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    all_indices = []
    all_valid = []
    all_probabilities = []
    all_prequantized = []
    all_quantized = []
    was_training = decoder.training
    decoder.eval()
    try:
        for start in range(0, len(examples), int(batch_size)):
            batch = examples[start : start + int(batch_size)]
            input_ids, _, token_valid = collate_segmented_answer_supervision(
                batch,
                pad_id=pad_id,
                device=device,
            )
            output = decoder(input_ids, token_valid)
            if (
                output.proposed_code_indices is None
                or output.proposed_code_valid is None
                or output.code_probabilities is None
                or output.prequantized_codes is None
            ):
                raise RuntimeError("discrete decoder did not return code traces")
            indices = output.proposed_code_indices
            summary_slots = (
                indices.shape[1] // output.writes_applied.shape[1]
            )
            written = output.writes_applied.repeat_interleave(
                summary_slots,
                dim=1,
            )
            valid = output.proposed_code_valid & written
            quantized = decoder.compressor.codebook.embedding(
                indices.clamp_min(0)
            )
            quantized = quantized * valid.unsqueeze(-1)
            all_indices.append(indices.reshape(1, -1).cpu())
            all_valid.append(valid.reshape(1, -1).cpu())
            all_probabilities.append(
                output.code_probabilities.reshape(
                    1,
                    -1,
                    decoder.compressor.codebook_size,
                ).cpu()
            )
            all_prequantized.append(
                output.prequantized_codes.reshape(
                    1,
                    -1,
                    decoder.compressor.model_width,
                ).cpu()
            )
            all_quantized.append(
                quantized.reshape(
                    1,
                    -1,
                    decoder.compressor.model_width,
                ).cpu()
            )
    finally:
        decoder.train(was_training)

    return summarize_codebook(
        torch.cat(all_indices, dim=1),
        torch.cat(all_valid, dim=1),
        torch.cat(all_probabilities, dim=1),
        codebook_size=decoder.compressor.codebook_size,
        prequantized=torch.cat(all_prequantized, dim=1),
        quantized=torch.cat(all_quantized, dim=1),
    )


@torch.no_grad()
def collect_codebook_traces(
    decoder: SegmentedContinuousDecoder,
    examples: Sequence[EncodedQAExample],
    *,
    batch_size: int,
    pad_id: int,
    device: torch.device | str,
) -> tuple[dict[str, object], ...]:
    """Return one JSON-compatible discrete-memory trace per example."""
    if not isinstance(decoder, SegmentedContinuousDecoder):
        raise TypeError("decoder must be a SegmentedContinuousDecoder")
    if not isinstance(decoder.compressor, DiscreteMemoryCompressor):
        raise ValueError("code traces require a discrete compressor")
    if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)):
        raise TypeError("examples must be a sequence")
    if not examples or not all(
        isinstance(example, EncodedQAExample) for example in examples
    ):
        raise ValueError("examples must contain EncodedQAExample values")
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    traces = []
    was_training = decoder.training
    decoder.eval()
    try:
        for start in range(0, len(examples), int(batch_size)):
            batch = examples[start : start + int(batch_size)]
            input_ids, _, token_valid = collate_segmented_answer_supervision(
                batch,
                pad_id=pad_id,
                device=device,
            )
            output = decoder(input_ids, token_valid)
            if (
                output.proposed_code_indices is None
                or output.proposed_code_valid is None
                or output.memory_codes is None
            ):
                raise RuntimeError("discrete decoder did not return code traces")
            summary_slots = decoder.compressor.summary_slots
            for row, example in enumerate(batch):
                segment_count = (
                    len(example.input_ids) + decoder.segment_length - 1
                ) // decoder.segment_length
                proposal_count = segment_count * summary_slots
                codes = output.proposed_code_indices[
                    row,
                    :proposal_count,
                ].reshape(segment_count, summary_slots)
                valid = output.proposed_code_valid[
                    row,
                    :proposal_count,
                ].reshape(segment_count, summary_slots)
                writes = output.writes_applied[row, :segment_count]
                written_codes = [
                    [int(code) for code in codes[index][valid[index]].cpu()]
                    if bool(writes[index])
                    else []
                    for index in range(segment_count)
                ]
                traces.append(
                    {
                        "source_example_id": example.source_example_id,
                        "proposed_codes": [
                            [
                                int(code)
                                for code in codes[index][valid[index]].cpu()
                            ]
                            for index in range(segment_count)
                        ],
                        "writes_applied": [
                            bool(value) for value in writes.cpu()
                        ],
                        "written_codes": written_codes,
                        "final_memory_codes": [
                            int(code)
                            for code in output.memory_codes[row][
                                output.memory_valid[row]
                            ].cpu()
                        ],
                    }
                )
    finally:
        decoder.train(was_training)
    return tuple(traces)
