"""Hash-checked, final-weight checkpoints for the paired readout modules."""

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load, save

from tinymem.memory.readout_interface import OneShotEncoder, ReadoutBridge, ReadoutKind


def _identity(reader_width: int, kind: ReadoutKind) -> dict[str, str]:
    if type(reader_width) is not int or reader_width <= 0 or kind not in ("affine", "gelu"):
        raise ValueError("invalid checkpoint identity")
    return {"format": "tinymem-readout-v1", "reader_width": str(reader_width), "kind": kind}


def _weights(encoder: OneShotEncoder, bridge: ReadoutBridge) -> dict[str, torch.Tensor]:
    return {f"{prefix}.{name}": tensor for prefix, module in (("encoder", encoder), ("bridge", bridge))
            for name, tensor in module.state_dict().items()}


def _validate(weights: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    if weights.keys() != expected.keys():
        raise ValueError("checkpoint parameter keys do not match")
    for name, tensor in weights.items():
        if tensor.shape != expected[name].shape or tensor.dtype != torch.float32:
            raise ValueError(f"checkpoint parameter {name} must have the declared shape and FP32 dtype")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"checkpoint parameter {name} must be finite")


def save_checkpoint(path: Path, encoder: OneShotEncoder, bridge: ReadoutBridge) -> str:
    """Create a new weights-only artifact; never overwrite a previous checkpoint."""
    metadata = _identity(encoder.reader_width, bridge.kind)
    if encoder.reader_width != bridge.output_projection.out_features:
        raise ValueError("encoder and bridge checkpoint identity differs")
    weights = _weights(encoder, bridge)
    # Validate canonical shapes without advancing the caller's random stream.
    with torch.random.fork_rng(devices=[]):
        expected = _weights(OneShotEncoder(encoder.reader_width), ReadoutBridge(encoder.reader_width, bridge.kind))
    _validate(weights, expected)
    payload = save({name: value.detach().cpu().contiguous() for name, value in weights.items()}, metadata=metadata)
    with Path(path).open("xb") as output:
        output.write(payload)
    return hashlib.sha256(payload).hexdigest()


def load_checkpoint(
    path: Path, *, expected_sha256: str, reader_width: int, kind: ReadoutKind,
) -> tuple[OneShotEncoder, ReadoutBridge]:
    """Verify the exact bytes before parsing; return fresh CPU modules in eval mode.

    This is not optimizer resume. Reader and data provenance belong to the run
    protocol. The supplied digest must come from that independently bound record.
    """
    identity = _identity(reader_width, kind)
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("checkpoint SHA-256 mismatch")
    weights = load(payload)
    # Safetensors validates the header and offsets above; inspect those same bytes.
    header_length = int.from_bytes(payload[:8], "little")
    metadata = json.loads(payload[8:8 + header_length]).get("__metadata__")
    if metadata != identity:
        raise ValueError("checkpoint identity mismatch")
    with torch.random.fork_rng(devices=[]):
        encoder, bridge = OneShotEncoder(reader_width), ReadoutBridge(reader_width, kind)
    _validate(weights, _weights(encoder, bridge))
    for prefix, module in (("encoder", encoder), ("bridge", bridge)):
        module.load_state_dict({name.removeprefix(prefix + "."): value
                                for name, value in weights.items() if name.startswith(prefix + ".")}, strict=True)
        module.eval()
    return encoder, bridge
