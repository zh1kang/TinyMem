"""Shared JSON, hashing, and module-construction helpers for study artifacts."""

from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
import time

import torch
from safetensors.torch import load_file, save_file

from tinymem.memory.delta_slots import DeltaSlotWriter
from tinymem.memory.gated_slots import QueryPoolSlotWriter
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.reader.adapter import configure_read_adapter
from tinymem.studies.delta.readout import SlotReadout, own_state, read_answer
from tinymem.studies.delta.training import TrainingExample, Writer, train_batch
from tinymem.reader.pretrained import PretrainedReader
from tinymem.studies.runtime import allocation_metrics, synchronize


CELLS = (("gated", 8), ("delta", 8), ("gated", 32), ("delta", 32))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def frozen_base_hash(reader: PretrainedReader) -> str:
    """Hash the base including buffers, excluding only ordinary LoRA weights."""
    digest = hashlib.sha256()
    for name, value in sorted(reader.model.state_dict().items()):
        if ".lora_A." in name or ".lora_B." in name:
            continue
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def new_modules(kind: str, width: int, reader_width: int, device: torch.device, seed: int):
    if (kind, width) not in CELLS:
        raise ValueError("unsupported writer or state width")
    torch.manual_seed(seed)
    writer = (QueryPoolSlotWriter(reader_width, width, 2) if kind == "gated"
              else DeltaSlotWriter(reader_width, width, key_width=4 if width == 8 else 8))
    # Readout initialization is independent of the writer's parameter count.
    torch.manual_seed(seed + 1)
    bridge = SlotReadout(memory_width=width, reader_width=reader_width)
    return writer.to(device), bridge.to(device)


@torch.no_grad()
def replay_state(writer: Writer, example: TrainingExample) -> LatentSlotState:
    state = writer.empty(1)
    for hidden in example.features:
        features = hidden.to(state.values.device).unsqueeze(0)
        state = writer(state, features, torch.ones(features.shape[:2], device=features.device, dtype=torch.bool))
    return own_state(state, 0)


def _checkpoint_tensors(reader, writer, bridge, adapter_names):
    result = {"writer." + name: value.detach().cpu().contiguous().clone()
              for name, value in writer.state_dict().items()}
    result.update({"bridge." + name: value.detach().cpu().contiguous().clone()
                   for name, value in bridge.state_dict().items()})
    result.update({"adapter." + name: value.detach().cpu().contiguous().clone()
                   for name, value in reader.model.named_parameters() if name in adapter_names})
    return result


def profile_cell(
    reader: PretrainedReader, batches: Sequence[Sequence[TrainingExample]], directory: Path, *,
    kind: str, width: int, seed: int, warmup_steps: int, learning_rate: float = 1e-3,
    max_new_tokens: int = 8,
) -> dict[str, object]:
    """Train a fresh writer/readout and the already reset reader adapter.

    The caller owns adapter reset between cells and external provenance.
    Checkpoints are disposable and are not a confirmation training endpoint.
    """
    if not batches or not 0 <= warmup_steps < len(batches):
        raise ValueError("profile must contain measured steps after warmup")
    if any(not batch or any(e.split != "train" for e in batch) for batch in batches):
        raise ValueError("profile accepts nonempty training batches only")
    directory.mkdir(parents=True, exist_ok=False)
    device = reader.model.device
    writer, bridge = new_modules(kind, width, reader.model.config.hidden_size, device, seed)
    adapters = configure_read_adapter(reader, trainable=True)
    adapter_names = {name for name, p in reader.model.named_parameters() if p.requires_grad}
    initial = _checkpoint_tensors(reader, writer, bridge, adapter_names)
    save_file(initial, str(directory / "initial.safetensors"))
    optimizer = torch.optim.AdamW([*writer.parameters(), *bridge.parameters(), *adapters],
                                 lr=learning_rate, weight_decay=0.01)
    records = []
    with (directory / "metrics.jsonl").open("x") as handle:
        for step, batch in enumerate(batches):
            synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            metric = train_batch(reader, writer, bridge, batch, optimizer, adapter_parameters=adapters)
            synchronize(device)
            metric.update(step=step + 1, warmup=step < warmup_steps,
                          seconds=time.perf_counter() - start, **allocation_metrics(device))
            handle.write(json.dumps(metric, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            records.append(metric)
            print(json.dumps({"cell": f"{kind}_{bridge.state_bytes}", **metric}), flush=True)
    final = _checkpoint_tensors(reader, writer, bridge, adapter_names)
    changed = {group: any(not torch.equal(value, initial[name]) for name, value in final.items()
                          if name.startswith(group + ".")) for group in ("writer", "bridge", "adapter")}
    save_file(final, str(directory / "final_disposable.safetensors"))
    configure_read_adapter(reader, trainable=False)
    writer.zero_grad(set_to_none=True)
    bridge.zero_grad(set_to_none=True)
    writer.requires_grad_(False).eval()
    bridge.requires_grad_(False).eval()
    example = batches[-1][-1]
    state = replay_state(writer, example)
    before = torch.tensor(example.before_ids, device=device)
    question = torch.tensor(example.endpoints[-1].queries[0].after_ids, device=device)
    prediction = read_answer(reader, bridge, state, before, question, max_new_tokens=max_new_tokens)
    save_file({"values": state.values.cpu().contiguous(), "valid": state.valid.cpu().contiguous()},
              str(directory / "state.safetensors"))
    # Reload into fresh writer/readout instances and explicitly reset adapter weights.
    loaded = load_file(str(directory / "final_disposable.safetensors"), device=str(device))
    restored_writer, restored_bridge = new_modules(kind, width, reader.model.config.hidden_size, device, seed + 99)
    for label, module in (("writer", restored_writer), ("bridge", restored_bridge)):
        module.load_state_dict({k.removeprefix(label + "."): v for k, v in loaded.items() if k.startswith(label + ".")})
        module.requires_grad_(False).eval()
    with torch.no_grad():
        for name, parameter in reader.model.named_parameters():
            if name in adapter_names:
                parameter.zero_()
                parameter.copy_(loaded["adapter." + name])
    tensors = load_file(str(directory / "state.safetensors"), device=str(device))
    restored_state = LatentSlotState(tensors["values"], tensors["valid"])
    replayed = replay_state(restored_writer, example)
    if (not torch.equal(replayed.values, restored_state.values)
            or not torch.equal(replayed.valid, restored_state.valid)):
        raise ValueError("checkpoint replay changed the stored state")
    restored_prediction = read_answer(reader, restored_bridge, restored_state, before, question,
                                      max_new_tokens=max_new_tokens)
    if prediction != restored_prediction:
        raise ValueError("serialized state read changed after checkpoint reload")
    measured = records[warmup_steps:]
    summary = {
        "writer": kind, "memory_width": width, "persistent_bytes": bridge.state_bytes,
        "key_width": getattr(writer, "key_width", None), "seed": seed,
        "optimizer_steps": len(records), "warmup_steps": warmup_steps,
        "mean_measured_step_seconds": sum(r["seconds"] for r in measured) / len(measured),
        "parameters": {"writer": sum(p.numel() for p in writer.parameters()),
                       "bridge": sum(p.numel() for p in bridge.parameters()),
                       "adapter": sum(p.numel() for p in adapters)},
        "parameter_groups_changed": changed, "checkpoint_and_state_roundtrip": True,
        "training_read_example": example.episode_id, "training_read": prediction,
        "accuracy_scored": False, "checkpoint_reuse": False,
        "files": {p.name: file_hash(p) for p in sorted(directory.iterdir()) if p.is_file()},
    }
    write_json(directory / "report.json", summary)
    return summary
