"""Fresh bridge and read-side adapter fits with a fixed correct memory."""

import json
from pathlib import Path
import time

import torch
from safetensors.torch import load_file, save_file

from tinymem.reader.adapter import configure_read_adapter
from tinymem.studies.delta.fit import execution_record, save_state_records, selected_episodes
from tinymem.studies.artifacts import frozen_base_hash, write_json
from tinymem.studies.delta.protocol import cell_identity, seal_directory
from tinymem.studies.delta.readout import SlotReadout
from tinymem.studies.oracle.state import oracle_records
from tinymem.studies.oracle.training import encode_oracle_episode, oracle_validation_losses, train_oracle_batch
from tinymem.reader.lora import attach_reader_lora
from tinymem.studies.runtime import allocation_metrics, synchronize


def new_readout(reader, protocol: dict, cell: dict):
    if hasattr(reader.model, "peft_config"):
        raise ValueError("fresh oracle readout requires an unadapted base reader")
    torch.manual_seed(cell["adapter_seed"])
    attach_reader_lora(reader, rank=protocol["settings"]["lora_rank"], checkpointing=False)
    adapters = configure_read_adapter(reader, trainable=True)
    torch.manual_seed(cell["bridge_seed"])
    bridge = SlotReadout(memory_width=32, reader_width=reader.model.config.hidden_size)
    return bridge.to(reader.model.device), adapters


def checkpoint_tensors(reader, bridge) -> dict[str, torch.Tensor]:
    result = {"bridge." + name: tensor.detach().cpu().clone().contiguous()
              for name, tensor in bridge.state_dict().items()}
    result.update({"adapter." + name: parameter.detach().cpu().clone().contiguous()
                   for name, parameter in reader.model.named_parameters()
                   if ".lora_A." in name or ".lora_B." in name})
    return result


def freeze(reader, bridge) -> None:
    configure_read_adapter(reader, trainable=False)
    bridge.zero_grad(set_to_none=True)
    bridge.requires_grad_(False).eval()


def load_trained(reader, protocol: dict, cell: dict, path: Path):
    bridge, _ = new_readout(reader, protocol, cell)
    tensors = load_file(str(path))
    expected = checkpoint_tensors(reader, bridge)
    if tensors.keys() != expected.keys():
        raise ValueError("oracle checkpoint parameter schema differs")
    for name, tensor in tensors.items():
        if (tensor.shape != expected[name].shape or tensor.dtype != expected[name].dtype
                or not bool(torch.isfinite(tensor).all())):
            raise ValueError("oracle checkpoint shape, dtype, or finite values differ")
    bridge.load_state_dict({name.removeprefix("bridge."): tensor for name, tensor in tensors.items()
                            if name.startswith("bridge.")}, strict=True)
    with torch.no_grad():
        for name, parameter in reader.model.named_parameters():
            if "adapter." + name in tensors:
                parameter.copy_(tensors["adapter." + name])
    freeze(reader, bridge)
    return bridge


def train_cell(reader, study: Path, protocol: dict, dataset, index: int) -> dict:
    spec, cell = protocol["settings"], protocol["cells"][index]
    directory = study / "training" / str(index)
    directory.mkdir(parents=True, exist_ok=False)
    if reader.model.device.type != spec["device"]:
        raise ValueError("reader device differs from declaration")
    unadapted_base = frozen_base_hash(reader)
    bridge, adapters = new_readout(reader, protocol, cell)
    base_before = frozen_base_hash(reader)
    runtime = execution_record(reader)
    optimizer = torch.optim.AdamW([*bridge.parameters(), *adapters],
                                 lr=spec["learning_rate"], weight_decay=spec["weight_decay"])
    by_id = {episode.id: episode for episode in dataset.train}
    validation = selected_episodes(dataset.validation, spec)
    curves, step = [], 0
    with (directory / "metrics.jsonl").open("x") as handle:
        for epoch, schedule in enumerate(protocol["schedule"], 1):
            losses = []
            for ids in schedule:
                batch = tuple(encode_oracle_episode(reader, by_id[key]) for key in ids)
                synchronize(reader.model.device)
                started = time.perf_counter()
                metric = train_oracle_batch(reader, bridge, batch, optimizer, adapter_parameters=adapters)
                synchronize(reader.model.device)
                step += 1
                metric.update(step=step, epoch=epoch, seconds=time.perf_counter() - started,
                              **allocation_metrics(reader.model.device))
                handle.write(json.dumps(metric, allow_nan=False) + "\n")
                handle.flush()
                losses.append(metric["answer_ce"])
                if step == 1 or step % 25 == 0:
                    print(json.dumps({"cell": index, **metric}), flush=True)
            freeze(reader, bridge)
            curves.append({"epoch": epoch, "training_episode_mean_ce": sum(losses) / len(losses),
                           "validation": oracle_validation_losses(reader, bridge, validation)})
            write_json(directory / "curves.json", curves)
            if epoch < len(protocol["schedule"]):
                bridge.requires_grad_(True).train()
                adapters = configure_read_adapter(reader, trainable=True)
    base_after = frozen_base_hash(reader)
    if base_before != base_after:
        raise ValueError("oracle training changed the frozen base model")
    save_file(checkpoint_tensors(reader, bridge), str(directory / "checkpoint.safetensors"))
    save_state_records(oracle_records(tuple(e for e in dataset.train if e.condition == "no_write")),
                       directory / "training_states.safetensors")
    report = {"optimizer_steps": step, "epochs": len(curves), "runtime": runtime,
              "unadapted_base_sha256": unadapted_base,
              "base_before_sha256": base_before, "base_after_sha256": base_after,
              "test_scored": False, "checkpoint_selection": "fixed_final", "persistent_bytes": 258,
              "parameters": {"writer": 0, "bridge": sum(p.numel() for p in bridge.parameters()),
                             "adapter": sum(p.numel() for p in adapters)}}
    write_json(directory / "report.json", report)
    seal_directory(directory, cell_identity(study, protocol, index, "training"))
    return report
