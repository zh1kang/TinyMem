"""Fixed-endpoint phase-two training with descriptive validation curves."""

from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import time

import torch
from safetensors.torch import load_file, save_file

from tinymem.research.adapted_readout import configure_read_adapter
from tinymem.research.delta_fact_encoding import build_feature_cache, encode_episode
from tinymem.research.delta_fact_evaluation import StateRecord, collect_states, delta_geometry, fit_probe
from tinymem.research.delta_fact_profile import _checkpoint_tensors, frozen_base_hash, new_modules, write_json
from tinymem.research.delta_fact_protocol import cell_identity, seal_directory
from tinymem.research.delta_fact_training import train_batch
from tinymem.research.prefix_reader import prefix_answer_losses
from tinymem.research.reader_adaptation import attach_reader_lora
from tinymem.research.study_runtime import allocation_metrics, synchronize


def execution_record(reader) -> dict:
    device = reader.model.device
    return {"device": device.type, "reader_dtype": str(reader.model.dtype),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "peft", "safetensors", "numpy")},
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}


def freeze(reader, writer, bridge) -> None:
    configure_read_adapter(reader, trainable=False)
    for module in (writer, bridge):
        module.zero_grad(set_to_none=True)
        module.requires_grad_(False).eval()


def selected_episodes(episodes, spec):
    limit = spec["evaluation_prefix_limit"]
    if limit is None:
        return episodes
    ids = sorted({e.prefix_id for e in episodes})[:limit]
    return tuple(e for e in episodes if e.prefix_id in ids)


@torch.no_grad()
def validation_losses(reader, writer, bridge, episodes, features) -> dict:
    totals = {}
    device = reader.model.device
    for episode in episodes:
        example = encode_episode(reader, episode, features)
        if example.split != "validation":
            raise ValueError("validation curves accept validation examples only")
        endpoints = {e.after_write: e for e in example.endpoints}
        state = writer.empty(1)
        reads = []
        for position, hidden in enumerate(example.features, 1):
            hidden = hidden.to(device).unsqueeze(0)
            state = writer(state, hidden, torch.ones(hidden.shape[:2], dtype=torch.bool, device=device))
            if position in endpoints:
                memory = bridge(state)[0]
                for query in endpoints[position].queries:
                    reads.append((torch.tensor(example.before_ids, device=device), memory,
                                  torch.tensor(query.after_ids, device=device),
                                  torch.tensor(query.answer_ids, device=device)))
        value = float(prefix_answer_losses(reader, reads).mean())
        row = totals.setdefault(episode.condition, {"sum_episode_ce": 0.0, "episodes": 0})
        row["sum_episode_ce"] += value
        row["episodes"] += 1
    return {name: {"episode_mean_ce": row["sum_episode_ce"] / row["episodes"], "episodes": row["episodes"]}
            for name, row in totals.items()}


def save_state_records(records, path: Path) -> None:
    save_file({f"{i}.{name}": getattr(row, name).contiguous()
               for i, row in enumerate(records) for name in ("values", "valid")}, str(path))
    metadata = [{key: value for key, value in asdict(row).items() if key not in ("values", "valid")}
                for row in records]
    write_json(path.with_suffix(".json"), metadata)


def load_state_records(path: Path) -> tuple[StateRecord, ...]:
    tensors = load_file(str(path))
    metadata = json.loads(path.with_suffix(".json").read_text())
    if set(tensors) != {f"{i}.{name}" for i in range(len(metadata)) for name in ("values", "valid")}:
        raise ValueError("saved state schema differs from its metadata")
    return tuple(StateRecord(**{**row, "truth": tuple(row["truth"]),
                    "values": tensors[f"{i}.values"].clone().contiguous(),
                    "valid": tensors[f"{i}.valid"].clone().contiguous()})
                 for i, row in enumerate(metadata))


def load_trained(reader, protocol: dict, cell: dict, path: Path):
    if hasattr(reader.model, "peft_config"):
        raise ValueError("load requires the unadapted base reader")
    torch.manual_seed(cell["adapter_seed"])
    attach_reader_lora(reader, rank=protocol["settings"]["lora_rank"], checkpointing=False)
    adapters = configure_read_adapter(reader, trainable=True)
    adapter_ids = {id(p) for p in adapters}
    names = {name for name, p in reader.model.named_parameters() if id(p) in adapter_ids}
    writer, bridge = new_modules(cell["writer"], cell["width"], reader.model.config.hidden_size,
                                 reader.model.device, cell["writer_seed"])
    tensors = load_file(str(path), device=str(reader.model.device))
    expected = _checkpoint_tensors(reader, writer, bridge, names)
    if tensors.keys() != expected.keys():
        raise ValueError("checkpoint parameter schema differs")
    if any(tensors[name].shape != expected[name].shape or tensors[name].dtype != expected[name].dtype
           or not bool(torch.isfinite(tensors[name]).all()) for name in expected):
        raise ValueError("checkpoint shape, dtype, or finite-value contract differs")
    for label, module in (("writer", writer), ("bridge", bridge)):
        module.load_state_dict({n.removeprefix(label + "."): t for n, t in tensors.items() if n.startswith(label + ".")})
    with torch.no_grad():
        for name, parameter in reader.model.named_parameters():
            if name in names:
                if parameter.shape != tensors["adapter." + name].shape or parameter.dtype != tensors["adapter." + name].dtype:
                    raise ValueError("adapter checkpoint shape or dtype differs")
                parameter.copy_(tensors["adapter." + name])
    freeze(reader, writer, bridge)
    return writer, bridge


def train_cell(reader, study: Path, protocol: dict, dataset, index: int) -> dict:
    spec, cell = protocol["settings"], protocol["cells"][index]
    directory = study / "training" / str(index)
    directory.mkdir(parents=True, exist_ok=False)
    if reader.model.device.type != spec["device"]:
        raise ValueError("reader device differs from declaration")
    # Features are base-only, including the descriptive validation examples.
    validation = selected_episodes(dataset.validation, spec)
    features = build_feature_cache(reader, (*dataset.train, *validation))
    texts = sorted(features)
    save_file({str(i): features[text] for i, text in enumerate(texts)}, str(directory / "features.safetensors"))
    write_json(directory / "feature_texts.json", texts)
    unadapted_base = frozen_base_hash(reader)
    torch.manual_seed(cell["adapter_seed"])
    attach_reader_lora(reader, rank=spec["lora_rank"], checkpointing=False)
    adapters = configure_read_adapter(reader, trainable=True)
    adapter_names = {name for name, p in reader.model.named_parameters() if p.requires_grad}
    writer, bridge = new_modules(cell["writer"], cell["width"], reader.model.config.hidden_size,
                                 reader.model.device, cell["writer_seed"])
    base_before = frozen_base_hash(reader)
    runtime = execution_record(reader)
    optimizer = torch.optim.AdamW([*writer.parameters(), *bridge.parameters(), *adapters],
                                 lr=spec["learning_rate"], weight_decay=spec["weight_decay"])
    by_id = {e.id: e for e in dataset.train}
    curves, step = [], 0
    with (directory / "metrics.jsonl").open("x") as handle:
        for epoch, schedule in enumerate(protocol["schedule"], 1):
            train_loss = []
            for ids in schedule:
                batch = tuple(encode_episode(reader, by_id[key], features) for key in ids)
                synchronize(reader.model.device)
                started = time.perf_counter()
                metric = train_batch(reader, writer, bridge, batch, optimizer, adapter_parameters=adapters)
                synchronize(reader.model.device)
                step += 1
                metric.update(step=step, epoch=epoch, seconds=time.perf_counter() - started,
                              **allocation_metrics(reader.model.device))
                handle.write(json.dumps(metric, allow_nan=False) + "\n")
                handle.flush()
                train_loss.append(metric["answer_ce"])
                if step % 25 == 0 or step == 1:
                    print(json.dumps({"cell": index, **metric}), flush=True)
            freeze(reader, writer, bridge)
            curves.append({"epoch": epoch, "training_episode_mean_ce": sum(train_loss) / len(train_loss),
                           "validation": validation_losses(reader, writer, bridge, validation, features)})
            write_json(directory / "curves.json", curves)
            if epoch < len(protocol["schedule"]):
                writer.requires_grad_(True).train()
                bridge.requires_grad_(True).train()
                adapters = configure_read_adapter(reader, trainable=True)
    base_after = frozen_base_hash(reader)
    if base_before != base_after:
        raise ValueError("training changed the frozen base model")
    save_file(_checkpoint_tensors(reader, writer, bridge, adapter_names), str(directory / "checkpoint.safetensors"))
    prefix_records = collect_states(writer, tuple(e for e in dataset.train if e.condition == "no_write"), features)
    save_state_records(prefix_records, directory / "training_states.safetensors")
    probe = fit_probe(load_state_records(directory / "training_states.safetensors"))
    write_json(directory / "probe.json", probe)
    training_texts = {s.text for e in dataset.train for s in (*e.prefix, *e.tail)}
    geometry = delta_geometry(writer, {text: features[text] for text in sorted(training_texts)}) if cell["writer"] == "delta" else None
    report = {"optimizer_steps": step, "epochs": len(curves), "runtime": runtime,
              "unadapted_base_sha256": unadapted_base,
              "base_before_sha256": base_before, "base_after_sha256": base_after,
              "test_scored": False, "checkpoint_selection": "fixed_final",
              "persistent_bytes": cell["persistent_bytes"], "key_geometry_training_only": geometry,
              "parameters": {"writer": sum(p.numel() for p in writer.parameters()),
                             "bridge": sum(p.numel() for p in bridge.parameters()),
                             "adapter": sum(p.numel() for p in adapters)}}
    write_json(directory / "report.json", report)
    seal_directory(directory, cell_identity(study, protocol, index, "training"))
    return report
