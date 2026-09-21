"""Fixed, answer-only training for the storage frontier comparison."""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from tinymem.reader.adapter import configure_read_adapter
from tinymem.reader.lora import attach_reader_lora
from tinymem.studies.frontier.baselines import SUPPORTED_CODECS, TextStore
from tinymem.studies.frontier.data import StorageQuestion, write_records
from tinymem.studies.frontier.training import (
    AnswerTokens,
    answer_loss,
    encode_answer,
    optimizer_step,
    rollout,
    text_vectors,
)

_DEFAULT_BUDGETS = (64, 256, 1024)
_DEFAULT_CODECS = ("compressed_recent", "compressed_diverse", "dictionary_recent")


def _integer(settings: Mapping[str, Any], name: str, default: int, *, minimum: int = 1) -> int:
    value = settings.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return value


def _validate_questions(training: Sequence[StorageQuestion], validation: Sequence[StorageQuestion]) -> None:
    if not training or not validation:
        raise ValueError("training and validation questions are required")
    if any(question.split != "train" for question in training):
        raise ValueError("training must contain only official train questions")
    if any(question.split != "validation" for question in validation):
        raise ValueError("validation must contain only the held-out validation split")
    identities = [question.id for question in (*training, *validation)]
    if len(set(identities)) != len(identities):
        raise ValueError("question identities must be unique")
    if {question.story for question in training} & {question.story for question in validation}:
        raise ValueError("training and validation stories must be disjoint")


def _hash_parameters(parameters: Sequence[torch.nn.Parameter]) -> str:
    digest = hashlib.sha256()
    for parameter in parameters:
        value = parameter.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _base_parameters(reader: Any) -> tuple[torch.nn.Parameter, ...]:
    return tuple(
        parameter
        for name, parameter in reader.model.named_parameters()
        if "lora_" not in name.lower()
    )


def _save_state(path: Path, *, reader: Any, writer: torch.nn.Module | None,
                adapters: Sequence[torch.nn.Parameter]) -> None:
    from safetensors.torch import save_file

    path.mkdir(parents=False, exist_ok=False)
    adapter_ids = {id(parameter) for parameter in adapters}
    adapter_state = {
        f"adapter.{name}": parameter.detach().cpu().contiguous()
        for name, parameter in reader.model.named_parameters()
        if id(parameter) in adapter_ids
    }
    if len(adapter_state) != len(adapters):
        raise ValueError("every adapter parameter must have a stable model name")
    state = dict(adapter_state)
    if writer is not None:
        state.update({
            f"writer.{name}": value.detach().cpu().contiguous()
            for name, value in writer.state_dict().items()
        })
    save_file(state, str(path / "checkpoint.safetensors"))


def _noise_pool(noise: Mapping[str, tuple[str, ...]], split: str) -> tuple[str, ...]:
    values = noise.get(split)
    if values is None and split == "train":
        values = noise.get("training")
    if values is None and split == "validation":
        values = noise.get("dev")
    if values is None:
        raise ValueError(f"noise is missing the {split} pool")
    if not isinstance(values, tuple) or any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"noise[{split!r}] must be a tuple of nonempty strings")
    return values


def _feature_cache(reader: Any, features: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cache: dict[str, torch.Tensor] = {}
    for text, value in features.items():
        if not isinstance(text, str) or not isinstance(value, torch.Tensor):
            raise TypeError("features must map text records to tensors")
        if value.ndim != 2 or value.shape[0] == 0 or value.dtype not in (torch.float32, torch.float64):
            raise ValueError("features must be nonempty floating token matrices")
        if value.requires_grad or not torch.isfinite(value).all():
            raise ValueError("features must be detached and finite")
        cache[text] = value.float().detach()
    return cache


def _require_features(records: Sequence[str], cache: Mapping[str, torch.Tensor]) -> None:
    missing = tuple(dict.fromkeys(record for record in records if record not in cache))
    if missing:
        raise ValueError(f"missing fixed features for {len(missing)} current records")


def _make_store(factory: Callable[..., Any], codec: str, budget: int, dictionary: tuple[str, ...]) -> Any:
    if codec == "dictionary_recent":
        return factory(budget=budget, codec=codec, dictionary=dictionary)
    return factory(budget=budget, codec=codec)


def _store_records(store: Any, records: Sequence[str], budget: int) -> str:
    payload: Any = store.empty() if hasattr(store, "empty") else b""
    for record in records:
        if not hasattr(store, "update"):
            raise TypeError("text store must expose update(payload, record)")
        payload = store.update(payload, record)
    if not isinstance(payload, bytes) or len(payload) > budget:
        raise ValueError("text store returned a payload above its declared budget")
    if not hasattr(store, "decode"):
        raise TypeError("text store must expose decode(payload)")
    decoded = store.decode(payload)
    if not isinstance(decoded, str):
        raise TypeError("decoded text must be a string")
    return decoded


def _noise_level(epoch: int, index: int) -> int:
    if epoch == 0:
        return 0
    if epoch == 1:
        return index % 2
    return index % 3


def _probe_questions(validation: Sequence[StorageQuestion], limit: int) -> tuple[StorageQuestion, ...]:
    selected: list[StorageQuestion] = []
    for task in sorted({question.task for question in validation}):
        task_questions = [question for question in validation if question.task == task]
        selected.extend(task_questions[:limit])
    return tuple(selected)


def _cosine_lambda(step: int, total: int) -> float:
    warmup = max(1, int(total * 0.05))
    if step <= warmup:
        return max(1.0 / warmup, step / warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())


def _answer_batch(reader: Any, memory: torch.nn.Module | None,
                  histories: Sequence[Sequence[str] | str], tokens: Sequence[AnswerTokens],
                  cache: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if not histories or len(histories) != len(tokens):
        raise ValueError("one history is required per answer")
    if memory is None:
        texts = tuple(history if isinstance(history, str) else "\n".join(history)
                      for history in histories)
        memories = tuple(text_vectors(reader, texts))
    else:
        sequences = tuple(history for history in histories if not isinstance(history, str))
        if len(sequences) != len(histories):
            raise TypeError("learned memory histories must be record sequences")
        state = rollout(memory, sequences, cache)
        vectors = memory.memory_vectors(state)
        memories = tuple(vectors[index] for index in range(vectors.shape[0]))
    return answer_loss(reader, tokens, memories)


def _batched_answer_loss(
    reader: Any, memory: torch.nn.Module | None,
    histories: Sequence[Sequence[str] | str], tokens: Sequence[AnswerTokens],
    cache: Mapping[str, torch.Tensor], batch_size: int,
) -> torch.Tensor:
    """Evaluate a probe in bounded batches while weighting each question equally."""
    if not histories or len(histories) != len(tokens):
        raise ValueError("one history is required per answer")
    values = []
    total = len(histories)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        values.append(_answer_batch(reader, memory, histories[start:end], tokens[start:end], cache) * (end - start))
    return torch.stack(values).sum() / total


def _sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def fit(
    reader: Any,
    memory: torch.nn.Module | None,
    features: Mapping[str, torch.Tensor],
    training: tuple[StorageQuestion, ...],
    validation: tuple[StorageQuestion, ...],
    noise: dict[str, tuple[str, ...]],
    dictionary: tuple[str, ...],
    settings: dict[str, Any],
    output: Path,
    seed: int,
    *,
    text_store_factory: Callable[..., Any] = TextStore,
) -> dict[str, Any]:
    """Run the fixed storage comparison and write an auditable result bundle."""
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    _validate_questions(training, validation)
    if not isinstance(dictionary, tuple) or any(not isinstance(value, str) for value in dictionary):
        raise TypeError("dictionary must be a tuple of strings")
    epochs = _integer(settings, "epochs", 3)
    batch_size = _integer(settings, "batch_size", 8)
    probe_limit = _integer(settings, "probe_per_task", 20)
    schedule_seed = _integer(settings, "schedule_seed", seed, minimum=0)
    noise_seed = _integer(settings, "noise_seed", schedule_seed, minimum=0)
    budgets = tuple(settings.get("budgets", _DEFAULT_BUDGETS))
    invalid_budget = any(
        isinstance(value, bool) or not isinstance(value, int) or value < 8 for value in budgets
    )
    if budgets != _DEFAULT_BUDGETS or invalid_budget:
        raise ValueError("budgets must be exactly (64, 256, 1024)")
    if tuple(SUPPORTED_CODECS) != _DEFAULT_CODECS:
        raise ValueError("text-store codec set changed")
    train_noise = _noise_pool(noise, "train")
    validation_noise = _noise_pool(noise, "validation")
    if not train_noise or not validation_noise:
        raise ValueError("both noise pools must contain at least one record")
    if set(train_noise) & set(validation_noise):
        raise ValueError("training and validation noise pools must be disjoint")

    random.seed(seed)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    base_before = _hash_parameters(_base_parameters(reader))
    # This reset must remain immediately before adaptation.  Memory cells may
    # be initialized by the caller, and their shape must not perturb LoRA.
    torch.manual_seed(seed)
    attach_reader_lora(reader, rank=8, checkpointing=False)
    adapters = configure_read_adapter(reader, trainable=True)
    if not adapters:
        raise ValueError("fresh reader Q/V adapters are required")
    if memory is not None:
        memory.train()
        if next(memory.parameters()).device != reader.model.device:
            raise ValueError("memory and reader must share a device")

    writer_parameters = tuple(memory.parameters()) if memory is not None else ()
    parameter_groups = []
    if writer_parameters:
        parameter_groups.append({"params": writer_parameters, "lr": float(settings.get("writer_lr", 0.001)),
                                 "role": "writer"})
    parameter_groups.append({"params": adapters, "lr": float(settings.get("reader_lr", 0.0003)),
                             "role": "reader"})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=float(settings.get("weight_decay", 0.01)))
    total_steps = epochs * ((len(training) + batch_size - 1) // batch_size)
    for group in optimizer.param_groups:
        group["lr"] *= _cosine_lambda(0, total_steps)

    cache = _feature_cache(reader, features)
    probe = _probe_questions(validation, probe_limit)
    encoded_by_text: dict[tuple[str, str], AnswerTokens] = {}
    tokens: dict[str, AnswerTokens] = {}
    for question in (*training, *validation):
        key = (question.question, question.answer)
        if key not in encoded_by_text:
            encoded_by_text[key] = encode_answer(reader, question)
        tokens[question.id] = encoded_by_text[key]
    stores: dict[tuple[str, int], Any] = {}
    if memory is None:
        stores = {
            (codec, budget): _make_store(text_store_factory, codec, budget, dictionary)
            for codec in _DEFAULT_CODECS for budget in budgets
        }
    metrics_path = output / "metrics.jsonl"
    curves_path = output / "epoch_curves.jsonl"
    conditions: dict[str, int] = {}
    noise_levels: dict[str, int] = {"0": 0, "1": 0, "2": 0}
    condition_noise_counts: dict[str, dict[str, int]] = {}
    epoch_curves: list[dict[str, Any]] = []
    _save_state(output / "epoch0", reader=reader, writer=memory, adapters=adapters)

    def write_metric(row: dict[str, Any]) -> None:
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")

    def probe_epoch(epoch: int) -> None:
        for level, label in ((0, "clean"), (2, "level2")):
            target_noise = validation_noise if level else ()
            if memory is not None:
                memory.eval()
            with torch.no_grad():
                if memory is not None:
                    histories = tuple(
                        write_records(question, target_noise, level=level, seed=noise_seed)
                        for question in probe
                    )
                    for history in histories:
                        _require_features(history, cache)
                    value = float(_batched_answer_loss(
                        reader, memory, histories,
                        tuple(tokens[question.id] for question in probe), cache,
                        batch_size,
                    ))
                    rows = ((label, value),)
                else:
                    rows_list: list[tuple[str, float]] = []
                    probe_conditions = tuple(
                        (codec, budget) for codec in _DEFAULT_CODECS for budget in budgets
                    )
                    for condition_index, (codec, budget) in enumerate(probe_conditions):
                        selected = tuple(probe[condition_index::9])
                        if not selected:
                            continue
                        histories = tuple(
                            _store_records(
                                stores[(codec, budget)],
                                write_records(question, target_noise, level=level, seed=noise_seed),
                                budget,
                            )
                            for question in selected
                        )
                        value = float(_batched_answer_loss(
                            reader, None, histories,
                            tuple(tokens[question.id] for question in selected), cache,
                            batch_size,
                        ))
                        rows_list.append((f"{label}:{codec}:{budget}", value))
                    rows = tuple(rows_list)
            if memory is not None:
                memory.train()
            for condition, value in rows:
                row = {"epoch": epoch, "condition": condition, "probe_ce": value}
                epoch_curves.append(row)
                with curves_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")

    probe_epoch(0)
    schedule = random.Random(schedule_seed)
    condition_schedule = random.Random(schedule_seed + 0x5EED)
    step = 0
    answer_values: list[float] = []
    for epoch in range(epochs):
        order = list(training)
        schedule.shuffle(order)
        for batch_start in range(0, len(order), batch_size):
            batch = order[batch_start:batch_start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            histories: list[Sequence[str] | str] = []
            batch_tokens: list[AnswerTokens] = []
            for offset, question in enumerate(batch):
                index = batch_start + offset
                level = _noise_level(epoch, index)
                noise_levels[str(level)] += 1
                history = write_records(question, train_noise if level else (), level=level,
                                        seed=noise_seed)
                condition_index = condition_schedule.randrange(9)
                if memory is None and (epoch * len(order) + index) % 10 == 0:
                    decoded = "\n".join(question.records)
                    condition = "full_text"
                elif memory is None:
                    codec = _DEFAULT_CODECS[condition_index // len(budgets)]
                    budget = budgets[condition_index % len(budgets)]
                    decoded = _store_records(stores[(codec, budget)], history, budget)
                    condition = f"{codec}:{budget}"
                else:
                    decoded = history
                    condition = f"noise_level_{level}"
                if memory is not None:
                    _require_features(decoded, cache)
                conditions[condition] = conditions.get(condition, 0) + 1
                observed_level = 0 if condition == "full_text" else level
                condition_noise_counts.setdefault(condition, {"0": 0, "1": 0, "2": 0})[str(observed_level)] += 1
                histories.append(decoded)
                batch_tokens.append(tokens[question.id])
            step_started = time.perf_counter()
            _sync_cuda(reader.model.device)
            loss = _answer_batch(reader, memory, tuple(histories), tuple(batch_tokens), cache)
            result = optimizer_step(reader, memory, adapters, optimizer, loss)
            _sync_cuda(reader.model.device)
            step_seconds = time.perf_counter() - step_started
            step += 1
            for group in optimizer.param_groups:
                base = float(settings.get("writer_lr", 0.001) if group["role"] == "writer"
                             else settings.get("reader_lr", 0.0003))
                group["lr"] = base * _cosine_lambda(step, total_steps)
            answer_values.append(result["answer_ce"])
            write_metric({"epoch": epoch, "step": step, "question_count": len(batch),
                          "lr": [group["lr"] for group in optimizer.param_groups],
                          "condition_counts": dict(conditions),
                          "condition_noise_counts": condition_noise_counts,
                          "step_seconds": step_seconds,
                          "cuda_max_memory_bytes": (
                              torch.cuda.max_memory_allocated(reader.model.device)
                              if reader.model.device.type == "cuda" else 0
                          ), **result})
            if step == 1 or step % 25 == 0:
                print(
                    f"storage-fit step={step} epoch={epoch} "
                    f"answer_ce={result['answer_ce']:.5f} seconds={step_seconds:.2f}",
                    flush=True,
                )
        probe_epoch(epoch + 1)
        _save_state(output / f"epoch{epoch + 1}", reader=reader, writer=memory, adapters=adapters)
    _save_state(output / "final", reader=reader, writer=memory, adapters=adapters)
    (output / "checkpoint.safetensors").write_bytes(
        (output / "final" / "checkpoint.safetensors").read_bytes()
    )

    base_after = _hash_parameters(_base_parameters(reader))
    if base_before != base_after:
        raise RuntimeError("frozen reader base changed during fit")
    report: dict[str, Any] = {
        "mode": "learned_memory" if memory is not None else "strong_text",
        "seed": seed,
        "schedule_seed": schedule_seed,
        "noise_seed": noise_seed,
        "epochs": epochs,
        "steps": step,
        "question_counts": {"training": len(training), "validation": len(validation), "probe": len(probe)},
        "questions_per_epoch": len(training),
        "supervision": "answers only",
        "official_test_loaded": False,
        "base_hash_before": base_before,
        "base_hash_after": base_after,
        "sum_shared_trainable_param_bytes": sum(parameter.numel() * parameter.element_size()
                                                 for parameter in (*writer_parameters, *adapters)),
        "shared_dictionary_bytes": max(
            (store.shared_dictionary_bytes() for store in stores.values()
             if hasattr(store, "shared_dictionary_bytes")), default=0,
        ),
        "conditions": conditions,
        "condition_noise_counts": condition_noise_counts,
        "noise_levels": noise_levels,
        "budgets": list(budgets),
        "codecs": list(_DEFAULT_CODECS),
        "epoch_curves": epoch_curves,
        "mean_answer_ce": sum(answer_values) / len(answer_values),
        "timings_seconds": {"total": time.perf_counter() - started},
        "output": str(output),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
