"""Score the supervised learned-writer fact experiment.

The scorer keeps the writer and reader boundaries explicit.  A writer state is
read exactly as serialized, without rounding, clipping, sign repair, or a
query-dependent fallback.  Reader correctness is always recomputed from the
literal generated answer, so a mutable metric in an output file cannot change
the reported result.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from tinymem.reader.prompt import reader_exact_match
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.studies.distilled import fit as distilled_fit
from tinymem.studies.distilled import protocol as distilled_protocol
from tinymem.studies.distilled import replication as distilled_replication
from tinymem.studies.distilled import training as distilled_training
from tinymem.studies.oracle import fit as oracle_fit
from tinymem.studies.delta.data import ROOM_PAIRS, Episode, replay
from tinymem.studies.delta.encoding import (
    _cached_feature,
    _queries,
    build_feature_cache,
)
from tinymem.studies.delta.evaluation import StateRecord, collect_states
from tinymem.studies.delta.fit import load_state_records, save_state_records
from tinymem.studies.artifacts import file_hash, frozen_base_hash, write_json
from tinymem.studies.delta.readout import read_answer
from tinymem.studies.oracle.state import _state_for_texts, oracle_records

STATE_WIDTH = 32
STATE_SLOTS = 2
STATE_BYTES = STATE_SLOTS * STATE_WIDTH * 4 + STATE_SLOTS
FACT_CELL_INDICES = (0, 8, 16, 24)
ENDPOINTS = (8, 9, 16)
READ_MODES = ("real", "oracle", "zero", "donor")
READ_FIELDS = {"prediction", "generated_ids", "input_positions", "memory_positions",
               "native_envelope_tokens", "correct"}


def _record_metadata(record: StateRecord) -> dict[str, Any]:
    return {key: value for key, value in asdict(record).items()
            if key not in {"values", "valid"}}


def _record_identity(record: StateRecord) -> tuple[Any, ...]:
    return tuple(getattr(record, name) for name in
                 ("episode_id", "prefix_id", "split", "wording", "condition", "target", "after_write"))


def _owned_state(record: StateRecord) -> LatentSlotState:
    """Validate and copy one CPU state without changing its values."""

    values, valid = record.values, record.valid
    if (not isinstance(values, torch.Tensor) or not isinstance(valid, torch.Tensor)
            or values.shape != (1, STATE_SLOTS, STATE_WIDTH)
            or valid.shape != (1, STATE_SLOTS)
            or values.dtype != torch.float32 or valid.dtype != torch.bool
            or values.device.type != "cpu" or valid.device.type != "cpu"
            or values.requires_grad or valid.requires_grad
            or values.grad_fn is not None or valid.grad_fn is not None
            or not values.is_contiguous() or not valid.is_contiguous()
            or not bool(torch.isfinite(values).all())):
        raise ValueError("states must own finite CPU FP32 [1, 2, 32] values and boolean validity")
    if values.untyped_storage().nbytes() + valid.untyped_storage().nbytes() != STATE_BYTES:
        raise ValueError("state does not have the declared 258 persistent bytes")
    if not bool(valid.all()):
        raise ValueError("endpoint state must have both slots valid")
    return LatentSlotState(values.detach().clone().contiguous(), valid.detach().clone().contiguous())


def _truth(episode: Episode, endpoint: int) -> tuple[int | None, ...]:
    statements = (*episode.prefix, *episode.tail)
    if type(endpoint) is not int or not 1 <= endpoint <= len(statements):
        raise ValueError("endpoint is not declared for this episode")
    return tuple(replay(tuple(statement.text for statement in statements[:endpoint])))


def _direct_bits(record: StateRecord) -> list[int | None]:
    """Read fact signs directly from raw state cells.

    Exact zero is represented as ``None``.  No epsilon threshold or projection
    is applied; callers may count an undefined sign as incorrect while keeping
    the distinction visible in the saved diagnostics.
    """

    state = _owned_state(record)
    cells = state.values[0, 0, list(FACT_CELL_INDICES)]
    return [None if float(value) == 0.0 else int(float(value) > 0.0) for value in cells]


def _direct_bit_metrics(record: StateRecord, truth: Sequence[int | None]) -> dict[str, Any]:
    bits = _direct_bits(record)
    if len(truth) != 4 or any(value is not None and (type(value) is not int or value not in (0, 1))
                               for value in truth):
        raise ValueError("direct bit metrics require four binary values or None")
    known = [(bit, value) for bit, value in zip(bits, truth, strict=True) if value is not None]
    return {"direct_bits": bits, "truth": [None if value is None else int(value) for value in truth],
            "known_fact_correct": sum(bit is not None and bit == value for bit, value in known),
            "known_fact_count": len(known)}


def _state_sse(left: StateRecord, right: StateRecord) -> float:
    _owned_state(left)
    _owned_state(right)
    if left.values.shape != right.values.shape:
        raise ValueError("state shapes differ")
    value = torch.square(left.values - right.values).sum()
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("state SSE is nonfinite")
    return result


def _off_fact_sse(left: StateRecord, right: StateRecord) -> float:
    _owned_state(left)
    _owned_state(right)
    mask = torch.ones_like(left.values, dtype=torch.bool)
    mask[0, 0, list(FACT_CELL_INDICES)] = False
    result = float(torch.square(left.values - right.values).masked_select(mask).sum())
    if not np.isfinite(result):
        raise ValueError("off-fact state SSE is nonfinite")
    return result


def _read_metadata(value: object, answer: str, category: str) -> dict[str, Any]:
    """Validate generated-read metadata and recompute exact correctness."""

    if not isinstance(value, dict) or set(value) not in (READ_FIELDS, READ_FIELDS - {"correct"}):
        raise ValueError("distilled read metadata fields differ")
    prediction = value["prediction"]
    generated = value["generated_ids"]
    if type(prediction) is not str or type(generated) is not list:
        raise ValueError("distilled read metadata has invalid prediction fields")
    if (not 1 <= len(generated) <= 8
            or any(type(token) is not int or token < 0 for token in generated)):
        raise ValueError("distilled generated token count differs from the fixed read contract")
    for name in ("input_positions", "memory_positions", "native_envelope_tokens"):
        if type(value[name]) is not int or value[name] < 0:
            raise ValueError("distilled read metadata has invalid position counts")
    if (value["memory_positions"] != 2 or value["native_envelope_tokens"] <= 0
            or value["input_positions"] != value["memory_positions"] + value["native_envelope_tokens"]):
        raise ValueError("distilled read positions do not preserve the native envelope")
    correct = reader_exact_match(prediction, answer, category)
    if "correct" in value and (type(value["correct"]) is not bool or value["correct"] != correct):
        raise ValueError("stored distilled correctness disagrees with literal answer matching")
    return {**value, "correct": correct}


def _state_key(state: LatentSlotState, before_ids: tuple[int, ...], after_ids: tuple[int, ...]) -> tuple[Any, ...]:
    return (state.values.detach().cpu().contiguous().numpy().tobytes(),
            state.valid.detach().cpu().contiguous().numpy().tobytes(), before_ids, after_ids)


def _tensor_manifest_hash(tensors: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)],
                                 separators=(",", ":")).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _expected_donors(episodes: Sequence[Episode], donor_episodes: Sequence[Episode]) -> dict[tuple[str, str], Episode]:
    grouped: dict[str, list[Episode]] = defaultdict(list)
    for episode in donor_episodes:
        grouped[episode.wording].append(episode)
    result: dict[tuple[str, str], Episode] = {}
    for wording, values in grouped.items():
        ordered = sorted(values, key=lambda episode: episode.prefix_id)
        if len(ordered) < 2:
            raise ValueError("donor control requires at least two no-write prefixes per wording")
        for position, episode in enumerate(ordered):
            result[episode.prefix_id, wording] = ordered[(position + 1) % len(ordered)]
    return {(episode.prefix_id, episode.wording): result[(episode.prefix_id, episode.wording)]
            for episode in episodes}


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q, method="linear"))


def _paired_bootstrap(rows: Sequence[dict[str, Any]], control: str, *, condition: str,
                      scope: str, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Bootstrap prefix gaps after averaging wordings within each prefix."""

    if control not in {"oracle", "zero", "donor"}:
        raise ValueError("paired bootstrap control must be oracle, zero, or donor")
    selected = [row for row in rows if row["after_write"] == 16
                and row["condition"] == condition and row["scope"] == scope]
    if not selected:
        return {"status": "absent", "control": control}
    by_seed_prefix_wording: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in selected:
        key = (str(row["writer_seed"]), str(row["prefix_id"]), str(row["wording"]))
        by_seed_prefix_wording[key].append(
            float(row["reads"]["real"]["correct"]) - float(row["reads"][control]["correct"]))
    by_seed_prefix: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for (seed, prefix, wording), values in by_seed_prefix_wording.items():
        if wording in by_seed_prefix[seed, prefix]:
            raise ValueError("paired bootstrap contains duplicate wording rows")
        by_seed_prefix[seed, prefix][wording] = fmean(values)
    seeds = sorted({seed for seed, _ in by_seed_prefix})
    prefixes = sorted({prefix for _, prefix in by_seed_prefix})
    wordings = set().union(*(by_seed_prefix[key] for key in by_seed_prefix))
    if not wordings or any(set(by_seed_prefix[seed, prefix]) != wordings
                           for seed in seeds for prefix in prefixes):
        raise ValueError("paired bootstrap requires complete seed, prefix, and wording coverage")
    prefix_gaps = {(seed, prefix): fmean(by_seed_prefix[seed, prefix].values())
                   for seed in seeds for prefix in prefixes}
    per_seed = {seed: 100 * fmean(prefix_gaps[seed, prefix] for prefix in prefixes)
                for seed in seeds}
    samples = int(settings.get("bootstrap_samples", 10000))
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    rng = random.Random(int(settings.get("bootstrap_seed", 8237)))
    draws = []
    for _ in range(samples):
        sampled = [rng.choice(prefixes) for _ in prefixes]
        draws.append(100 * fmean(fmean(prefix_gaps[seed, prefix] for prefix in sampled)
                                 for seed in seeds))
    return {"status": "estimated", "control": control, "per_seed_gap_pp": per_seed,
            "mean_gap_pp": fmean(per_seed.values()),
            "interval_pp": [_percentile(draws, 0.5), _percentile(draws, 99.5)],
            "confidence": 0.99, "unit": "paired_prefixes_conditional_on_observed_seeds",
            "wordings_averaged_within_prefix": sorted(wordings), "seeds": seeds,
            "prefixes": len(prefixes), "resamples": samples,
            "bootstrap_seed": int(settings.get("bootstrap_seed", 8237))}


def _selected_episodes(episodes: Sequence[Episode], settings: Mapping[str, Any]) -> tuple[Episode, ...]:
    limit = settings.get("evaluation_prefix_limit")
    if limit is None:
        return tuple(episodes)
    if type(limit) is not int or limit <= 0:
        raise ValueError("evaluation_prefix_limit must be positive or null")
    prefix_ids = sorted({episode.prefix_id for episode in episodes})[:limit]
    return tuple(episode for episode in episodes if episode.prefix_id in prefix_ids)


def _feature_texts(episodes: Sequence[Episode]) -> list[str]:
    return sorted({statement.text for episode in episodes
                   for statement in (*episode.prefix, *episode.tail)})


def _save_evaluation_features(directory: Path, features: Mapping[str, torch.Tensor],
                              episodes: Sequence[Episode], reader_width: int) -> None:
    texts = _feature_texts(episodes)
    if set(features) != set(texts):
        raise ValueError("evaluation feature inventory differs from the declared episodes")
    tensors = {str(index): _cached_feature(features, text, reader_width)
               for index, text in enumerate(texts)}
    save_file(tensors, str(directory / "features.safetensors"))
    write_json(directory / "feature_texts.json", texts)


def _load_evaluation_features(directory: Path, episodes: Sequence[Episode]) -> dict[str, torch.Tensor]:
    texts = json.loads((directory / "feature_texts.json").read_text())
    expected = _feature_texts(episodes)
    if texts != expected:
        raise ValueError("stored evaluation feature texts differ from the declared episodes")
    tensors = load_file(str(directory / "features.safetensors"), device="cpu")
    if set(tensors) != {str(index) for index in range(len(texts))}:
        raise ValueError("stored evaluation feature tensor inventory differs")
    features = {text: tensors[str(index)].detach().clone().contiguous()
                for index, text in enumerate(texts)}
    reader_widths = {feature.shape[1] for feature in features.values()
                     if isinstance(feature, torch.Tensor) and feature.ndim == 2}
    if len(reader_widths) != 1:
        raise ValueError("stored evaluation features have inconsistent reader widths")
    for text in texts:
        _cached_feature(features, text, next(iter(reader_widths)))
    return features


def _assert_diagnostic_rows_equal(expected: Sequence[Mapping[str, Any]],
                                  actual: Sequence[Mapping[str, Any]]) -> None:
    if len(expected) != len(actual):
        raise ValueError("recomputed diagnostics changed endpoint coverage")
    metadata = ("episode_id", "prefix_id", "wording", "condition", "target", "after_write",
                "direct_bits", "truth", "known_fact_correct", "known_fact_count")
    metrics = ("state_sse", "off_fact_sse", "one_step_state_sse", "one_step_off_fact_sse")
    for before, after in zip(expected, actual, strict=True):
        if any(before[name] != after[name] for name in metadata):
            raise ValueError("recomputed diagnostics changed metadata or direct bits")
        if any(abs(float(before[name]) - float(after[name])) > 1e-6 for name in metrics):
            raise ValueError("recomputed diagnostics changed a numeric metric")


def _record_map(records: Sequence[StateRecord], episodes: Sequence[Episode]) -> dict[tuple[Any, ...], StateRecord]:
    episodes_by_id = {episode.id: episode for episode in episodes}
    expected: dict[tuple[Any, ...], tuple[str, str, str, int | None]] = {}
    for episode in episodes:
        endpoints = (8, 9, 16) if episode.tail else (8,)
        for endpoint in endpoints:
            expected[(episode.id, episode.prefix_id, episode.split, episode.wording,
                      episode.condition, episode.target, endpoint)] = (
                          episode.id, episode.prefix_id, episode.split, episode.wording)
    result: dict[tuple[Any, ...], StateRecord] = {}
    for record in records:
        _owned_state(record)
        key = _record_identity(record)
        if key in result:
            raise ValueError("state records contain duplicate endpoint identities")
        if key not in expected:
            raise ValueError("state records contain an undeclared endpoint")
        episode = episodes_by_id[record.episode_id]
        if tuple(record.truth) != _truth(episode, record.after_write):
            raise ValueError("state record truth disagrees with literal episode replay")
        result[key] = record
    if set(result) != set(expected):
        raise ValueError("state records omit or add endpoint cases")
    return result


def _oracle_target_record(episode: Episode, endpoint: int) -> StateRecord:
    statements = (*episode.prefix, *episode.tail)
    target = _state_for_texts(tuple(statement.text for statement in statements[:endpoint]))
    return StateRecord(
        episode.id, episode.prefix_id, episode.split, episode.wording,
        episode.condition, episode.target, endpoint,
        target.values.detach().clone().contiguous(), target.valid.detach().clone().contiguous(),
        _truth(episode, endpoint),
        float(torch.linalg.vector_norm(target.values)), None, None,
    )


def _validate_diagnostics(
    rows: object,
    episodes: Sequence[Episode],
    endpoint_records: Mapping[tuple[Any, ...], StateRecord],
) -> list[dict[str, Any]]:
    """Validate every-write diagnostics, using persisted states at read endpoints."""

    if not isinstance(rows, list) or not rows:
        raise ValueError("trajectory diagnostics must be a nonempty list")
    expected: dict[tuple[Any, ...], tuple[Episode, int]] = {}
    episodes_by_id = {episode.id: episode for episode in episodes}
    for episode in episodes:
        writes = len(episode.prefix) + len(episode.tail)
        for endpoint in range(1, writes + 1):
            expected[(episode.id, episode.prefix_id, episode.split, episode.wording,
                      episode.condition, episode.target, endpoint)] = (episode, endpoint)
    seen: set[tuple[Any, ...]] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("trajectory diagnostic rows must be objects")
        required = {"episode_id", "prefix_id", "wording", "condition", "target", "after_write",
                    "state_sse", "off_fact_sse", "direct_bits", "truth", "known_fact_correct",
                    "known_fact_count", "one_step_state_sse", "one_step_off_fact_sse"}
        if set(row) != required:
            raise ValueError("trajectory diagnostic schema differs")
        if (not isinstance(row["direct_bits"], list) or len(row["direct_bits"]) != 4
                or any(bit is not None and bit not in (0, 1) for bit in row["direct_bits"])
                or not isinstance(row["truth"], list) or len(row["truth"]) != 4
                or any(value is not None and value not in (0, 1) for value in row["truth"])):
            raise ValueError("trajectory diagnostic bits or truth are invalid")
        if type(row["episode_id"]) is not str:
            raise ValueError("trajectory diagnostic episode ID is invalid")
        episode = episodes_by_id.get(row["episode_id"])
        if episode is None:
            raise ValueError("trajectory diagnostic refers to an undeclared episode")
        key = (row["episode_id"], row["prefix_id"],
               episode.split,
               row["wording"],
               row["condition"], row["target"], row["after_write"])
        if key in seen:
            raise ValueError("trajectory diagnostics contain duplicate endpoint rows")
        if key not in expected:
            raise ValueError("trajectory diagnostic refers to an undeclared endpoint")
        seen.add(key)
        episode, endpoint = expected[key]
        truth = _truth(episode, endpoint)
        target = _oracle_target_record(episode, endpoint)
        raw = endpoint_records.get(key)
        state = raw if raw is not None else target
        for name in ("state_sse", "off_fact_sse", "one_step_state_sse", "one_step_off_fact_sse"):
            if type(row[name]) not in (int, float) or not np.isfinite(row[name]) or row[name] < 0:
                raise ValueError("trajectory diagnostic contains a nonfinite metric")
        metrics = _direct_bit_metrics(state, truth) if raw is not None else {
            "direct_bits": row["direct_bits"],
            "truth": [None if value is None else int(value) for value in truth],
            "known_fact_correct": sum(
                bit is not None and bit == expected
                for bit, expected in zip(row["direct_bits"], truth, strict=True)
                if expected is not None
            ),
            "known_fact_count": sum(value is not None for value in truth),
        }
        if (row["direct_bits"] != metrics["direct_bits"]
                or row["truth"] != metrics["truth"]
                or row["known_fact_correct"] != metrics["known_fact_correct"]
                or row["known_fact_count"] != metrics["known_fact_count"]):
            raise ValueError("trajectory diagnostic direct bits disagree with saved states")
        if raw is not None:
            if abs(row["state_sse"] - _state_sse(raw, target)) > 1e-6:
                raise ValueError("trajectory diagnostic state SSE disagrees with saved states")
            if abs(row["off_fact_sse"] - _off_fact_sse(raw, target)) > 1e-6:
                raise ValueError("trajectory diagnostic off-fact SSE disagrees with saved states")
        if row["off_fact_sse"] > row["state_sse"] + 1e-6:
            raise ValueError("trajectory off-fact SSE exceeds total state SSE")
        if row["one_step_off_fact_sse"] > row["one_step_state_sse"] + 1e-6:
            raise ValueError("one-step off-fact SSE exceeds total state SSE")
        result.append(row)
    if set(seen) != set(expected):
        raise ValueError("trajectory diagnostics omit or add endpoint rows")
    return result


def _assert_state_roundtrip(original: Sequence[StateRecord], loaded: Sequence[StateRecord]) -> None:
    if len(original) != len(loaded):
        raise ValueError("state serialization changed record coverage")
    for before, after in zip(original, loaded, strict=True):
        for field in before.__dataclass_fields__:
            left, right = getattr(before, field), getattr(after, field)
            equal = torch.equal(left, right) if isinstance(left, torch.Tensor) else left == right
            if not equal:
                raise ValueError("state serialization changed record: " + field)


def _parent_protocol(study: Path, protocol: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    parent = study / "parent"
    path = parent / "protocol.json"
    if not path.is_file():
        raise ValueError("copied parent protocol is missing")
    payload = json.loads(path.read_text())
    expected = protocol["parent_protocol_sha256"]
    if expected != file_hash(path):
        raise ValueError("parent protocol provenance differs")
    return parent, payload


def score_cell(reader, study: Path, protocol: dict, dataset, index: int) -> dict[str, Any]:
    """Score one learned-writer cell and seal every raw evaluation artifact."""

    study = Path(study)
    distilled_protocol.require_training_seal(study, protocol)
    evaluation_cells = distilled_replication.evaluation_cells(protocol)
    if type(index) is not int or not 0 <= index < len(evaluation_cells):
        raise ValueError("cell index is outside the declaration")
    settings = protocol["settings"]
    cell = evaluation_cells[index]
    directory = study / "evaluation" / str(index)
    directory.mkdir(parents=True, exist_ok=False)
    episodes = _selected_episodes(dataset.test, settings)
    donor_episodes = tuple(episode for episode in dataset.test if episode.condition == "no_write")
    all_episodes = tuple({episode.id: episode for episode in (*episodes, *donor_episodes)}.values())
    donors = _expected_donors(episodes, donor_episodes)

    # Build current-test features on the unadapted base before loading the old
    # bridge and LoRA reader checkpoint.
    base_before = frozen_base_hash(reader)
    feature_cache = build_feature_cache(reader, all_episodes)
    parent, parent_protocol = _parent_protocol(study, protocol)
    parent_index = cell.get("parent_index")
    if type(parent_index) is not int or not 0 <= parent_index < len(parent_protocol["cells"]):
        raise ValueError("cell parent_index is invalid")
    parent_cell = parent_protocol["cells"][parent_index]
    parent_training = parent / "training" / str(parent_index)
    parent_report = json.loads((parent_training / "report.json").read_text())
    if base_before != parent_report.get("unadapted_base_sha256"):
        raise ValueError("evaluation base does not match the frozen parent training base")
    bridge = oracle_fit.load_trained(reader, parent_protocol, parent_cell,
                                  parent_training / "checkpoint.safetensors")
    base_after = frozen_base_hash(reader)
    if base_after != parent_report.get("base_after_sha256"):
        raise ValueError("loaded parent reader base differs from the frozen parent endpoint")
    if oracle_fit.execution_record(reader) != parent_report.get("runtime"):
        raise ValueError("reader execution runtime differs from the frozen parent endpoint")
    parameters_before = oracle_fit.checkpoint_tensors(reader, bridge)
    writer = distilled_fit.load_writer(study, protocol, distilled_replication.training_index(cell))
    if next(writer.parameters()).device.type != "cpu":
        raise ValueError("learned writer evaluation must run on CPU")
    _save_evaluation_features(directory, feature_cache, all_episodes, reader.model.config.hidden_size)
    restored_features = _load_evaluation_features(directory, all_episodes)
    if any(not torch.equal(feature_cache[text], restored_features[text]) for text in feature_cache):
        raise ValueError("evaluation feature serialization changed values")
    feature_cache = restored_features

    learned_records = tuple(collect_states(writer, all_episodes, feature_cache))
    oracle_all = tuple(oracle_records(all_episodes))
    test_ids, donor_ids = {episode.id for episode in episodes}, {episode.id for episode in donor_episodes}
    learned_test = tuple(record for record in learned_records if record.episode_id in test_ids)
    learned_donor = tuple(record for record in learned_records if record.episode_id in donor_ids)
    oracle_test = tuple(record for record in oracle_all if record.episode_id in test_ids)
    oracle_donor = tuple(record for record in oracle_all if record.episode_id in donor_ids)
    all_by_id = {episode.id: episode for episode in all_episodes}
    _record_map(learned_records, all_episodes)
    _record_map(oracle_all, all_episodes)
    learned_test_map = _record_map(learned_test, episodes)
    oracle_test_map = _record_map(oracle_test, episodes)
    _record_map(learned_donor, donor_episodes)
    _record_map(oracle_donor, donor_episodes)
    save_state_records(learned_test, directory / "states.safetensors")
    save_state_records(learned_donor, directory / "donor_states.safetensors")
    save_state_records(oracle_test, directory / "oracle_states.safetensors")
    save_state_records(oracle_donor, directory / "oracle_donor_states.safetensors")
    loaded = load_state_records(directory / "states.safetensors")
    _assert_state_roundtrip(learned_test, loaded)
    learned_test = loaded
    loaded = load_state_records(directory / "donor_states.safetensors")
    _assert_state_roundtrip(learned_donor, loaded)
    learned_donor = loaded
    loaded = load_state_records(directory / "oracle_states.safetensors")
    _assert_state_roundtrip(oracle_test, loaded)
    oracle_test = loaded
    loaded = load_state_records(directory / "oracle_donor_states.safetensors")
    _assert_state_roundtrip(oracle_donor, loaded)
    oracle_donor = loaded
    learned_test_map = _record_map(learned_test, episodes)
    oracle_test_map = _record_map(oracle_test, episodes)
    _record_map(learned_donor, donor_episodes)
    _record_map(oracle_donor, donor_episodes)

    diagnostics = distilled_training.trajectory_diagnostics(writer, episodes, feature_cache)
    _validate_diagnostics(diagnostics, episodes, learned_test_map)
    write_json(directory / "diagnostics.json", diagnostics)
    predictions: dict[tuple[Any, ...], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    donor_by_key = {(record.prefix_id, record.wording, record.after_write): record for record in learned_donor}
    with (directory / "predictions.jsonl").open("x") as handle:
        for record in learned_test:
            episode = all_by_id[record.episode_id]
            endpoint = record.after_write
            current = (*episode.prefix, *episode.tail[:max(0, endpoint - 8)])
            before_ids, queries = _queries(reader, episode, current, endpoint)
            oracle_record = oracle_test_map[_record_identity(record)]
            donor_episode = donors[(episode.prefix_id, episode.wording)]
            donor = donor_by_key[(donor_episode.prefix_id, donor_episode.wording, 8)]
            memories = {"real": _owned_state(record), "oracle": _owned_state(oracle_record),
                        "zero": LatentSlotState(torch.zeros_like(record.values),
                                                torch.ones_like(record.valid)),
                        "donor": _owned_state(donor)}
            truth = _truth(episode, endpoint)
            direct = _direct_bit_metrics(record, truth)
            for entity, query in enumerate(queries):
                answer = ROOM_PAIRS[entity][truth[entity]]
                outputs: dict[str, dict[str, Any]] = {}
                for mode in READ_MODES:
                    memory = memories[mode]
                    key = _state_key(memory, before_ids, query.after_ids)
                    if key not in predictions:
                        generated = read_answer(reader, bridge,
                                                LatentSlotState(memory.values.to(reader.model.device),
                                                                memory.valid.to(reader.model.device)),
                                                torch.tensor(before_ids, device=reader.model.device),
                                                torch.tensor(query.after_ids, device=reader.model.device),
                                                max_new_tokens=settings["max_new_tokens"])
                        predictions[key] = dict(generated)
                    outputs[mode] = _read_metadata({**predictions[key]}, answer, query.category)
                donor_truth = _truth(donor_episode, 8)[entity]
                scope = ("all" if episode.target is None else
                         "target" if episode.target == entity else "unspoken")
                row = {**_record_metadata(record), "writer_seed": str(cell["seed"]),
                       "key": f"{episode.id}/write{endpoint}/entity{entity}", "entity": entity,
                       "scope": scope, "truth_bit": truth[entity], "answer": answer,
                       "direct_bits": direct["direct_bits"], "known_fact_correct": direct["known_fact_correct"],
                       "known_fact_count": direct["known_fact_count"],
                       "direct_bit": direct["direct_bits"][entity],
                       "direct_bit_correct": (direct["direct_bits"][entity] is not None
                                              and direct["direct_bits"][entity] == truth[entity]),
                       "state_sse": _state_sse(record, oracle_record),
                       "off_fact_sse": _off_fact_sse(record, oracle_record), "reads": outputs,
                       "donor_prefix": donor_episode.prefix_id, "donor_truth": list(_truth(donor_episode, 8)),
                       "donor_truth_agrees": donor_truth == truth[entity],
                       "donor_answer": ROOM_PAIRS[entity][donor_truth]}
                if "replication" in settings:
                    row["reader_seed"] = str(cell["reader_seed"])
                rows.append(row)
                handle.write(json.dumps(row, allow_nan=False) + "\n")
        handle.flush()
    if frozen_base_hash(reader) != base_after:
        raise ValueError("reader base changed during learned-writer evaluation")
    parameters_after = oracle_fit.checkpoint_tensors(reader, bridge)
    if parameters_before.keys() != parameters_after.keys() or any(
            not torch.equal(parameters_before[name], parameters_after[name]) for name in parameters_before):
        raise ValueError("parent reader bridge or adapter changed during evaluation")
    report = {"schema": "distilled_fact_cell_v1", "cell": cell, "rows": len(rows),
              "unique_generated_reads": len(predictions), "state_records": len(learned_test),
              "state_bytes": STATE_BYTES, "runtime": oracle_fit.execution_record(reader),
              "writer_runtime": distilled_fit._runtime(),
              "unadapted_base_sha256": base_before,
              "adapted_base_sha256_before": base_after,
              "adapted_base_sha256_after": frozen_base_hash(reader),
              "parent_parameters_sha256_before": _tensor_manifest_hash(parameters_before),
              "parent_parameters_sha256_after": _tensor_manifest_hash(parameters_after),
              "parent_parameters_unchanged": True, "raw_state_read": True,
              "diagnostics": "diagnostics.json", "modes": list(READ_MODES),
              "interpretation": "supervised learned-writer states read through frozen successful parent readers"}
    write_json(directory / "report.json", report)
    identity = {**distilled_protocol.evaluation_identity(study, protocol, index),
                "training_seal_sha256": file_hash(study / "training_sealed.json")}
    distilled_protocol.seal_directory(directory, identity)
    return report


def _validate_row(row: object, episodes: Mapping[str, Episode], writer_seed: str) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise TypeError("distilled evaluation rows must be objects")
    required = {"key", "episode_id", "prefix_id", "wording", "condition", "target", "after_write",
                "entity", "scope", "truth_bit", "answer", "direct_bits", "known_fact_correct",
                "known_fact_count", "direct_bit", "direct_bit_correct", "state_sse", "off_fact_sse",
                "reads", "donor_prefix",
                "donor_truth", "donor_truth_agrees", "donor_answer", "writer_seed"}
    if not required.issubset(row):
        raise ValueError("distilled evaluation row is missing required fields")
    if str(row["writer_seed"]) != writer_seed:
        raise ValueError("distilled row writer seed differs from its sealed cell")
    episode = episodes.get(row["episode_id"])
    if episode is None or row["prefix_id"] != episode.prefix_id or row["wording"] != episode.wording:
        raise ValueError("distilled row metadata disagrees with dataset")
    if row["condition"] != episode.condition or row["target"] != episode.target:
        raise ValueError("distilled row condition or target disagrees with dataset")
    endpoint, entity = row["after_write"], row["entity"]
    if type(endpoint) is not int or endpoint not in ENDPOINTS or (endpoint != 8 and not episode.tail):
        raise ValueError("distilled row endpoint is invalid")
    if type(entity) is not int or entity not in range(4):
        raise ValueError("distilled row entity is invalid")
    truth = _truth(episode, endpoint)
    if row["truth_bit"] != truth[entity] or row["answer"] != ROOM_PAIRS[entity][truth[entity]]:
        raise ValueError("distilled row truth or answer disagrees with replay")
    expected_scope = "all" if episode.target is None else "target" if episode.target == entity else "unspoken"
    if row["scope"] != expected_scope:
        raise ValueError("distilled row scope disagrees with dataset")
    if row["direct_bits"] != list(row["direct_bits"]) or len(row["direct_bits"]) != 4:
        raise ValueError("distilled direct bits have invalid shape")
    if any(value is not None and value not in (0, 1) for value in row["direct_bits"]):
        raise ValueError("distilled direct bits are not binary or undefined")
    if type(row["known_fact_correct"]) is not int or type(row["known_fact_count"]) is not int:
        raise ValueError("distilled direct bit counts have invalid types")
    if row["known_fact_count"] != 4 or row["known_fact_correct"] != sum(
            bit is not None and bit == expected for bit, expected in zip(row["direct_bits"], truth, strict=True)):
        raise ValueError("distilled direct bit counts disagree with literal state signs")
    if (row["direct_bit"] != row["direct_bits"][entity]
            or type(row["direct_bit_correct"]) is not bool
            or row["direct_bit_correct"] != (row["direct_bit"] is not None
                                               and row["direct_bit"] == truth[entity])):
        raise ValueError("distilled entity direct bit disagrees with literal state signs")
    for name in ("state_sse", "off_fact_sse"):
        if type(row[name]) not in (int, float) or not np.isfinite(row[name]) or row[name] < 0:
            raise ValueError("distilled state metrics are invalid")
    if not isinstance(row["reads"], dict) or set(row["reads"]) != set(READ_MODES):
        raise ValueError("distilled read controls differ")
    for read in row["reads"].values():
        _read_metadata(read, row["answer"], "update_known")
    expected_key = f'{row["episode_id"]}/write{endpoint}/entity{entity}'
    if row["key"] != expected_key:
        raise ValueError("distilled row key differs")
    if not isinstance(row["donor_prefix"], str) or not isinstance(row["donor_truth"], list):
        raise TypeError("distilled donor metadata is invalid")
    if len(row["donor_truth"]) != 4 or any(value not in (0, 1) for value in row["donor_truth"]):
        raise ValueError("distilled donor truth is invalid")
    if type(row["donor_truth_agrees"]) is not bool or not isinstance(row["donor_answer"], str):
        raise ValueError("distilled donor agreement metadata is invalid")
    return row


def _validate_donor_rows(rows: Sequence[dict[str, Any]], episodes: Sequence[Episode], donor_episodes: Sequence[Episode]) -> None:
    expected = _expected_donors(episodes, donor_episodes)
    by_id = {episode.id: episode for episode in episodes}
    for row in rows:
        episode = by_id[row["episode_id"]]
        donor = expected[(episode.prefix_id, episode.wording)]
        donor_truth = _truth(donor, 8)
        if row["donor_prefix"] != donor.prefix_id or row["donor_truth"] != list(donor_truth):
            raise ValueError("distilled donor differs from the declared cyclic donor")
        entity = row["entity"]
        if row["donor_truth_agrees"] != (donor_truth[entity] == row["truth_bit"]):
            raise ValueError("distilled donor truth agreement disagrees with replay")
        if row["donor_answer"] != ROOM_PAIRS[entity][donor_truth[entity]]:
            raise ValueError("distilled donor answer disagrees with replay")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                raise ValueError("blank distilled evaluation row")
            rows.append(json.loads(line))
    return rows


def _trajectory_summary(rows: Sequence[dict[str, Any]], seed: str) -> dict[str, Any]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["wording"]), str(row["condition"]), int(row["after_write"]))].append(row)
    result: dict[str, Any] = {}
    for (wording, condition, endpoint), values in sorted(grouped.items()):
        known = sum(int(row["known_fact_count"]) for row in values)
        result[f"{wording}/{condition}/write{endpoint}"] = {
            "seed": str(seed),
            "n": len(values),
            "state_sse": fmean(float(row["state_sse"]) for row in values),
            "off_fact_sse": fmean(float(row["off_fact_sse"]) for row in values),
            "one_step_state_sse": fmean(float(row["one_step_state_sse"]) for row in values),
            "one_step_off_fact_sse": fmean(float(row["one_step_off_fact_sse"]) for row in values),
            "direct_known_fact_accuracy": (
                sum(int(row["known_fact_correct"]) for row in values) / known if known else 0.0
            ),
            "known_fact_count": known,
        }
    return result


def _parent_provenance(study: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    parent, parent_protocol = _parent_protocol(study, protocol)
    if not parent_protocol.get("cells"):
        raise ValueError("parent protocol contains no reader cells")
    for cell in distilled_replication.evaluation_cells(protocol):
        index = cell.get("parent_index")
        if type(index) is not int or not 0 <= index < len(parent_protocol["cells"]):
            raise ValueError("new cell refers to an invalid parent reader")
        checkpoint = parent / "training" / str(index) / "checkpoint.safetensors"
        if not checkpoint.is_file():
            raise ValueError("parent reader checkpoint is missing")
    return {"path": "parent", "protocol_sha256": file_hash(parent / "protocol.json"),
            "files_sha256": protocol["parent_files_sha256"]}
