"""Deterministic answer-evaluation streams and state catalog."""

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy

from tinymem.research.independent_fact_data import ENTITIES, ROOM_PAIRS


_TRAINING_ORDERS = {
    (0, 1, 3, 2),
    (1, 2, 0, 3),
    (2, 3, 1, 0),
    (3, 0, 2, 1),
}
_PROGRAMS = {
    "forward_interleaved": {
        "targets": (0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3, 0, 1, 2, 3),
        "first_actions": ("C", "R", "C", "R", "C", "R", "C", "R"),
        "tail_actions": ("R",) * 8,
    },
    "reverse_interleaved": {
        "targets": (3, 3, 2, 2, 1, 1, 0, 0, 3, 2, 1, 0, 3, 2, 1, 0),
        "first_actions": ("R", "C", "R", "C", "R", "C", "R", "C"),
        "tail_actions": ("R",) * 8,
    },
}


def _event_text(fact: int, bit: int) -> str:
    return f"{ENTITIES[fact]} moved to the {ROOM_PAIRS[fact][bit]}."


def _make_event(step: int, before: int, fact: int, action: str) -> dict[str, object]:
    old_bit = (before >> fact) & 1
    bit = old_bit if action == "R" else 1 - old_bit
    after = (before & ~(1 << fact)) | (bit << fact)
    return {
        "step": step,
        "before_code": before,
        "after_code": after,
        "target_fact": fact,
        "new_bit": bit,
        "action": action,
        "text": _event_text(fact, bit),
    }


def _make_stream(stream_id: str, code: int, program: str) -> dict[str, object]:
    specification = _PROGRAMS[program]
    actions = specification["first_actions"] + specification["tail_actions"]
    current = code
    events = []
    for step, (fact, action) in enumerate(zip(specification["targets"], actions, strict=True), start=1):
        event = _make_event(step, current, fact, action)
        events.append(event)
        current = event["after_code"]
    return {
        "id": stream_id,
        "kind": "new_evaluation",
        "program": program,
        "initial_code": code,
        "events": events,
    }


def _validate_stream(stream: Mapping[str, object], horizon: int) -> int:
    if not isinstance(stream, Mapping):
        raise ValueError("stream must be a mapping")
    stream_id = stream.get("id")
    code = stream.get("initial_code")
    events = stream.get("events")
    if not isinstance(stream_id, str) or not stream_id:
        raise ValueError("stream id is invalid")
    if type(code) is not int or not 0 <= code < 16:
        raise ValueError("stream initial code is invalid")
    if not isinstance(events, Sequence) or len(events) != horizon:
        raise ValueError("stream horizon differs")
    current = code
    for step, event in enumerate(events, start=1):
        if not isinstance(event, Mapping):
            raise ValueError("stream event is invalid")
        expected_keys = {"step", "before_code", "after_code", "target_fact", "new_bit", "action", "text"}
        if set(event) != expected_keys:
            raise ValueError("stream event schema differs")
        if any(type(event[key]) is not int for key in ("step", "before_code", "after_code", "new_bit")):
            raise ValueError("event numeric fields must be integers")
        fact, action = event["target_fact"], event["action"]
        if type(fact) is not int or fact not in range(4) or action not in ("C", "R"):
            raise ValueError("stream event assignment differs")
        expected = _make_event(step, current, fact, action)
        if dict(event) != expected:
            raise ValueError("stream event truth or text differs")
        current = expected["after_code"]
    return current


def _validate_training_streams(streams: object) -> None:
    if not isinstance(streams, Sequence) or len(streams) != 256:
        raise ValueError("training stream coverage differs")
    if len({stream.get("id") for stream in streams if isinstance(stream, Mapping)}) != 256:
        raise ValueError("duplicate training stream id")
    counts: Counter[tuple[int, int, int]] = Counter()
    position_counts: list[Counter[tuple[int, str]]] = [Counter() for _ in range(4)]
    orders = set()
    for stream in streams:
        _validate_stream(stream, 4)
        order = []
        for step, event in enumerate(stream["events"]):
            counts[(event["before_code"], event["target_fact"], event["new_bit"])] += 1
            position_counts[step][(event["target_fact"], event["action"])] += 1
            order.append(event["target_fact"])
        orders.add(tuple(order))
    if len(counts) != 128 or set(counts.values()) != {8}:
        raise ValueError("training transition exposure differs")
    if any(len(counter) != 8 or set(counter.values()) != {32} for counter in position_counts):
        raise ValueError("training position exposure differs")
    if orders != _TRAINING_ORDERS:
        raise ValueError("training target orders differ")


def _validate_evaluation_streams(
    streams: object,
    *,
    expected_kind: str | None,
    require_new_programs: bool = False,
) -> None:
    if not isinstance(streams, Sequence) or len(streams) != 32:
        raise ValueError("evaluation stream coverage differs")
    ids = [stream.get("id") for stream in streams if isinstance(stream, Mapping)]
    if len(ids) != 32 or len(set(ids)) != 32:
        raise ValueError("duplicate evaluation stream id")
    programs: dict[str, list[Mapping[str, object]]] = {}
    for stream in streams:
        final = _validate_stream(stream, 16)
        if expected_kind is not None and stream.get("kind") != expected_kind:
            raise ValueError("evaluation stream kind differs")
        if stream["events"][7]["after_code"] != stream["initial_code"] ^ 15 or final != stream["initial_code"] ^ 15:
            raise ValueError("evaluation endpoint differs")
        if any(event["action"] != "R" for event in stream["events"][8:]):
            raise ValueError("evaluation tail must repeat")
        if require_new_programs:
            program = stream.get("program")
            if program not in _PROGRAMS:
                raise ValueError("unknown new evaluation program")
            if stream["id"] != f"fresh-{stream['initial_code']:02d}-{program}":
                raise ValueError("new evaluation stream id differs")
            programs.setdefault(program, []).append(stream)
            first = stream["events"][:8]
            for fact in range(4):
                if sum(event["target_fact"] == fact and event["action"] == "C" for event in first) != 1:
                    raise ValueError("new evaluation changes do not cover each fact")
                if sum(event["target_fact"] == fact and event["action"] == "R" for event in first) != 1:
                    raise ValueError("new evaluation repetitions do not cover each fact")
    if require_new_programs:
        if set(programs) != set(_PROGRAMS) or any(len(rows) != 16 for rows in programs.values()):
            raise ValueError("new evaluation programs are incomplete")
        for program, rows in programs.items():
            expected_targets = _PROGRAMS[program]["targets"]
            expected_actions = _PROGRAMS[program]["first_actions"] + _PROGRAMS[program]["tail_actions"]
            for stream in rows:
                if tuple(event["target_fact"] for event in stream["events"]) != expected_targets:
                    raise ValueError("new evaluation target program differs")
                if tuple(event["action"] for event in stream["events"]) != expected_actions:
                    raise ValueError("new evaluation action program differs")


def _validate_window_exclusion(manifest: Mapping[str, object]) -> None:
    orders = _TRAINING_ORDERS
    for stream in manifest["evaluation_streams"]:
        targets = [event["target_fact"] for event in stream["events"]]
        if any(tuple(targets[index:index + 4]) in orders for index in range(13)):
            raise ValueError("new evaluation target window overlaps training")


def build_answer_manifest(previous_manifest: Mapping[str, object]) -> dict[str, object]:
    """Build the fixed answer-evaluation manifest without reading model outcomes."""
    if not isinstance(previous_manifest, Mapping):
        raise TypeError("previous_manifest must be a mapping")
    _validate_training_streams(previous_manifest.get("training_streams"))
    _validate_evaluation_streams(previous_manifest.get("evaluation_streams"), expected_kind=None)
    old_streams = []
    for original in previous_manifest["evaluation_streams"]:
        stream = deepcopy(original)
        stream["kind"] = "old_continuity"
        stream["benchmark"] = "completed_recurrence"
        old_streams.append(stream)
    new_streams = [
        _make_stream(f"fresh-{code:02d}-{program}", code, program)
        for code in range(16)
        for program in _PROGRAMS
    ]
    manifest = {
        "kind": "independent_fact_answer_protocol_v1",
        "training_streams": deepcopy(previous_manifest["training_streams"]),
        "old_continuity_streams": old_streams,
        "evaluation_streams": new_streams,
        "training_horizon": 4,
        "old_continuity_horizon": 16,
        "evaluation_horizon": 16,
        "programs": list(_PROGRAMS),
    }
    validate_answer_manifest(manifest)
    return manifest


def validate_answer_manifest(manifest: Mapping[str, object]) -> None:
    """Validate all labels and deterministic coverage in an answer manifest."""
    if not isinstance(manifest, Mapping) or manifest.get("kind") != "independent_fact_answer_protocol_v1":
        raise ValueError("answer manifest kind differs")
    if manifest.get("programs") != list(_PROGRAMS):
        raise ValueError("answer program declaration differs")
    _validate_training_streams(manifest.get("training_streams"))
    _validate_evaluation_streams(
        manifest.get("old_continuity_streams"), expected_kind="old_continuity",
    )
    _validate_evaluation_streams(
        manifest.get("evaluation_streams"), expected_kind="new_evaluation", require_new_programs=True,
    )
    all_ids = [stream["id"] for key in ("training_streams", "old_continuity_streams", "evaluation_streams") for stream in manifest[key]]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("stream ids overlap")
    old_signatures = {
        tuple((event["target_fact"], event["action"]) for event in stream["events"])
        for stream in manifest["old_continuity_streams"]
    }
    new_signatures = {
        tuple((event["target_fact"], event["action"]) for event in stream["events"])
        for stream in manifest["evaluation_streams"]
    }
    if old_signatures & new_signatures:
        raise ValueError("new evaluation program reuses old continuity program")
    _validate_window_exclusion(manifest)


def state_catalog(manifest: Mapping[str, object]) -> dict[str, dict[str, object]]:
    """Return explicit initial, primitive-write, and old/new prefix state metadata."""
    validate_answer_manifest(manifest)
    catalog: dict[str, dict[str, object]] = {}
    for code in range(16):
        catalog[f"initial:{code:02d}"] = {
            "code": code, "kind": "initial", "before_code": None, "fact": None,
            "new_bit": None, "after_code": code,
        }
    for code in range(16):
        for fact in range(4):
            for bit in range(2):
                after = (code & ~(1 << fact)) | (bit << fact)
                case_id = f"code-{code:02d}:fact-{fact}:value-{bit}"
                catalog[f"one:{case_id}"] = {
                    "code": after, "kind": "one", "before_code": code, "fact": fact,
                    "new_bit": bit, "after_code": after,
                }
    for group, kind in (("old_continuity_streams", "old_continuity"), ("evaluation_streams", "new_evaluation")):
        for stream in manifest[group]:
            for event in stream["events"]:
                catalog[f"{stream['id']}:step-{event['step']:02d}"] = {
                    "code": event["after_code"], "kind": kind, "stream_id": stream["id"],
                    "step": event["step"], "before_code": event["before_code"],
                    "fact": event["target_fact"], "new_bit": event["new_bit"],
                    "after_code": event["after_code"],
                }
    if len(catalog) != 1168:
        raise ValueError("answer state catalog coverage differs")
    return catalog
