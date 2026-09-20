"""Bounded, pure-Python data for recurrent four-fact experiments.

The module deliberately keeps learner-specific concerns out of the dataset.
An episode contains rendered statements plus the integer labels needed to
audit the data and to train a caller-owned reader or writer.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product
from typing import Final, TypeAlias

ENTITIES: Final[tuple[str, ...]] = ("Alice", "Bob", "Clara", "David")
ROOM_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("bathroom", "hallway"),
    ("bedroom", "kitchen"),
    ("garden", "office"),
    ("bathroom", "hallway"),
)
FAMILIAR_FORMS: Final[tuple[str, ...]] = (
    "{entity} is in the {room}.",
    "{entity} moved to the {room}.",
    "{entity} went to the {room}.",
)
HELDOUT_FORMS: Final[tuple[str, ...]] = (
    "The {room} is where {entity} is.",
    "{entity}'s location is the {room}.",
)
_WORDINGS: Final[tuple[str, ...]] = ("familiar", "heldout")
_CONDITIONS: Final[tuple[str, ...]] = (
    "no_write",
    "repeat",
    "correction",
    "balanced",
)
_ESCAPED_ENTITIES = "|".join(re.escape(entity) for entity in ENTITIES)
_ESCAPED_ROOMS = "|".join(
    sorted((re.escape(room) for pair in ROOM_PAIRS for room in pair), key=len, reverse=True)
)
_STATEMENT_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "familiar",
        re.compile(rf"(?P<entity>{_ESCAPED_ENTITIES}) is in the (?P<room>{_ESCAPED_ROOMS})\.\Z"),
    ),
    (
        "familiar",
        re.compile(rf"(?P<entity>{_ESCAPED_ENTITIES}) moved to the (?P<room>{_ESCAPED_ROOMS})\.\Z"),
    ),
    (
        "familiar",
        re.compile(rf"(?P<entity>{_ESCAPED_ENTITIES}) went to the (?P<room>{_ESCAPED_ROOMS})\.\Z"),
    ),
    (
        "heldout",
        re.compile(rf"The (?P<room>{_ESCAPED_ROOMS}) is where (?P<entity>{_ESCAPED_ENTITIES}) is\.\Z"),
    ),
    (
        "heldout",
        re.compile(rf"(?P<entity>{_ESCAPED_ENTITIES})'s location is the (?P<room>{_ESCAPED_ROOMS})\.\Z"),
    ),
    (
        "heldout",
        re.compile(rf"(?P<entity>{_ESCAPED_ENTITIES}) can be found in the (?P<room>{_ESCAPED_ROOMS})\.\Z"),
    ),
    (
        "heldout",
        re.compile(rf"At present, (?P<entity>{_ESCAPED_ENTITIES}) is in the (?P<room>{_ESCAPED_ROOMS})\.\Z"),
    ),
)


@dataclass(frozen=True)
class Statement:
    """One fact assignment and its rendered text.

    ``entity`` and ``value`` are audit labels.  A learner can use only
    ``text`` while a validator independently recovers the labels from it.
    """

    entity: int
    value: int
    text: str


_StatementLike: TypeAlias = Statement | str


@dataclass(frozen=True)
class Episode:
    """A recurrent prefix and one evaluation tail."""

    id: str
    prefix_id: str
    split: str
    wording: str
    condition: str
    target: int | None
    prefix: tuple[Statement, ...]
    tail: tuple[Statement, ...]


@dataclass(frozen=True)
class FactDataset:
    """Train, validation, and test episode groups."""

    train: tuple[Episode, ...]
    validation: tuple[Episode, ...]
    test: tuple[Episode, ...]


def _room_for(entity: int, value: int) -> str:
    if type(entity) is not int or not 0 <= entity < len(ENTITIES):
        raise ValueError("entity must be an integer from zero through three")
    if type(value) is not int or value not in (0, 1):
        raise ValueError("value must be zero or one")
    return ROOM_PAIRS[entity][value]


def _statement(entity: int, value: int, wording: str, form_index: int) -> Statement:
    forms = FAMILIAR_FORMS if wording == "familiar" else HELDOUT_FORMS
    if wording not in _WORDINGS or type(form_index) is not int or not 0 <= form_index < len(forms):
        raise ValueError("invalid wording or form index")
    text = forms[form_index].format(entity=ENTITIES[entity], room=_room_for(entity, value))
    return Statement(entity, value, text)


def parse_statement(text: str) -> Statement:
    """Parse one supported statement without executing or evaluating text.

    The returned statement retains the input text so callers can compare the
    parsed labels with the metadata carried by a generated episode.
    """

    return _parse_statement_details(text)[0]


def _parse_statement_details(text: str) -> tuple[Statement, str]:
    if not isinstance(text, str):
        raise ValueError("statement text must be a string")
    for wording, pattern in _STATEMENT_PATTERNS:
        match = pattern.fullmatch(text)
        if match is None:
            continue
        entity = ENTITIES.index(match.group("entity"))
        room = match.group("room")
        try:
            value = ROOM_PAIRS[entity].index(room)
        except ValueError as exc:
            raise ValueError(f"room does not belong to entity {entity}: {text!r}") from exc
        return Statement(entity, value, text), wording
    raise ValueError(f"unsupported fact statement: {text!r}")


def _parsed(statement: _StatementLike) -> Statement:
    if isinstance(statement, str):
        return parse_statement(statement)
    try:
        return _parsed_with_wording(statement)[0]
    except ValueError as exc:
        if not isinstance(statement, Statement):
            raise ValueError("replay accepts Statement values or rendered strings") from exc
        raise


def _parsed_with_wording(statement: Statement) -> tuple[Statement, str]:
    if not isinstance(statement, Statement):
        raise ValueError("episode statements must be Statement values")
    if type(statement.entity) is not int or type(statement.value) is not int:
        raise ValueError("statement metadata must use integer entity and value labels")
    parsed, wording = _parse_statement_details(statement.text)
    if (statement.entity, statement.value) != (parsed.entity, parsed.value):
        raise ValueError("statement metadata does not agree with its text")
    return parsed, wording


def replay(statements: Iterable[_StatementLike]) -> tuple[int | None, ...]:
    """Replay fact writes into a four-entry current assignment.

    Missing facts remain ``None``.  Later writes replace earlier values for
    the same entity, which makes corrections and truthful repetitions explicit.
    """

    state: list[int | None] = [None] * len(ENTITIES)
    for statement in statements:
        parsed = _parsed(statement)
        state[parsed.entity] = parsed.value
    return tuple(state)


def _validate_count(name: str, value: int) -> None:
    if type(value) is not int or value <= 0 or value % 16:
        raise ValueError(f"{name} must be a positive multiple of sixteen")


def _make_prefix(rng: random.Random, code: int) -> tuple[Statement, ...]:
    order = list(range(len(ENTITIES)))
    rng.shuffle(order)
    initial = [(code >> entity) & 1 for entity in range(len(ENTITIES))]
    logical: list[tuple[int, int]] = [(entity, initial[entity]) for entity in order]

    for _ in range(4):
        entity = rng.randrange(len(ENTITIES))
        value = rng.randrange(2)
        logical.append((entity, value))
    return tuple(Statement(entity, value, "") for entity, value in logical)


def _logical_signature(prefix: tuple[Statement, ...]) -> tuple[tuple[int, int], ...]:
    return tuple((statement.entity, statement.value) for statement in prefix)


def _render_logical(
    logical: tuple[Statement, ...], wording: str, rng: random.Random
) -> tuple[Statement, ...]:
    forms = FAMILIAR_FORMS if wording == "familiar" else HELDOUT_FORMS
    return tuple(_statement(statement.entity, statement.value, wording, rng.randrange(len(forms))) for statement in logical)


def _render_tail(
    logical: tuple[tuple[int, int], ...], wording: str, condition: str, rng: random.Random
) -> tuple[Statement, ...]:
    forms = FAMILIAR_FORMS if wording == "familiar" else HELDOUT_FORMS
    if condition in ("repeat", "correction") and logical:
        form_index = rng.randrange(len(forms))
        return tuple(_statement(entity, value, wording, form_index) for entity, value in logical)
    return _render_logical(tuple(Statement(entity, value, "") for entity, value in logical), wording, rng)


def _tail_program(
    prefix: tuple[Statement, ...], condition: str, target: int | None, rng: random.Random
) -> tuple[tuple[int, int], ...]:
    before = replay(prefix)
    if condition == "no_write":
        if target is not None:
            raise ValueError("no-write episodes cannot have a target")
        return ()
    if condition == "repeat":
        if target is None or before[target] is None:
            raise ValueError("repeat target must have a known prefix value")
        return tuple((target, before[target]) for _ in range(8))
    if condition == "correction":
        if target is None or before[target] is None:
            raise ValueError("correction target must have a known prefix value")
        corrected = 1 - before[target]
        return ((target, corrected),) + tuple((target, corrected) for _ in range(7))
    if condition == "balanced":
        first = list(range(len(ENTITIES)))
        second = list(range(len(ENTITIES)))
        rng.shuffle(first)
        rng.shuffle(second)
        return tuple((entity, before[entity]) for entity in first + second)
    raise ValueError(f"unknown condition: {condition}")


def _episode_id(split: str, prefix_number: int, wording: str, condition: str, target: int | None) -> str:
    target_text = "none" if target is None else str(target)
    return f"{split}/p{prefix_number:04d}/{wording}/{condition}-{target_text}"


def _build_split(
    split: str,
    prefix_count: int,
    wordings: tuple[str, ...],
    rng: random.Random,
    used_signatures: set[tuple[tuple[int, int], ...]],
) -> tuple[Episode, ...]:
    episodes: list[Episode] = []
    for prefix_number in range(prefix_count):
        for _attempt in range(1000):
            logical_prefix = _make_prefix(rng, prefix_number % 16)
            signature = _logical_signature(logical_prefix)
            if signature not in used_signatures:
                used_signatures.add(signature)
                break
        else:
            raise RuntimeError("could not create a fresh logical prefix within 1000 attempts")

        prefix_id = f"{split}/p{prefix_number:04d}"
        conditions = (("no_write", None), *( ("repeat", target) for target in range(4)),
                      *(("correction", target) for target in range(4)), ("balanced", None))
        familiar_prefix = _render_logical(logical_prefix, "familiar", rng)
        logical_tails = tuple(
            (condition, target, _tail_program(familiar_prefix, condition, target, rng))
            for condition, target in conditions
        )
        for wording in wordings:
            rendered_prefix = familiar_prefix if wording == "familiar" else _render_logical(logical_prefix, wording, rng)
            for condition, target, logical_tail in logical_tails:
                rendered_tail = _render_tail(logical_tail, wording, condition, rng)
                episodes.append(
                    Episode(
                        id=_episode_id(split, prefix_number, wording, condition, target),
                        prefix_id=prefix_id,
                        split=split,
                        wording=wording,
                        condition=condition,
                        target=target,
                        prefix=rendered_prefix,
                        tail=rendered_tail,
                    )
                )
    return tuple(episodes)


def build_dataset(
    *,
    seed: int = 20260913,
    train_prefixes: int = 256,
    validation_prefixes: int = 32,
    test_prefixes: int = 64,
) -> FactDataset:
    """Build deterministic split-disjoint prefixes and their evaluation forks."""

    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    _validate_count("train_prefixes", train_prefixes)
    _validate_count("validation_prefixes", validation_prefixes)
    _validate_count("test_prefixes", test_prefixes)
    rng = random.Random(seed)
    used_signatures: set[tuple[tuple[int, int], ...]] = set()
    train = _build_split("train", train_prefixes, ("familiar",), rng, used_signatures)
    validation = _build_split("validation", validation_prefixes, ("familiar",), rng, used_signatures)
    test = _build_split("test", test_prefixes, _WORDINGS, rng, used_signatures)
    dataset = FactDataset(train, validation, test)
    validate_dataset(dataset)
    return dataset


def _expected_id(episode: Episode) -> str:
    if not re.fullmatch(r"(?:train|validation|test)/p\d{4}", episode.prefix_id):
        raise ValueError("invalid prefix id")
    prefix_split, _ = episode.prefix_id.split("/", 1)
    if prefix_split != episode.split:
        raise ValueError("prefix id split does not agree with episode split")
    prefix_number = int(episode.prefix_id.rsplit("p", 1)[1])
    return _episode_id(episode.split, prefix_number, episode.wording, episode.condition, episode.target)


def validate_dataset(dataset: FactDataset) -> None:
    """Validate rendered labels, branch semantics, and split invariants.

    Validation reparses every statement from text before checking metadata and
    branch labels, so it does not rely on the generator's label fields.
    """

    if not isinstance(dataset, FactDataset):
        raise ValueError("dataset must be a FactDataset")
    all_episodes: list[Episode] = []
    containers = (("train", dataset.train), ("validation", dataset.validation), ("test", dataset.test))
    for expected_split, episodes in containers:
        if not isinstance(episodes, tuple) or not episodes:
            raise ValueError(f"{expected_split} must be a non-empty tuple")
        for episode in episodes:
            if not isinstance(episode, Episode) or episode.split != expected_split:
                raise ValueError("episode split metadata does not agree with its container")
            if episode.wording not in _WORDINGS or episode.condition not in _CONDITIONS:
                raise ValueError("invalid episode wording or condition")
            if episode.condition in ("repeat", "correction") and (
                type(episode.target) is not int or episode.target not in range(4)
            ):
                raise ValueError("target metadata is invalid")
            if episode.condition in ("no_write", "balanced") and episode.target is not None:
                raise ValueError("target must be absent for this condition")
            if episode.id != _expected_id(episode):
                raise ValueError("episode id does not agree with metadata")
            if len(episode.prefix) != 8:
                raise ValueError("prefix must contain eight writes")
            parsed_prefix_details = tuple(_parsed_with_wording(statement) for statement in episode.prefix)
            parsed_tail_details = tuple(_parsed_with_wording(statement) for statement in episode.tail)
            if any(wording != episode.wording for _, wording in parsed_prefix_details + parsed_tail_details):
                raise ValueError("rendered statement template family does not match episode wording")
            parsed_prefix = tuple(parsed for parsed, _ in parsed_prefix_details)
            parsed_tail = tuple(parsed for parsed, _ in parsed_tail_details)
            if tuple((s.entity, s.value) for s in parsed_prefix) != tuple((s.entity, s.value) for s in episode.prefix):
                raise ValueError("prefix statement metadata does not agree with its text")
            if tuple((s.entity, s.value) for s in parsed_tail) != tuple((s.entity, s.value) for s in episode.tail):
                raise ValueError("tail statement metadata does not agree with its text")
            if {statement.entity for statement in parsed_prefix[:4]} != set(range(4)):
                raise ValueError("first four prefix writes must cover each entity once")
            before = replay(parsed_prefix)
            after = replay(parsed_prefix + parsed_tail)
            if episode.condition == "no_write":
                if parsed_tail or after != before:
                    raise ValueError("no-write branch changed the assignment")
            elif episode.condition == "repeat":
                assert episode.target is not None
                if len(parsed_tail) != 8 or any(
                    statement.entity != episode.target or statement.value != before[episode.target]
                    for statement in parsed_tail
                ) or after != before:
                    raise ValueError("repeat branch does not preserve the target truth")
            elif episode.condition == "correction":
                assert episode.target is not None
                corrected = 1 - before[episode.target]
                if len(parsed_tail) != 8 or parsed_tail[0] != Statement(episode.target, corrected, parsed_tail[0].text) or any(
                    statement.entity != episode.target or statement.value != corrected
                    for statement in parsed_tail
                ) or after[episode.target] != corrected or any(
                    after[entity] != before[entity] for entity in range(4) if entity != episode.target
                ):
                    raise ValueError("correction branch is not an isolated correction")
            else:
                if len(parsed_tail) != 8 or sorted(statement.entity for statement in parsed_tail) != [0, 0, 1, 1, 2, 2, 3, 3]:
                    raise ValueError("balanced branch must contain two writes for each entity")
                if any(statement.value != before[statement.entity] for statement in parsed_tail) or after != before:
                    raise ValueError("balanced branch does not repeat prefix truths")
            all_episodes.append(episode)

    if len({episode.id for episode in all_episodes}) != len(all_episodes):
        raise ValueError("episode ids must be globally unique")
    groups: dict[tuple[str, str, str], list[Episode]] = {}
    for episode in all_episodes:
        groups.setdefault((episode.split, episode.prefix_id, episode.wording), []).append(episode)
    prefix_wordings: dict[tuple[str, str], set[str]] = {}
    for (split, _prefix_id, wording), episodes in groups.items():
        expected_wordings = ("familiar",) if split in ("train", "validation") else _WORDINGS
        if wording not in expected_wordings:
            raise ValueError("heldout wording is excluded from training and validation")
        condition_targets = {(episode.condition, episode.target) for episode in episodes}
        expected_targets = {
            ("no_write", None),
            ("balanced", None),
            *(("repeat", target) for target in range(4)),
            *(("correction", target) for target in range(4)),
        }
        if condition_targets != expected_targets or len(episodes) != len(expected_targets):
            raise ValueError("each logical prefix must have one episode per condition")
        prefix_wordings.setdefault((split, _prefix_id), set()).add(wording)
    for (split, _prefix_id), wordings in prefix_wordings.items():
        expected_wordings = {"familiar"} if split in ("train", "validation") else set(_WORDINGS)
        if wordings != expected_wordings:
            raise ValueError("test prefixes must have paired familiar and heldout wording")
    for (split, prefix_id, wording), episodes in groups.items():
        if any(episode.prefix != episodes[0].prefix for episode in episodes[1:]):
            raise ValueError("condition branches must share the exact rendered prefix")
    prefix_signatures: dict[str, tuple[tuple[int, int], ...]] = {}
    split_signatures: dict[str, set[tuple[tuple[int, int], ...]]] = {key: set() for key, _ in containers}
    split_codes: dict[str, Counter[tuple[int, ...]]] = {key: Counter() for key, _ in containers}
    for (split, prefix_id, _wording), episodes in groups.items():
        signature = _logical_signature(episodes[0].prefix)
        if prefix_id in prefix_signatures and prefix_signatures[prefix_id] != signature:
            raise ValueError("wording branches disagree about the logical prefix")
        prefix_signatures[prefix_id] = signature
        split_signatures[split].add(signature)
        if _wording == "familiar":
            initial_code = tuple(statement.value for statement in sorted(episodes[0].prefix[:4], key=lambda item: item.entity))
            split_codes[split][initial_code] += 1
    if any(left & right for index, left in enumerate(split_signatures.values()) for right in list(split_signatures.values())[index + 1:]):
        raise ValueError("logical prefixes must be disjoint between splits")
    for split, signatures in split_signatures.items():
        if len(signatures) == 0 or len(signatures) % 16 or set(split_codes[split]) != set(product((0, 1), repeat=4)):
            raise ValueError("each split must contain a positive multiple of sixteen prefixes with all initial codes")
        expected_per_code = len(signatures) // 16
        if set(split_codes[split].values()) != {expected_per_code}:
            raise ValueError("initial truth codes must be balanced within each split")

    programs: dict[tuple[str, str, str, int | None], dict[str, tuple[tuple[int, int], ...]]] = {}
    for episode in all_episodes:
        key = (episode.split, episode.prefix_id, episode.condition, episode.target)
        tail_signature = tuple((statement.entity, statement.value) for statement in episode.tail)
        wording_programs = programs.setdefault(key, {})
        if episode.wording in wording_programs and wording_programs[episode.wording] != tail_signature:
            raise ValueError("duplicate wording branches disagree about their tail program")
        wording_programs[episode.wording] = tail_signature
    for (split, _prefix_id, _condition, _target), wording_programs in programs.items():
        if split == "test" and set(wording_programs) == set(_WORDINGS):
            if len(set(wording_programs.values())) != 1:
                raise ValueError("paired test wording branches must share their logical tail")
