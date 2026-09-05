"""Paired one-event updates of the existing opaque, multi-question worlds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import random
import re
from typing import Literal

from tinymem.data.opaque_qa1 import ROOMS, make_opaque_qa1_world
from tinymem.data.reader_gate import ReaderCase
from tinymem.data.symbolic_world import MOVEMENT_SEPARATORS


EventKind = Literal["addition", "repetition", "correction"]
EVENT_KINDS: tuple[EventKind, ...] = ("addition", "repetition", "correction")
_ENTITY = r"person[0-9a-f]{20}"
_FACT = re.compile(
    rf"({_ENTITY})(?:{'|'.join(map(re.escape, MOVEMENT_SEPARATORS))})"
    rf"({'|'.join(ROOMS)})\."
)


@dataclass(frozen=True)
class UpdateBranch:
    kind: EventKind
    event: str
    target: str
    queries: tuple[ReaderCase, ...]


@dataclass(frozen=True)
class UpdateEpisode:
    episode_id: str
    source_group_ids: tuple[str, str]
    source_case_ids: tuple[str, str]
    source_context_sha256: tuple[str, str]
    entities: tuple[str, ...]
    initial_chunks: tuple[str, ...]
    before: tuple[ReaderCase, ...]
    branches: tuple[UpdateBranch, ...]


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def replay_update_chunks(chunks: Sequence[str]) -> dict[str, str]:
    """Independent strict replay: do not consume generated answers or qa1 parsing."""
    if isinstance(chunks, (str, bytes)) or not chunks:
        raise ValueError("expected nonempty history chunks")
    state = {}
    for chunk in chunks:
        if not isinstance(chunk, str) or not chunk or chunk != chunk.strip():
            raise ValueError("history chunks must be nonempty and stripped")
        for line in chunk.split("\n"):
            match = _FACT.fullmatch(line)
            if match is None:
                raise ValueError("history contains a noncanonical opaque movement fact")
            state[match[1]] = match[2]
    return state


def _queries(episode_id: str, stage: str, chunks: tuple[str, ...],
             entities: tuple[str, ...], answers: Mapping[str, str]) -> tuple[ReaderCase, ...]:
    # Every chunk owns the same trailing separator, including before an event.
    context = "\n\n".join(chunks)
    return tuple(ReaderCase(
        f"{episode_id}:{stage}:query-{index}",
        "update_known" if entity in answers else "update_missing",
        text_sha256(context), context, f"Where is {entity}?", answers.get(entity, "unknown"),
    ) for index, entity in enumerate(entities))


def make_update_episode(
    sources: tuple[ReaderCase, ReaderCase], source_group_ids: tuple[str, str],
    *, episode_id: str, seed: int,
) -> UpdateEpisode:
    """Generate labels by state transitions, then check with independent replay."""
    world = make_opaque_qa1_world(sources, source_group_ids, world_id=episode_id, seed=seed)
    rng = random.Random(f"memory-update-v1:{seed}:{episode_id}")
    entities = list(world.entities)
    while len(entities) < 10:
        entity = f"person{rng.getrandbits(80):020x}"
        if entity not in entities:
            entities.append(entity)
    entities = tuple(entities)
    initial = {entity: case.answer for entity, case in zip(entities[:8], world.queries[:8], strict=True)}
    target = rng.choice(entities[:8])
    new_room = rng.choice([room for room in ROOMS if room != initial[target]])
    repeated_fact = [line for chunk in world.chunks for line in chunk.splitlines()
                     if line.startswith(target + " ")][-1]
    separator = next(value for value in MOVEMENT_SEPARATORS
                     if repeated_fact == target + value + initial[target] + ".")
    before = _queries(episode_id, "before", world.chunks, entities, initial)
    branches = []
    for kind in EVENT_KINDS:
        event_target = entities[8] if kind == "addition" else target
        room = initial[target] if kind == "repetition" else new_room
        event = event_target + separator + room + "."
        answers = {**initial, event_target: room}
        branches.append(UpdateBranch(kind, event, event_target, _queries(
            episode_id, kind, (*world.chunks, event), entities, answers,
        )))
    episode = UpdateEpisode(
        episode_id, world.source_group_ids, world.source_case_ids,
        tuple(text_sha256(source.context) for source in sources), entities, world.chunks,
        before, tuple(branches),
    )
    validate_update_episode(episode)
    return episode


def validate_update_episode(episode: UpdateEpisode) -> None:
    """Validate serialized or constructed data without trusting its gold labels."""
    if not isinstance(episode.episode_id, str) or not episode.episode_id or episode.episode_id != episode.episode_id.strip():
        raise ValueError("episode ID must be nonempty and stripped")
    for values in (episode.source_group_ids, episode.source_case_ids, episode.source_context_sha256):
        if (not isinstance(values, tuple) or len(values) != 2
                or any(not isinstance(value, str) or not value or value != value.strip() for value in values)
                or len(set(values)) != 2):
            raise ValueError("two distinct source identities are required")
    if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in episode.source_context_sha256):
        raise ValueError("source context hashes must be SHA-256 digests")
    if (not isinstance(episode.entities, tuple) or len(episode.entities) != 10
            or any(not isinstance(entity, str) or re.fullmatch(_ENTITY, entity) is None for entity in episode.entities)
            or len(set(episode.entities)) != 10):
        raise ValueError("ten distinct opaque query entities are required")
    if not isinstance(episode.initial_chunks, tuple) or len(episode.initial_chunks) != 4:
        raise ValueError("four initial history chunks are required")
    initial = replay_update_chunks(episode.initial_chunks)
    if set(initial) != set(episode.entities[:8]):
        raise ValueError("initial history must contain exactly the first eight entities")
    if not isinstance(episode.branches, tuple) or tuple(branch.kind for branch in episode.branches) != EVENT_KINDS:
        raise ValueError("exactly one addition, repetition, and correction is required in declared order")
    stages = [("before", episode.initial_chunks, episode.before)]
    repeated, corrected = episode.branches[1:]
    if repeated.target != corrected.target:
        raise ValueError("repetition and correction must target the same initial binding")
    for branch in episode.branches:
        if not isinstance(branch.event, str) or "\n" in branch.event:
            raise ValueError("each branch must contain exactly one event")
        event = replay_update_chunks((branch.event,))
        if tuple(event) != (branch.target,):
            raise ValueError("event target does not match its visible fact")
        if branch.kind == "addition":
            if branch.target != episode.entities[8]:
                raise ValueError("addition must introduce the ninth entity")
        else:
            if branch.target not in initial:
                raise ValueError("repetition or correction must target an existing binding")
            same = event[branch.target] == initial[branch.target]
            if same != (branch.kind == "repetition"):
                raise ValueError("repetition must preserve and correction must change the old value")
        if branch.kind == "repetition" and branch.event not in "\n".join(episode.initial_chunks).splitlines():
            raise ValueError("repetition must repeat an exact visible fact")
        stages.append((branch.kind, (*episode.initial_chunks, branch.event), branch.queries))
    seen = set()
    for stage, chunks, queries in stages:
        state = replay_update_chunks(chunks)
        context = "\n\n".join(chunks)
        if not isinstance(queries, tuple) or len(queries) != 10:
            raise ValueError("every state must serve all ten questions")
        for index, (entity, case) in enumerate(zip(episode.entities, queries, strict=True)):
            expected = ReaderCase(
                f"{episode.episode_id}:{stage}:query-{index}",
                "update_known" if entity in state else "update_missing",
                text_sha256(context), context, f"Where is {entity}?", state.get(entity, "unknown"),
            )
            if case != expected or case.case_id in seen:
                raise ValueError("query metadata or answer disagrees with independent replay")
            seen.add(case.case_id)


def validate_update_splits(
    splits: Mapping[str, Sequence[UpdateEpisode]], *,
    source_groups: Mapping[str, Mapping[str, object]],
    blocked_groups: set[str],
) -> None:
    """Reject aliases, repeated histories, reserve use, and dishonest source IDs."""
    if set(splits) != {"train", "development", "confirmation"} or any(not rows for rows in splits.values()):
        raise ValueError("nonempty train, development, and confirmation splits are required")
    episodes, groups, contexts, source_episodes = set(), set(), set(), set()
    for split, rows in splits.items():
        for episode in rows:
            validate_update_episode(episode)
            if episode.episode_id in episodes:
                raise ValueError("duplicate episode identity")
            episodes.add(episode.episode_id)
            for group, case, context in zip(episode.source_group_ids, episode.source_case_ids,
                                            episode.source_context_sha256, strict=True):
                source_episode = case.rsplit(":question-", 1)[0]
                if group in groups or context in contexts or source_episode in source_episodes:
                    raise ValueError("source/history overlap within or across splits")
                if group in blocked_groups:
                    raise ValueError("source belongs to an existing reserved or forbidden group")
                metadata = source_groups.get(group)
                # These are membership sets, not index-aligned lists. The builder
                # verifies each exact representative against the raw source first.
                if (metadata is None or source_episode not in metadata["episodes"]
                        or context not in metadata["context_sha256"]):
                    raise ValueError("source identity disagrees with the connected-group manifest")
                if split == "confirmation" and metadata["excluded"] is not False:
                    raise ValueError("confirmation source was consumed by the prior native study")
                groups.add(group)
                contexts.add(context)
                source_episodes.add(source_episode)


def update_episode_from_dict(row: dict) -> UpdateEpisode:
    """Restore tuple ownership at the JSON boundary and reject invalid records."""
    episode = UpdateEpisode(
        episode_id=row["episode_id"],
        **{key: tuple(row[key]) for key in ("source_group_ids", "source_case_ids", "source_context_sha256",
                                           "entities", "initial_chunks")},
        before=tuple(ReaderCase(**case) for case in row["before"]),
        branches=tuple(UpdateBranch(branch["kind"], branch["event"], branch["target"],
                                    tuple(ReaderCase(**case) for case in branch["queries"]))
                       for branch in row["branches"]),
    )
    if set(row) != set(episode.__dataclass_fields__) or any(
        set(branch) != set(UpdateBranch.__dataclass_fields__) for branch in row["branches"]
    ):
        raise ValueError("unexpected episode or branch fields")
    validate_update_episode(episode)
    return episode
