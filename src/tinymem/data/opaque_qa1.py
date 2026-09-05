"""A labeled bAbI-derived association stress, not an official bAbI score."""

import hashlib
import random
from dataclasses import dataclass

from tinymem.data.reader_gate import ReaderCase
from tinymem.data.symbolic_world import MOVEMENT_SEPARATORS, parse_qa1_movement, parse_qa1_question


PEOPLE = ("Mary", "John", "Daniel", "Sandra")
ROOMS = ("bathroom", "bedroom", "garden", "hallway", "kitchen", "office")
SHORT_NAMES = (*PEOPLE, "Alice", "Bob", "Clara", "David", "Eve")


@dataclass(frozen=True)
class OpaqueQA1World:
    world_id: str
    source_group_ids: tuple[str, str]
    source_case_ids: tuple[str, str]
    entities: tuple[str, ...]
    chunks: tuple[str, ...]
    queries: tuple[ReaderCase, ...]


def make_opaque_qa1_world(
    sources: tuple[ReaderCase, ReaderCase], source_group_ids: tuple[str, str],
    *, world_id: str, seed: int, opaque: bool = True, counterfactual: bool = False,
) -> OpaqueQA1World:
    """Interleave two stories, rename entities, and replay all eight bindings."""
    if not isinstance(world_id, str) or not world_id or world_id != world_id.strip():
        raise ValueError("world_id must be a nonempty stripped string")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(opaque, bool) or not isinstance(counterfactual, bool):
        raise TypeError("opaque and counterfactual must be booleans")
    if len(sources) != 2 or len(source_group_ids) != 2:
        raise ValueError("exactly two source histories and group IDs are required")
    if any(not isinstance(group, str) or not group for group in source_group_ids):
        raise ValueError("source group IDs must be nonempty strings")
    if len(set(source_group_ids)) != 2 or sources[0].context == sources[1].context:
        raise ValueError("source histories must belong to distinct groups")
    parsed = []
    for source in sources:
        if source.category != "babi_qa1":
            raise ValueError("sources must be babi_qa1 cases")
        facts = []
        final = {}
        for index, sentence in enumerate(source.context.splitlines(), 1):
            person, location = parse_qa1_movement(sentence, index)
            if person not in PEOPLE or location.value not in ROOMS:
                raise ValueError("source must use the four bAbI people and six rooms")
            separator = next((value for value in MOVEMENT_SEPARATORS
                              if sentence == person + value + location.value + "."), None)
            if separator is None:
                raise ValueError("source must use an exact supported movement sentence")
            facts.append((separator, person, location.value))
            final[person] = location.value
        if set(final) != set(PEOPLE):
            raise ValueError("each source must contain all four people")
        if final.get(parse_qa1_question(source.question)) != source.answer:
            raise ValueError("source answer disagrees with symbolic replay")
        parsed.append(facts)
    rng = random.Random(f"opaque-qa1-v1:{seed}:{world_id}")
    entities = []
    while len(entities) < 9:
        entity = f"person{rng.getrandbits(80):020x}"
        if entity not in entities:
            entities.append(entity)
    if not opaque:
        entities = list(SHORT_NAMES)
    rooms = list(ROOMS)
    rng.shuffle(rooms)
    if counterfactual:
        rooms = rooms[1:] + rooms[:1]
    room_map = dict(zip(ROOMS, rooms, strict=True))
    stories = []
    for index, facts in enumerate(parsed):
        names = dict(zip(PEOPLE, entities[index * 4:index * 4 + 4], strict=True))
        stories.append([
            names[person] + separator + room_map[room] + "."
            for separator, person, room in facts
        ])
    left, right = stories
    a, b = len(left) // 2, len(right) // 2
    chunks = tuple("\n".join(lines) for lines in (left[:a], right[:b], left[a:], right[b:]))
    context = "\n".join(chunks)
    final = {}
    for index, sentence in enumerate(context.splitlines(), 1):
        person, location = parse_qa1_movement(sentence, index)
        final[person] = location.value
    history_id = hashlib.sha256(context.encode()).hexdigest()
    variant = ("opaque" if opaque else "short") + (":counterfactual" if counterfactual else "")
    queries = tuple(ReaderCase(
        f"{world_id}:{variant}:query-{index}",
        "opaque_qa1_known" if entity in final else "opaque_qa1_missing",
        history_id, context, f"Where is {entity}?", final.get(entity, "unknown"),
    ) for index, entity in enumerate(entities))
    return OpaqueQA1World(world_id, tuple(source_group_ids), tuple(case.case_id for case in sources),
                          tuple(entities), chunks, queries)
