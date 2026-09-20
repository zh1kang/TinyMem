"""Fresh evaluation data and independent writer/reader cells for replication."""

import random
from dataclasses import replace

from tinymem.research.delta_fact_data import (
    ENTITIES,
    ROOM_PAIRS,
    Episode,
    FactDataset,
    Statement,
    _build_split,
    _logical_signature,
    validate_dataset,
)

WRITER_SEEDS = tuple(range(4101, 4113))
TEST_SEED = 2026091501
RESERVED_FORMS = (
    "{entity} can be found in the {room}.",
    "At present, {entity} is in the {room}.",
)


def declaration(*, smoke: bool) -> dict:
    return {"version": 1, "writer_seeds": list(WRITER_SEEDS[:2] if smoke else WRITER_SEEDS),
            "test_seed": TEST_SEED, "test_prefixes": 16 if smoke else 64,
            "reserved_forms": list(RESERVED_FORMS),
            "excluded_prefixes": "all original train, validation, and test logical histories",
            "reader_measurements": "all parent readers for each writer; not independent seeds",
            "uncertainty": "paired writer seeds conditional on fixed corpus, panel, and readers",
            "confidence": 0.95, "bootstrap_samples": 10000, "bootstrap_seed": 2026091502}


def build_replication_dataset(original: FactDataset, *, smoke: bool) -> FactDataset:
    """Keep training exactly fixed and replace only the evaluation panel."""

    validate_dataset(original)
    used = {_logical_signature(episode.prefix)
            for episodes in (original.train, original.validation, original.test)
            for episode in episodes}
    spec = declaration(smoke=smoke)
    rng = random.Random(spec["test_seed"])
    test = _build_split("test", spec["test_prefixes"], ("familiar", "heldout"), rng, used)
    rendered_prefixes: dict[str, tuple[Statement, ...]] = {}

    def render(statement: Statement, form: int) -> Statement:
        text = RESERVED_FORMS[form].format(
            entity=ENTITIES[statement.entity], room=ROOM_PAIRS[statement.entity][statement.value])
        return replace(statement, text=text)

    reserved: list[Episode] = []
    for episode in test:
        if episode.wording == "familiar":
            reserved.append(episode)
            continue
        if episode.prefix_id not in rendered_prefixes:
            rendered_prefixes[episode.prefix_id] = tuple(
                render(statement, rng.randrange(len(RESERVED_FORMS))) for statement in episode.prefix)
        tail_form = rng.randrange(len(RESERVED_FORMS))
        tail = tuple(render(statement, tail_form if episode.condition in ("repeat", "correction")
                            else rng.randrange(len(RESERVED_FORMS))) for statement in episode.tail)
        reserved.append(replace(episode, prefix=rendered_prefixes[episode.prefix_id], tail=tail))
    dataset = FactDataset(original.train, original.validation, tuple(reserved))
    validate_dataset(dataset)
    return dataset


def evaluation_cells(protocol: dict) -> list[dict]:
    """Keep old cell identities intact; enumerate the new writer-reader product."""

    if "replication" not in protocol["settings"]:
        return protocol["cells"]
    return [{"index": index, "seed": writer["seed"], "writer_index": writer["index"],
             "parent_index": reader["index"], "reader_seed": reader["seed"],
             "persistent_bytes": 258}
            for index, (writer, reader) in enumerate(
                (writer, reader) for writer in protocol["cells"] for reader in protocol["readers"])]


def training_index(cell: dict) -> int:
    return cell.get("writer_index", cell["index"])
