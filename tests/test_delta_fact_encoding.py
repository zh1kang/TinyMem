import torch
import pytest

from test_readout_runner import tiny_reader
from tinymem.memory.delta_slots import DeltaSlotWriter
from tinymem.research.adapted_readout import frozen_history_features
from tinymem.research.delta_fact_data import ROOM_PAIRS, build_dataset, replay
from tinymem.research.delta_fact_encoding import build_feature_cache, encode_episode
from tinymem.research.delta_fact_readout import SlotReadout
from tinymem.research.delta_fact_training import train_batch
from tinymem.research.reader_adaptation import attach_reader_lora


def test_feature_cache_uses_unique_statement_text_and_owned_cpu_features(tiny_reader):
    dataset = build_dataset(seed=17, train_prefixes=16, validation_prefixes=16, test_prefixes=16)
    episodes = dataset.train[:2]

    cache = build_feature_cache(tiny_reader, episodes)

    texts = {statement.text for episode in episodes for statement in (*episode.prefix, *episode.tail)}
    assert set(cache) == texts
    assert all(feature.device.type == "cpu" for feature in cache.values())
    assert all(feature.dtype == torch.float32 for feature in cache.values())
    assert all(not feature.requires_grad and feature.grad_fn is None for feature in cache.values())
    text = next(iter(cache))
    history_ids = tuple(tiny_reader.tokenizer.encode(text + "\n\n", add_special_tokens=False))
    expected = frozen_history_features(tiny_reader, history_ids)
    torch.testing.assert_close(cache[text], expected, rtol=0, atol=0)


def test_feature_cache_rejects_a_frozen_attached_lora_reader(tiny_reader):
    dataset = build_dataset(seed=19, train_prefixes=16, validation_prefixes=16, test_prefixes=16)
    attach_reader_lora(tiny_reader, rank=2, checkpointing=False)
    tiny_reader.model.requires_grad_(False)
    tiny_reader.model.eval()

    with pytest.raises(ValueError, match="unadapted base reader"):
        build_feature_cache(tiny_reader, dataset.train[:1])


def test_encode_episode_uses_replay_labels_at_prefix_and_total_endpoints(tiny_reader):
    dataset = build_dataset(seed=23, train_prefixes=16, validation_prefixes=16, test_prefixes=16)
    episode = next(row for row in dataset.train if row.condition == "correction")
    cache = build_feature_cache(tiny_reader, (episode,))
    snapshot = {text: feature.clone() for text, feature in cache.items()}

    encoded = encode_episode(tiny_reader, episode, cache)

    assert encoded.episode_id == episode.id
    assert encoded.split == "train"
    assert encoded.before_ids
    assert [endpoint.after_write for endpoint in encoded.endpoints] == [8, 16]
    assert len(encoded.features) == 16
    assert not hasattr(encoded, "history_ids")
    for feature, statement in zip(encoded.features, (*episode.prefix, *episode.tail), strict=True):
        assert feature.shape[1] == tiny_reader.model.config.hidden_size
        assert feature.device.type == "cpu" and feature.dtype == torch.float32
        assert not feature.requires_grad and feature.grad_fn is None
        torch.testing.assert_close(feature, snapshot[statement.text], rtol=0, atol=0)
    for endpoint, statements in zip(encoded.endpoints, (episode.prefix, (*episode.prefix, *episode.tail)), strict=True):
        expected = replay(statements)
        assert len(endpoint.queries) == 4
        assert [query.answer for query in endpoint.queries] == [
            ROOM_PAIRS[entity][expected[entity]] for entity in range(4)
        ]
        assert all(query.category == "update_known" for query in endpoint.queries)
    assert all(torch.equal(cache[text], snapshot[text]) for text in cache)


def test_no_write_episode_has_only_prefix_endpoint(tiny_reader):
    dataset = build_dataset(seed=29, train_prefixes=16, validation_prefixes=16, test_prefixes=16)
    episode = next(row for row in dataset.train if row.condition == "no_write")
    encoded = encode_episode(tiny_reader, episode, build_feature_cache(tiny_reader, (episode,)))

    assert [endpoint.after_write for endpoint in encoded.endpoints] == [8]
    assert len(encoded.features) == 8


def test_query_ids_and_native_question_fragments_are_stable_across_world_wordings(tiny_reader):
    dataset = build_dataset(seed=31, train_prefixes=16, validation_prefixes=16, test_prefixes=16)
    familiar = next(row for row in dataset.test if row.prefix_id == "test/p0000" and row.wording == "familiar"
                    and row.condition == "repeat" and row.target == 0)
    heldout = next(row for row in dataset.test if row.prefix_id == familiar.prefix_id and row.wording == "heldout"
                   and row.condition == familiar.condition and row.target == familiar.target)
    familiar_encoded = encode_episode(tiny_reader, familiar, build_feature_cache(tiny_reader, (familiar,)))
    heldout_encoded = encode_episode(tiny_reader, heldout, build_feature_cache(tiny_reader, (heldout,)))

    for left, right in zip(familiar_encoded.endpoints, heldout_encoded.endpoints, strict=True):
        assert [query.case_id for query in left.queries] == [query.case_id for query in right.queries]
        assert [query.after_ids for query in left.queries] == [query.after_ids for query in right.queries]
        assert all(query.answer_ids for query in left.queries)


def test_heldout_split_is_preserved_and_train_batch_rejects_it(tiny_reader):
    dataset = build_dataset(seed=37, train_prefixes=16, validation_prefixes=16, test_prefixes=16)
    episode = dataset.test[0]
    encoded = encode_episode(tiny_reader, episode, build_feature_cache(tiny_reader, (episode,)))
    writer = DeltaSlotWriter(16, 8, key_width=4)
    bridge = SlotReadout(memory_width=8, reader_width=16)
    optimizer = torch.optim.AdamW([*writer.parameters(), *bridge.parameters()])

    assert encoded.split == "test"
    with pytest.raises(ValueError, match="training examples only"):
        train_batch(tiny_reader, writer, bridge, (encoded,), optimizer)
