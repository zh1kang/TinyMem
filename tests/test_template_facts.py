import pytest
import torch

from tinymem.data.opaque_qa1 import ROOMS, SHORT_NAMES
from tinymem.data.symbolic_world import MOVEMENT_SEPARATORS
from tinymem.memory.packed_tokens import PackedTokenState
from tinymem.memory.template_facts import TemplateFactRetention


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_all_movement_forms_exact_round_trip_and_fixed_allocation(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    codec = TemplateFactRetention(66)
    for person in (*SHORT_NAMES, "person0123456789abcdef0123"):
        for verb in MOVEMENT_SEPARATORS:
            for room in ROOMS:
                sentence = person + verb + room + "."
                state = codec.append_sentence(codec.empty(device=device), sentence)
                assert state.nbytes == state.payload.untyped_storage().nbytes() == 66
                assert TemplateFactRetention(66).text(state) == sentence + "\n\n"
                assert state.payload.device.type == device


def test_six_opaque_sentences_fit_exactly_and_seventh_evicts_whole_oldest():
    codec = TemplateFactRetention(66)
    state = codec.empty()
    sentences = [f"person{index:020x} moved to the kitchen." for index in range(7)]
    for sentence in sentences[:6]:
        state = codec.append_sentence(state, sentence)
    assert codec.length_bits + 6 * 87 == 66 * 8
    assert codec.sentences(state) == sentences[:6]
    previous = state.payload.clone()
    updated = codec.append_sentence(state, sentences[-1])
    assert codec.sentences(updated) == sentences[1:]
    assert torch.equal(state.payload, previous)


def test_update_removes_stale_sentence_and_changes_eviction_order():
    codec = TemplateFactRetention(66)
    state = codec.empty()
    sentences = [f"person{index:020x} moved to the kitchen." for index in range(6)]
    for sentence in sentences:
        state = codec.append_sentence(state, sentence)
    update = sentences[0].replace("moved to the kitchen", "went back to the office")
    state = codec.append_sentence(state, update)
    state = codec.append_sentence(state, "person00000000000000000006 went to the bedroom.")
    assert codec.sentences(state)[:4] == sentences[2:]
    assert codec.sentences(state)[-2] == update


def test_short_names_fit_and_small_budget_rejects_oversized_entry():
    codec = TemplateFactRetention(13)
    state = codec.empty()
    for name in SHORT_NAMES:
        state = codec.append_sentence(state, name + " went to the office.")
    assert len(codec.sentences(state)) == 9
    tiny = TemplateFactRetention(2)
    empty = tiny.empty()
    assert tiny.text(tiny.append_sentence(empty, "person00000000000000000000 went to the office.")) == ""


@pytest.mark.parametrize("sentence", ["Mary moved to the attic.", "Mary moved to the bedroom .", "Mary  moved to the bedroom.",
    "Alice moved to the bedroom.\nJohn went to the garden.", "person0123 moved to the kitchen.",
    "person0123456789ABCDEF0123 went to the kitchen.", "(no facts supplied)"])
def test_unsupported_text_is_never_silently_altered(sentence):
    codec = TemplateFactRetention(66)
    with pytest.raises(ValueError):
        codec.append_sentence(codec.empty(), sentence)


@pytest.mark.parametrize("budget", [0, -1, True, 1, 2.0])
def test_invalid_budget(budget):
    with pytest.raises(ValueError):
        TemplateFactRetention(budget)


def test_rejects_trailing_payload_bits_and_wrong_shape():
    codec = TemplateFactRetention(66)
    damaged = codec.empty().payload.clone()
    damaged[0, -1] = 128
    with pytest.raises(ValueError, match="unused"):
        codec.sentences(PackedTokenState(damaged))
    with pytest.raises(ValueError, match="format"):
        codec.sentences(TemplateFactRetention(65).empty())


def test_rejects_invalid_codes_and_duplicate_people():
    codec = TemplateFactRetention(66)
    for name, verb, room in ((15, 0, 0), (0, 7, 0), (0, 0, 7)):
        word = 1 | (name << (codec.length_bits + 1)) | (verb << (codec.length_bits + 5)) | (room << (codec.length_bits + 8))
        state = PackedTokenState(torch.tensor([list(word.to_bytes(66, "little"))], dtype=torch.uint8))
        with pytest.raises(ValueError):
            codec.sentences(state)
    state = codec._pack(["Mary went to the office.", "Mary moved to the kitchen."], torch.device("cpu"))
    with pytest.raises(ValueError, match="distinct"):
        codec.sentences(state)


def test_shared_dictionary_is_not_per_stream_state():
    codec = TemplateFactRetention(66)
    assert codec.dictionary_serialized_bytes > 0
    assert set(vars(codec.empty())) == {"payload"}
