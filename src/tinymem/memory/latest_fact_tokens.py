"""Domain-aware qa1 raw-sentence retention without a future-query input."""

from collections.abc import Sequence
from typing import Protocol

import torch

from tinymem.data.symbolic_world import MOVEMENT_SEPARATORS, parse_qa1_movement, validate_fact_inputs
from tinymem.memory.packed_tokens import PackedTokenRetention, PackedTokenState
from tinymem.memory.vocabulary_tokens import VocabularyTokenRetention


class SentenceTokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool) -> str: ...


def _check_sentence(sentence: str) -> None:
    validate_fact_inputs(sentence, 1)
    if sentence.splitlines() != [sentence] or any(marker in sentence[:-1] for marker in (".", "?", "!")):
        raise ValueError("input must be one complete sentence without internal sentence punctuation")


def _movement_person(sentence: str) -> str:
    _check_sentence(sentence)
    if sum(sentence.count(separator) for separator in MOVEMENT_SEPARATORS) != 1:
        raise ValueError("sentence must contain one movement")
    return parse_qa1_movement(sentence, 1)[0]


def _encode_sentences(tokenizer: SentenceTokenizer, sentences: Sequence[str]) -> list[int]:
    if not sentences:
        return []
    return tokenizer.encode("\n".join(sentences) + "\n\n", add_special_tokens=False)


def append_latest_fact_sentence(
    packing: PackedTokenRetention | VocabularyTokenRetention, tokenizer: SentenceTokenizer,
    state: PackedTokenState, sentence: str,
) -> PackedTokenState:
    """Apply one raw-fact selection policy with either lossless storage format."""
    _check_sentence(sentence)
    ids, valid = packing.materialize(state, pad_id=0)
    if ids.shape[0] != 1:
        raise ValueError("sentence retention requires a single stream")
    retained_ids = ids[0, valid[0]].tolist()
    retained_text = tokenizer.decode(retained_ids, skip_special_tokens=False)
    retained = [line for line in retained_text.splitlines() if line]
    keys = [_movement_person(line) for line in retained]
    if len(set(keys)) != len(keys):
        raise ValueError("retained state must contain at most one fact per person")
    if _encode_sentences(tokenizer, retained) != retained_ids:
        raise ValueError("retained text must use the complete-sentence format")
    if not any(separator in sentence for separator in MOVEMENT_SEPARATORS):
        return state
    person = _movement_person(sentence)
    retained = [line for key, line in zip(keys, retained, strict=True) if key != person]
    if packing.fits(_encode_sentences(tokenizer, [sentence])):
        retained.append(sentence)
    selected_ids = _encode_sentences(tokenizer, retained)
    while not packing.fits(selected_ids):
        retained.pop(0)
        selected_ids = _encode_sentences(tokenizer, retained)
    current = torch.tensor([selected_ids], dtype=torch.long, device=state.payload.device)
    return packing.append(packing.empty(1, device=state.payload.device), current, torch.ones_like(current, dtype=torch.bool))


class LatestFactTokenRetention(PackedTokenRetention):
    """Keep latest movement sentences, evicting the least recently updated person.

    Input boundaries are visible complete sentences, not supporting-fact labels.
    The fixed qa1 grammar ignores unrelated prose. No symbolic values are stored.
    """

    def __init__(self, capacity: int, vocab_size: int, tokenizer: SentenceTokenizer) -> None:
        super().__init__(capacity, vocab_size)
        self.tokenizer = tokenizer

    def append_sentence(self, state: PackedTokenState, sentence: str) -> PackedTokenState:
        return append_latest_fact_sentence(self, self.tokenizer, state, sentence)
