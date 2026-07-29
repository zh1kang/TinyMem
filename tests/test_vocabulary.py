import pytest
import torch

from tinymem.data.vocabulary import ControlledVocabulary


def test_controlled_vocabulary_round_trip_and_tensor() -> None:
    text = "Mary moved to the garden.\nQUERY Mary location?"
    vocabulary = ControlledVocabulary.from_texts([text])
    assert vocabulary.decode(vocabulary.encode(text)) == text
    tensor = vocabulary.encode_tensor(text)
    assert tensor.dtype == torch.long
    assert tensor.ndim == 1


def test_controlled_vocabulary_is_deterministic() -> None:
    assert ControlledVocabulary.from_texts(["b a"]).id_to_token == ControlledVocabulary.from_texts(["a b"]).id_to_token


def test_controlled_vocabulary_rejects_special_token_collision() -> None:
    with pytest.raises(ValueError, match="collide"):
        ControlledVocabulary(["<pad>"])
