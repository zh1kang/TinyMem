import pytest
import torch

from tinymem.tokenization.byte_tokenizer import ByteTokenizer


@pytest.mark.parametrize("text", ["", "plain ASCII", "東京 🧠 café", "line one\nline two"])
def test_byte_tokenizer_is_perfectly_reversible(text: str) -> None:
    tokenizer = ByteTokenizer()
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_special_tokens_do_not_collide_with_bytes() -> None:
    tokenizer = ByteTokenizer()
    assert min(tokenizer.special_tokens.values()) > 255
    encoded = tokenizer.encode("abc", add_bos=True, add_eos=True)
    assert tokenizer.decode(encoded) == "abc"


def test_byte_tokenizer_returns_long_tensor() -> None:
    tensor = ByteTokenizer().encode_tensor("東京")
    assert tensor.dtype == torch.long
    assert tensor.ndim == 1
