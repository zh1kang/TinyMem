"""Reversible UTF-8 byte tokenizer with separate special-token IDs."""

from collections.abc import Iterable

import torch


class ByteTokenizer:
    """Map UTF-8 bytes to IDs 0 through 255 without learned state."""

    special_tokens = {"<pad>": 256, "<bos>": 257, "<eos>": 258, "<unk>": 259}
    vocab_size = 260

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        token_ids = list(text.encode("utf-8"))
        if add_bos:
            token_ids.insert(0, self.special_tokens["<bos>"])
        if add_eos:
            token_ids.append(self.special_tokens["<eos>"])
        return token_ids

    def decode(self, token_ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        byte_values: list[int] = []
        for token_id in token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("token IDs must be integers")
            if 0 <= token_id <= 255:
                byte_values.append(token_id)
            elif token_id in self.special_tokens.values():
                if not skip_special_tokens:
                    raise ValueError("special tokens cannot be decoded as UTF-8 bytes")
            else:
                raise ValueError(f"token ID is out of range: {token_id}")
        try:
            return bytes(byte_values).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("token IDs do not form valid UTF-8 text") from error

    def encode_tensor(self, text: str, **kwargs: bool) -> torch.Tensor:
        """Encode text as a one-dimensional integer tensor."""
        return torch.tensor(self.encode(text, **kwargs), dtype=torch.long)
