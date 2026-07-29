"""Lossless controlled-text vocabulary built from first principles."""

from __future__ import annotations

import re
from collections.abc import Iterable

import torch


SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")
_TOKEN_PATTERN = re.compile(r"\s+|\w+|[^\w\s]", flags=re.UNICODE)


class ControlledVocabulary:
    """A deterministic vocabulary for the finite controlled-data language."""

    def __init__(self, tokens: Iterable[str]) -> None:
        token_set = set(tokens)
        if any(not isinstance(token, str) or not token for token in token_set):
            raise ValueError("vocabulary tokens must be nonempty strings")
        collisions = token_set.intersection(SPECIAL_TOKENS)
        if collisions:
            raise ValueError("corpus tokens must not collide with special tokens")
        self.id_to_token = SPECIAL_TOKENS + tuple(sorted(token_set))
        self.token_to_id = {token: index for index, token in enumerate(self.id_to_token)}

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> ControlledVocabulary:
        tokens: list[str] = []
        for text in texts:
            if not isinstance(text, str):
                raise TypeError("texts must contain strings")
            tokens.extend(_TOKEN_PATTERN.findall(text))
        if not tokens:
            raise ValueError("at least one nonempty text is required")
        return cls(tokens)

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        ids = [self.token_to_id.get(token, self.token_to_id["<unk>"]) for token in _TOKEN_PATTERN.findall(text)]
        if add_bos:
            ids.insert(0, self.token_to_id["<bos>"])
        if add_eos:
            ids.append(self.token_to_id["<eos>"])
        return ids

    def decode(self, token_ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        tokens: list[str] = []
        for token_id in token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("token IDs must be integers")
            if not 0 <= token_id < len(self.id_to_token):
                raise ValueError(f"token ID is out of range: {token_id}")
            token = self.id_to_token[token_id]
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return "".join(tokens)

    def encode_tensor(self, text: str, **kwargs: bool) -> torch.Tensor:
        """Encode text as a one-dimensional integer tensor."""
        return torch.tensor(self.encode(text, **kwargs), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.id_to_token)
