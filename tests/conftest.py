"""Shared tiny-reader fixtures; random weights only, never model-quality evidence."""
import re

import pytest
import torch

from tinymem.research.pretrained import PretrainedReader


class WordTokenizer:
    """Reversible local test vocabulary; no pretrained weights or tokenizer."""
    eos_token_id = 0

    def __init__(self):
        self.pieces = [""]
        self.ids = {}

    def encode(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        result = []
        for piece in re.findall(r"\w+|[^\w]", text):
            if piece not in self.ids:
                self.ids[piece] = len(self.pieces)
                self.pieces.append(piece)
            result.append(self.ids[piece])
        return result

    def decode(self, ids, *, skip_special_tokens=False):
        return "".join(self.pieces[int(i)] if int(i) < len(self.pieces) else "?" for i in ids)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and add_generation_prompt and not enable_thinking
        return f"<system>{messages[0]['content']}<user>{messages[1]['content']}<assistant>"


@pytest.fixture
def tiny_reader():
    transformers = pytest.importorskip("transformers")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(
        vocab_size=256, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=2048, eos_token_id=0, attention_dropout=0.0,
    )).requires_grad_(False).eval()
    try:
        yield PretrainedReader(model, WordTokenizer())
    finally:
        torch.set_num_threads(previous)
