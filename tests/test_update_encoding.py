from pathlib import Path
from types import SimpleNamespace

import pytest

from tinymem.research.memory_prompt import encode_memory_example
from tinymem.research.pretrained import PretrainedReader
from tinymem.research.update_encoding import encode_update_chunks
from test_memory_prompt import BoundaryTokenizer
from test_memory_updates import episode


def reader(tokenizer):
    return PretrainedReader(SimpleNamespace(config=SimpleNamespace(max_position_embeddings=4096)), tokenizer)


def check_prefixes(shared_reader, row, limit):
    before = encode_update_chunks(shared_reader, row.initial_chunks, max_chunk_tokens=limit)
    stages = [(row.before, before)]
    for branch in row.branches:
        after = encode_update_chunks(shared_reader, (*row.initial_chunks, branch.event), max_chunk_tokens=limit)
        assert after[:4] == before
        stages.append((branch.queries, after))
    for cases, chunks in stages:
        encoded = [encode_memory_example(shared_reader, case) for case in cases]
        assert all(item.history_ids == tuple(token for chunk in chunks for token in chunk) for item in encoded)
        assert len({item.before_ids for item in encoded}) == 1
        assert len({item.after_ids for item in encoded}) == 10
    return before


def test_punctuation_merges_and_shared_queries_keep_identical_prefixes():
    check_prefixes(reader(BoundaryTokenizer()), episode(), 4096)


def test_native_qwen_tokenizer_without_loading_model_weights():
    transformers = pytest.importorskip("transformers")
    snapshot = Path("data/raw/pretrained/qwen3-1.7b")
    if not (snapshot / "tokenizer.json").is_file():
        pytest.skip("optional pinned local tokenizer is not staged")
    tokenizer = transformers.AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    for index in range(5):
        check_prefixes(reader(tokenizer), episode(index), 512)


def test_cross_chunk_merges_fail_closed():
    class MergingTokenizer(BoundaryTokenizer):
        def encode(self, text, *, add_special_tokens):
            ids = super().encode(text, add_special_tokens=add_special_tokens)
            return ids[:-1] if "\n\nb" in text else ids

    with pytest.raises(ValueError, match="merges"):
        encode_update_chunks(reader(MergingTokenizer()), ("a", "b"), max_chunk_tokens=100)


@pytest.mark.parametrize("chunks", [(), "a", ("",), ("a\n",), (" a",), ("a\n\nb",), (12,)])
def test_bad_chunks_fail(chunks):
    with pytest.raises(ValueError):
        encode_update_chunks(reader(BoundaryTokenizer()), chunks, max_chunk_tokens=100)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, 1])
def test_bad_or_too_small_limit_fails(limit):
    with pytest.raises(ValueError):
        encode_update_chunks(reader(BoundaryTokenizer()), ("whole fact",), max_chunk_tokens=limit)
