"""Query-blind streaming tokenization with stable before/after boundaries."""

from collections.abc import Sequence

from tinymem.research.pretrained import PretrainedReader


def encode_update_chunks(
    reader: PretrainedReader, chunks: Sequence[str], *, max_chunk_tokens: int,
) -> tuple[tuple[int, ...], ...]:
    """Each write owns two newlines; reject merges rather than changing history."""
    if type(max_chunk_tokens) is not int or max_chunk_tokens <= 0:
        raise ValueError("max_chunk_tokens must be a positive integer")
    if (isinstance(chunks, (str, bytes)) or not chunks
            or any(not isinstance(chunk, str) or not chunk or chunk != chunk.strip() or "\n\n" in chunk for chunk in chunks)):
        raise ValueError("chunks must be nonempty, stripped strings")
    encoded = tuple(tuple(reader.tokenizer.encode(chunk + "\n\n", add_special_tokens=False)) for chunk in chunks)
    expected = tuple(reader.tokenizer.encode("\n\n".join(chunks) + "\n\n", add_special_tokens=False))
    if any(not ids or len(ids) > max_chunk_tokens for ids in encoded):
        raise ValueError("empty or overlength write; no truncation or filtering is allowed")
    if tuple(token for ids in encoded for token in ids) != expected:
        raise ValueError("tokenizer merges across a streaming write boundary")
    return encoded
