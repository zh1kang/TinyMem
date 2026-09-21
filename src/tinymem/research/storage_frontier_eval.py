"""Cache-free batched native-token reads for storage-frontier evaluation."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import torch

from tinymem.evaluation.reader_gate import normalized_answer
from tinymem.research.adapted_readout import configure_read_adapter
from tinymem.research.prefix_reader import _check_ids
from tinymem.research.pretrained import PretrainedReader
from tinymem.research.storage_frontier_training import AnswerTokens


@contextmanager
def base_encoder(reader: PretrainedReader) -> Iterator[None]:
    """Run the unadapted base model, then restore the frozen adapted reader.

    PEFT re-enables ``requires_grad`` on the active adapter when its
    ``disable_adapter`` context exits, which breaks the frozen-reader
    invariant that ``generate`` and the feature encoder require.
    """
    with reader.model.disable_adapter():
        yield
    configure_read_adapter(reader, trainable=False)


def _reader_is_frozen(reader: PretrainedReader) -> None:
    if any(module.training for module in reader.model.modules()):
        raise ValueError("generation requires the reader in evaluation mode")
    if any(parameter.requires_grad for parameter in reader.model.parameters()):
        raise ValueError("generation requires a frozen reader")
    if any(parameter.grad is not None for parameter in reader.model.parameters()):
        raise ValueError("generation requires a reader without parameter gradients")


def _memory(reader: PretrainedReader, value: torch.Tensor, width: int) -> torch.Tensor:
    if (value.ndim != 2 or value.shape[1] != width or not value.is_floating_point()):
        raise ValueError("memory must have floating [positions, reader_width] shape")
    if value.device != reader.model.device:
        raise ValueError("memory and reader must share a device")
    cast = value.to(dtype=reader.model.get_input_embeddings().weight.dtype)
    if not torch.isfinite(cast).all():
        raise ValueError("memory must be finite in the reader dtype")
    return cast


def _stop_ids(reader: PretrainedReader) -> set[int]:
    eos = reader.model.generation_config.eos_token_id
    if isinstance(eos, int):
        return {eos}
    return {int(token) for token in (eos or ())}


def _position_ids(mask: torch.Tensor) -> torch.Tensor:
    positions = mask.cumsum(dim=-1) - 1
    return positions.masked_fill(mask == 0, 0)


@torch.inference_mode()
def generate(
    reader: PretrainedReader,
    tokens: Sequence[AnswerTokens],
    memories: Sequence[torch.Tensor],
    max_new_tokens: int = 8,
) -> list[dict[str, object]]:
    """Generate native greedy answers for variable-length memory prompts in one batch.

    The answer field in ``AnswerTokens`` is intentionally never read.  Every
    row is decoded from the reader's full vocabulary and no key/value cache or
    hidden state survives the call.
    """
    _reader_is_frozen(reader)
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    if len(tokens) != len(memories):
        raise ValueError("one memory is required per answer")
    if not tokens:
        return []

    embedding = reader.model.get_input_embeddings()
    device = reader.model.device
    width = embedding.embedding_dim
    rows: list[tuple[torch.Tensor, torch.Tensor]] = []
    lengths: list[int] = []
    for item, memory in zip(tokens, memories, strict=True):
        before = torch.as_tensor(item.before, dtype=torch.long, device=device)
        after = torch.as_tensor(item.after, dtype=torch.long, device=device)
        _check_ids(reader, before, "before_ids")
        _check_ids(reader, after, "after_ids")
        if after.numel() == 0:
            raise ValueError("after_ids must contain the question and assistant suffix")
        current = _memory(reader, memory, width)
        length = before.numel() + current.shape[0] + after.numel()
        if length + max_new_tokens > reader.model.config.max_position_embeddings:
            raise ValueError("input and continuation exceed reader context; truncation is forbidden")
        rows.append((torch.cat((embedding(before), current, embedding(after))), current))
        lengths.append(length)

    # Left padding keeps each row's native positions while producing one dense
    # tensor.  The model sees zero attention for pads and position zero there.
    max_prompt = max(lengths)
    if max_prompt + max_new_tokens > reader.model.config.max_position_embeddings:
        raise ValueError("input and continuation exceed reader context; truncation is forbidden")
    batch = embedding.weight.new_zeros((len(rows), max_prompt, width))
    mask = torch.zeros((len(rows), max_prompt), dtype=torch.long, device=device)
    for row, (prompt, _) in enumerate(rows):
        batch[row, -prompt.shape[0]:] = prompt
        mask[row, -prompt.shape[0]:] = 1

    stop = _stop_ids(reader)
    pad = reader.model.generation_config.pad_token_id
    if pad is None:
        pad = getattr(reader.tokenizer, "pad_token_id", None)
    if pad is None:
        pad = next(iter(stop), 0)
    if not isinstance(pad, int) or not 0 <= pad < reader.model.config.vocab_size:
        raise ValueError("reader must define an in-vocabulary pad token")

    generated: list[list[int]] = [[] for _ in rows]
    finished = torch.zeros(len(rows), dtype=torch.bool, device=device)
    for _ in range(max_new_tokens):
        output = reader.model(
            inputs_embeds=batch,
            attention_mask=mask,
            position_ids=_position_ids(mask),
            use_cache=False,
            logits_to_keep=1,
        )
        next_ids = output.logits[:, -1, :].argmax(dim=-1)
        active = ~finished
        for row, token in enumerate(next_ids.tolist()):
            if bool(active[row]):
                generated[row].append(token)
        next_ids = torch.where(finished, torch.full_like(next_ids, pad), next_ids)
        batch = torch.cat((batch, embedding(next_ids).unsqueeze(1)), dim=1)
        mask = torch.cat((mask, active.to(dtype=mask.dtype).unsqueeze(1)), dim=1)
        finished |= torch.tensor([token in stop for token in next_ids.tolist()], device=device)
        if bool(finished.all()):
            break

    return [
        {"prediction": reader.tokenizer.decode(ids, skip_special_tokens=True), "generated_ids": ids}
        for ids in generated
    ]


def normalized_exact_match(prediction: str, reference: str) -> bool:
    """Compare answers with the project's shared text normalization."""
    return normalized_answer(prediction) == normalized_answer(reference)
