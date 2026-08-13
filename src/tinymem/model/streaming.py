"""Streaming inference over bounded local attention windows."""

from numbers import Integral

import torch

from tinymem.model.kv_cache import KVCache
from tinymem.model.transformer import DecoderOnlyTransformer


class StreamingDecoder:
    """Process an unbounded token stream with a bounded local KV cache."""

    def __init__(self, model: DecoderOnlyTransformer, *, segment_length: int) -> None:
        if not isinstance(model, DecoderOnlyTransformer):
            raise TypeError(
                f"model must be a DecoderOnlyTransformer, got {type(model)}"
            )
        if isinstance(segment_length, bool) or not isinstance(segment_length, Integral):
            raise TypeError("segment_length must be an integer")
        if segment_length <= 0:
            raise ValueError("segment_length must be positive")
        if segment_length > model.config.max_local_tokens:
            raise ValueError("segment_length must not exceed max_local_tokens")

        self.model = model
        self.segment_length = int(segment_length)
        self.caches = [
            KVCache(max_length=model.config.max_local_tokens)
            for _ in range(model.config.n_layers)
        ]
        self._position = 0
        self._batch_size: int | None = None

    @property
    def position(self) -> int:
        """Return the absolute position of the next stream token."""
        return self._position

    @property
    def cache_bytes(self) -> int:
        """Return total key/value storage across all Transformer layers."""
        return sum(cache.nbytes for cache in self.caches)

    @torch.no_grad()
    def process_segment(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Process one segment and return logits for each segment token."""
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(
                f"input_ids must be a torch.Tensor, got {type(input_ids)}"
            )
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must be a rank-two tensor, got shape {input_ids.shape}"
            )
        if input_ids.shape[1] == 0:
            raise ValueError("input segment must contain at least one token")
        if input_ids.shape[1] > self.segment_length:
            raise ValueError(
                f"input segment exceeds segment_length ({self.segment_length})"
            )
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                f"input_ids must be an integer tensor, got dtype {input_ids.dtype}"
            )
        if self._batch_size is None:
            self._batch_size = input_ids.shape[0]
        elif input_ids.shape[0] != self._batch_size:
            raise ValueError("all stream segments must use the same batch size")

        outputs = []
        for index in range(input_ids.shape[1]):
            token_logits = self.model(
                input_ids[:, index : index + 1],
                position_offset=self._position,
                caches=self.caches,
            )
            outputs.append(token_logits)
            self._position = self.caches[0].end_position

        return torch.cat(outputs, dim=1)

    def reset(self) -> None:
        """Clear all cached states and restart absolute positions at zero."""
        for cache in self.caches:
            cache.clear()
        self._position = 0
        self._batch_size = None
