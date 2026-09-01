"""Streaming inference over bounded local attention windows."""

from numbers import Integral

import torch

from tinymem.memory.attention_tracker import (
    AttentionScoreState,
    CumulativeAttentionTracker,
)
from tinymem.memory.candidates import build_scored_token_candidates
from tinymem.memory.interfaces import MemoryPolicy
from tinymem.memory.state import MemoryState
from tinymem.memory.token_window import LocalTokenWindow, RawTokenBatch
from tinymem.model.kv_cache import KVCache
from tinymem.model.memory_input import AttentionMemory
from tinymem.model.transformer import DecoderOnlyTransformer


class StreamingDecoder:
    """Process a token stream with a local cache and optional persistent memory."""

    def __init__(
        self,
        model: DecoderOnlyTransformer,
        *,
        segment_length: int,
        memory_policy: MemoryPolicy | None = None,
    ) -> None:
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
        if memory_policy is not None and not isinstance(memory_policy, MemoryPolicy):
            raise TypeError("memory_policy must be a MemoryPolicy or None")

        self.model = model
        self.segment_length = int(segment_length)
        self.memory_policy = memory_policy
        self.caches = [
            KVCache(max_length=model.config.max_local_tokens)
            for _ in range(model.config.n_layers)
        ]
        self._token_window = (
            LocalTokenWindow(model.config.max_local_tokens)
            if memory_policy is not None
            else None
        )
        self._attention_tracker = (
            CumulativeAttentionTracker()
            if memory_policy is not None
            else None
        )
        self._memory_state: MemoryState | None = None
        self._expired_scores: AttentionScoreState | None = None
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

    @property
    def memory_bytes(self) -> int:
        """Return persistent-memory storage, or zero before initialization."""
        if self._memory_state is None:
            return 0
        return self._memory_state.nbytes

    @property
    def memory_state(self) -> MemoryState:
        """Return a defensive copy of the current persistent memory."""
        if self.memory_policy is None:
            raise RuntimeError("streaming decoder has no memory policy")
        if self._memory_state is None:
            raise RuntimeError("persistent memory has not been initialized")
        return MemoryState(
            values=self._memory_state.values.clone(),
            valid=self._memory_state.valid.clone(),
            positions=self._memory_state.positions.clone(),
            token_ids=(
                None
                if self._memory_state.token_ids is None
                else self._memory_state.token_ids.clone()
            ),
            scores=(
                None
                if self._memory_state.scores is None
                else self._memory_state.scores.clone()
            ),
        )

    def _initialize_memory(self, input_ids: torch.Tensor) -> None:
        if self.memory_policy is None or self._memory_state is not None:
            return
        self._memory_state = self.memory_policy.initialize(
            batch_size=input_ids.shape[0],
            model_width=self.model.config.d_model,
            device=self.model.token_embedding.weight.device,
            dtype=self.model.token_embedding.weight.dtype,
            with_scores=True,
        )

    def _memory_for_attention(self) -> AttentionMemory | None:
        if self._memory_state is None:
            return None
        return AttentionMemory(
            values=self._memory_state.values,
            valid=self._memory_state.valid,
            positions=self._memory_state.positions,
        )

    def _observe_attention(
        self,
        attention_prob: torch.Tensor,
        key_positions: torch.Tensor,
    ) -> None:
        if self._attention_tracker is None:
            raise RuntimeError("attention tracker is not initialized")
        expired_scores = self._attention_tracker.update(
            attention_prob,
            key_positions,
        )
        if expired_scores.positions.numel() == 0:
            return
        if self._expired_scores is not None:
            raise RuntimeError("more than one score batch expired in one token step")
        self._expired_scores = expired_scores

    def _store_expired_tokens(self, expired_tokens: RawTokenBatch) -> None:
        expired_scores = self._expired_scores
        self._expired_scores = None
        if expired_tokens.positions.numel() == 0:
            if expired_scores is not None:
                raise RuntimeError("attention scores expired without raw tokens")
            return
        if expired_scores is None:
            raise RuntimeError("raw tokens expired without attention scores")
        if self.memory_policy is None or self._memory_state is None:
            raise RuntimeError("persistent memory is not initialized")

        scored_candidates = build_scored_token_candidates(
            expired_tokens=expired_tokens,
            expired_scores=expired_scores,
            token_embedding=self.model.token_embedding,
        )
        self._memory_state = self.memory_policy.update(
            self._memory_state,
            scored_candidates,
        )

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
        self._initialize_memory(input_ids)

        outputs = []
        for index in range(input_ids.shape[1]):
            position_offset = self._position
            token_ids = input_ids[:, index : index + 1]
            token_logits = self.model(
                token_ids,
                position_offset=position_offset,
                caches=self.caches,
                memory=self._memory_for_attention(),
                attention_observer=(
                    self._observe_attention
                    if self.memory_policy is not None
                    else None
                ),
            )
            outputs.append(token_logits)
            self._position = self.caches[0].end_position
            if self._token_window is not None:
                expired_tokens = self._token_window.append(
                    token_ids,
                    position_offset=position_offset,
                )
                self._store_expired_tokens(expired_tokens)

        return torch.cat(outputs, dim=1)

    def reset(self) -> None:
        """Clear all cached states and restart absolute positions at zero."""
        for cache in self.caches:
            cache.clear()
        if self._token_window is not None:
            self._token_window.reset()
        if self._attention_tracker is not None:
            self._attention_tracker.reset()
        self._memory_state = None
        self._expired_scores = None
        self._position = 0
        self._batch_size = None
