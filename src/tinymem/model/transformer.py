"""Minimal decoder-only Transformer for TinyMem."""

import torch
from torch import nn

from tinymem.model.block import TransformerBlock
from tinymem.model.config import ModelConfig
from tinymem.model.normalization import RMSNorm


class DecoderOnlyTransformer(nn.Module):
    """Map token IDs to next-token logits.

    Input shape:  [B, T]
    Output shape: [B, T, V]
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        if not isinstance(config, ModelConfig):
            raise TypeError(
                f"config must be a ModelConfig, got {type(config)}"
            )

        self.config = config

        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
        )
        self.transformer_blocks = nn.ModuleList(
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                max_position_embeddings=config.max_local_tokens,
                dropout=config.dropout,
            )
            for _ in range(config.n_layers)
        )
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """Return vocabulary logits for every input position."""

        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(
                f"input_ids must be a torch.Tensor, got {type(input_ids)}"
            )
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must be a rank-two tensor, got shape {input_ids.shape}"
            )
        if input_ids.shape[1] == 0:
            raise ValueError("input_ids must contain at least one token")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                f"input_ids must be an integer tensor, got dtype {input_ids.dtype}"
            )
        if (input_ids < 0).any() or (input_ids >= self.config.vocab_size).any():
            raise ValueError(
                "token ID values must be in "
                f"[0, {self.config.vocab_size}), got {input_ids}"
            )
        if isinstance(position_offset, bool) or not isinstance(position_offset, int):
            raise TypeError(
                f"position_offset must be an int, got {type(position_offset)}"
            )
        if position_offset < 0:
            raise ValueError(
                f"position_offset must be nonnegative, got {position_offset}"
            )
        if position_offset + input_ids.shape[1] > self.config.max_local_tokens:
            raise ValueError(
                "input sequence exceeds max_local_tokens "
                f"({self.config.max_local_tokens})"
            )
        hidden_states = self.token_embedding(input_ids)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, position_offset=position_offset)
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits
