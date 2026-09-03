"""Pre-normalized residual Transformer block for TinyMem."""

from numbers import Real

import torch
from torch import nn

from tinymem.model.attention import AttentionObserver, CausalSelfAttention
from tinymem.model.feedforward import FeedForward
from tinymem.model.kv_cache import KVCache
from tinymem.model.latent_kv_cache import LatentKVCache
from tinymem.model.memory_input import AttentionMemory
from tinymem.model.mla_lite import MLALiteAttention
from tinymem.model.normalization import RMSNorm


class TransformerBlock(nn.Module):
    """Combine normalized attention and feed-forward residual updates.

    Input and output shape: [B, T, D]
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_position_embeddings: int,
        dropout: float = 0.0,
        attention_type: str = "mha",
        kv_latent_dim: int | None = None,
    ) -> None:
        super().__init__()

        if isinstance(d_model, bool) or not isinstance(d_model, int):
            raise TypeError(f"d_model must be an int, got {type(d_model)}")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")

        if isinstance(n_heads, bool) or not isinstance(n_heads, int):
            raise TypeError(f"n_heads must be an int, got {type(n_heads)}")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {n_heads}")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        if isinstance(d_ff, bool) or not isinstance(d_ff, int):
            raise TypeError(f"d_ff must be an int, got {type(d_ff)}")
        if d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {d_ff}")

        if (
            isinstance(max_position_embeddings, bool)
            or not isinstance(max_position_embeddings, int)
        ):
            raise TypeError(
                "max_position_embeddings must be an int, "
                f"got {type(max_position_embeddings)}"
            )
        if max_position_embeddings <= 0:
            raise ValueError(
                "max_position_embeddings must be positive, "
                f"got {max_position_embeddings}"
            )

        if isinstance(dropout, bool) or not isinstance(dropout, Real):
            raise TypeError(f"dropout must be a real number, got {type(dropout)}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}")
        if not isinstance(attention_type, str):
            raise TypeError("attention_type must be a string")
        if attention_type not in {"mha", "mla_lite"}:
            raise ValueError("attention_type must be 'mha' or 'mla_lite'")
        if attention_type == "mha" and kv_latent_dim is not None:
            raise ValueError("kv_latent_dim must be None for MHA")
        if attention_type == "mla_lite":
            if isinstance(kv_latent_dim, bool) or not isinstance(kv_latent_dim, int):
                raise TypeError("kv_latent_dim must be an integer for MLA-lite")
            if not 0 < kv_latent_dim < d_model:
                raise ValueError("kv_latent_dim must be between zero and d_model")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_position_embeddings = max_position_embeddings
        self.dropout_probability = dropout
        self.attention_type = attention_type
        self.kv_latent_dim = kv_latent_dim

        self.norm_attention = RMSNorm(d_model)
        if attention_type == "mha":
            self.attention = CausalSelfAttention(
                d_model=d_model,
                n_heads=n_heads,
                max_position_embeddings=max_position_embeddings,
                dropout=dropout,
            )
        else:
            assert kv_latent_dim is not None
            self.attention = MLALiteAttention(
                d_model=d_model,
                n_heads=n_heads,
                kv_latent_dim=kv_latent_dim,
                max_position_embeddings=max_position_embeddings,
                dropout=dropout,
            )
        self.norm_feed_forward = RMSNorm(d_model)
        self.feed_forward = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )
    def forward(
        self,
        x: torch.Tensor,
        *,
        position_offset: int = 0,
        cache: KVCache | LatentKVCache | None = None,
        attention_observer: AttentionObserver | None = None,
        memory: AttentionMemory | None = None,
    ) -> torch.Tensor:
        """Apply two pre-normalized residual updates."""

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x)}")
        if not x.is_floating_point():
            raise TypeError(f"x must be a floating-point tensor, got {x.dtype}")
        if x.ndim != 3:
            raise ValueError(f"x must be a 3D tensor, got {x.ndim}D")
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"x last dimension must be {self.d_model}, got {x.shape[-1]}"
            )
        if memory is not None and not isinstance(memory, AttentionMemory):
            raise TypeError("memory must be an AttentionMemory or None")

        normalized = self.norm_attention(x)
        normalized_memory = None
        if memory is not None:
            normalized_memory = AttentionMemory(
                values=self.norm_attention(memory.values),
                valid=memory.valid,
                positions=memory.positions,
            )

        attention_output = self.attention(
            normalized,
            position_offset=position_offset,
            cache=cache,
            attention_observer=attention_observer,
            memory=normalized_memory,
        )
        x = x + attention_output

        normalized = self.norm_feed_forward(x)
        feed_forward_output = self.feed_forward(normalized)
        x = x + feed_forward_output

        return x
