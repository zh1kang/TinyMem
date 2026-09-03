"""MLA-lite attention with a shared low-rank key-value latent."""

from collections.abc import Callable
from numbers import Real

import torch
from torch import nn

from tinymem.model.latent_kv_cache import LatentKVCache
from tinymem.model.memory_input import AttentionMemory
from tinymem.model.rope import RotaryEmbedding


AttentionObserver = Callable[[torch.Tensor, torch.Tensor], None]


class MLALiteAttention(nn.Module):
    """Compute causal attention from a shared compressed KV representation.

    Input shape:  [B, T, D]
    Output shape: [B, T, D]
    Latent shape: [B, T, C]
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        kv_latent_dim: int,
        max_position_embeddings: int,
        dropout: float = 0.0,
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
        if (d_model // n_heads) % 2 != 0:
            raise ValueError("the dimension of each attention head must be even")
        if isinstance(kv_latent_dim, bool) or not isinstance(kv_latent_dim, int):
            raise TypeError(
                f"kv_latent_dim must be an int, got {type(kv_latent_dim)}"
            )
        if kv_latent_dim <= 0:
            raise ValueError(
                f"kv_latent_dim must be positive, got {kv_latent_dim}"
            )
        if kv_latent_dim >= d_model:
            raise ValueError("kv_latent_dim must be smaller than d_model")
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

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.kv_latent_dim = kv_latent_dim
        self.max_position_embeddings = max_position_embeddings
        self.dropout_probability = float(dropout)

        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_down_proj = nn.Linear(d_model, kv_latent_dim, bias=False)
        self.k_up_proj = nn.Linear(kv_latent_dim, d_model, bias=False)
        self.v_up_proj = nn.Linear(kv_latent_dim, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary = RotaryEmbedding(self.head_dim, max_position_embeddings)
        self.dropout = nn.Dropout(self.dropout_probability)

    def forward(
        self,
        x: torch.Tensor,
        *,
        position_offset: int = 0,
        cache: LatentKVCache | None = None,
        attention_observer: AttentionObserver | None = None,
        memory: AttentionMemory | None = None,
    ) -> torch.Tensor:
        """Return causal attention output and optionally update a latent cache."""

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x)}")
        if x.ndim != 3:
            raise ValueError(f"x must be 3-dimensional, got {x.ndim}")
        if x.shape[2] != self.d_model:
            raise ValueError(
                f"x.shape[2] must be {self.d_model}, got {x.shape[2]}"
            )
        if not x.is_floating_point():
            raise TypeError(f"x must be a floating-point tensor, got {x.dtype}")
        if isinstance(position_offset, bool) or not isinstance(position_offset, int):
            raise TypeError(
                f"position_offset must be an int, got {type(position_offset)}"
            )
        if position_offset < 0:
            raise ValueError("position_offset must be nonnegative")
        if cache is not None and not isinstance(cache, LatentKVCache):
            raise TypeError(
                f"cache must be a LatentKVCache or None, got {type(cache)}"
            )
        if cache is not None and position_offset != cache.end_position:
            raise ValueError(
                "position_offset must equal cache.end_position when using a cache"
            )
        if cache is not None and cache.latents is not None:
            cached_latents = cache.get()
            if cached_latents.shape[0] != x.shape[0]:
                raise ValueError("cache and x must have the same batch size")
            if cached_latents.shape[2] != self.kv_latent_dim:
                raise ValueError(
                    f"cache latent dimension must be {self.kv_latent_dim}"
                )
            if cached_latents.dtype != x.dtype:
                raise ValueError("cache and x must have the same dtype")
            if cached_latents.device != x.device:
                raise ValueError("cache and x must be on the same device")
        if attention_observer is not None and not callable(attention_observer):
            raise TypeError("attention_observer must be callable or None")
        if memory is not None:
            if not isinstance(memory, AttentionMemory):
                raise TypeError("memory must be an AttentionMemory or None")
            if memory.values.shape[0] != x.shape[0]:
                raise ValueError("memory and x must have the same batch size")
            if memory.values.shape[2] != self.d_model:
                raise ValueError(f"memory model width must be {self.d_model}")
            if memory.values.dtype != x.dtype:
                raise ValueError("memory and x must have the same dtype")
            if memory.values.device != x.device:
                raise ValueError("memory and x must be on the same device")
            if (memory.positions[memory.valid] >= position_offset).any():
                raise ValueError("valid memory positions must precede local queries")

        queries = self.q_proj(x)
        current_latents = self.kv_down_proj(x)
        if cache is None:
            attention_latents = current_latents
            key_start_position = position_offset
        else:
            if x.shape[1] > cache.max_length:
                raise ValueError("input sequence exceeds the latent cache length")
            cache.append(current_latents)
            attention_latents = cache.get()
            key_start_position = cache.start_position

        keys = self.k_up_proj(attention_latents)
        values = self.v_up_proj(attention_latents)
        queries = queries.reshape(
            x.shape[0], x.shape[1], self.n_heads, self.head_dim
        ).transpose(1, 2)
        keys = keys.reshape(
            x.shape[0], attention_latents.shape[1], self.n_heads, self.head_dim
        ).transpose(1, 2)
        values = values.reshape(
            x.shape[0], attention_latents.shape[1], self.n_heads, self.head_dim
        ).transpose(1, 2)
        queries = self.rotary(queries, position_offset=position_offset)
        keys = self.rotary(keys, position_offset=key_start_position)

        memory_slot_count = 0
        if memory is not None and memory.slot_count > 0:
            memory_latents = self.kv_down_proj(memory.values)
            memory_keys = self.k_up_proj(memory_latents)
            memory_values = self.v_up_proj(memory_latents)
            memory_keys = memory_keys.reshape(
                x.shape[0], memory.slot_count, self.n_heads, self.head_dim
            ).transpose(1, 2)
            memory_values = memory_values.reshape(
                x.shape[0], memory.slot_count, self.n_heads, self.head_dim
            ).transpose(1, 2)
            safe_positions = memory.positions.masked_fill(~memory.valid, 0)
            memory_keys = self.rotary(memory_keys, position_ids=safe_positions)
            keys = torch.cat((memory_keys, keys), dim=-2)
            values = torch.cat((memory_values, values), dim=-2)
            memory_slot_count = memory.slot_count

        attention_scores = queries @ keys.transpose(-2, -1)
        attention_scores = attention_scores / (self.head_dim**0.5)
        query_positions = torch.arange(
            position_offset,
            position_offset + x.shape[1],
            device=x.device,
        )
        local_key_positions = torch.arange(
            key_start_position,
            key_start_position + attention_latents.shape[1],
            device=x.device,
        )
        local_causal_mask = (
            local_key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        )
        if memory_slot_count > 0:
            assert memory is not None
            memory_mask = (~memory.valid).unsqueeze(1).expand(
                -1, x.shape[1], -1
            )
            local_mask = local_causal_mask.unsqueeze(0).expand(
                x.shape[0], -1, -1
            )
            attention_mask = torch.cat(
                (memory_mask, local_mask), dim=-1
            ).unsqueeze(1)
        else:
            attention_mask = local_causal_mask
        attention_scores = attention_scores.masked_fill(
            attention_mask, float("-inf")
        )
        attention_probabilities = torch.softmax(attention_scores, dim=-1)
        if attention_observer is not None:
            attention_observer(
                attention_probabilities.detach().clone(),
                local_key_positions.clone(),
            )
        attention_probabilities = self.dropout(attention_probabilities)

        attention_output = attention_probabilities @ values
        attention_output = attention_output.transpose(1, 2).reshape(
            x.shape[0], x.shape[1], self.d_model
        )
        return self.out_proj(attention_output)
