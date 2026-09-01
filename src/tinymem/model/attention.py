"""Causal multi-head self-attention for the TinyMem decoder."""

from collections.abc import Callable
from numbers import Real

import torch
from torch import nn

from tinymem.model.kv_cache import KVCache
from tinymem.model.memory_input import AttentionMemory
from tinymem.model.rope import RotaryEmbedding


AttentionObserver = Callable[[torch.Tensor, torch.Tensor], None]


class CausalSelfAttention(nn.Module):
    """Compute causal self-attention over [batch, sequence, model_dim] inputs.

    Input shape:  [B, T, D]
    Output shape: [B, T, D]
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
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
        self.max_position_embeddings = max_position_embeddings
        self.dropout_probability = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary = RotaryEmbedding(self.head_dim, self.max_position_embeddings)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        position_offset: int = 0,
        cache: KVCache | None = None,
        attention_observer: AttentionObserver | None = None,
        memory: AttentionMemory | None = None,
    ) -> torch.Tensor:
        """Return attention output without allowing future-token access."""

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x)}")
        if x.dim() != 3:
            raise ValueError(f"x must be 3-dimensional, got {x.dim()}")
        if x.size(2) != self.d_model:
            raise ValueError(
                f"x.size(2) must be {self.d_model}, got {x.size(2)}"
            )
        if not x.is_floating_point():
            raise TypeError(f"x must be a floating-point tensor, got {x.dtype}")
        if isinstance(position_offset, bool) or not isinstance(position_offset, int):
            raise TypeError(
                f"position_offset must be an int, got {type(position_offset)}"
            )
        if position_offset < 0:
            raise ValueError("position_offset must be nonnegative")
        if cache is not None and not isinstance(cache, KVCache):
            raise TypeError(f"cache must be a KVCache or None, got {type(cache)}")
        if cache is not None and position_offset != cache.end_position:
            raise ValueError(
                "position_offset must equal cache.end_position when using a cache"
            )
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

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(x.size(0), x.size(1), self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(x.size(0), x.size(1), self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(x.size(0), x.size(1), self.n_heads, self.head_dim).transpose(1, 2)

        q = self.rotary(q, position_offset=position_offset)
        k = self.rotary(k, position_offset=position_offset)

        if cache is None:
            attention_keys = k
            attention_values = v
            key_start_position = position_offset
        else:
            cache.append(k, v)
            attention_keys, attention_values = cache.get()
            key_start_position = cache.start_position

        memory_slot_count = 0
        if memory is not None and memory.slot_count > 0:
            memory_keys = self.k_proj(memory.values)
            memory_values = self.v_proj(memory.values)
            memory_keys = memory_keys.reshape(
                x.size(0),
                memory.slot_count,
                self.n_heads,
                self.head_dim,
            ).transpose(1, 2)
            memory_values = memory_values.reshape(
                x.size(0),
                memory.slot_count,
                self.n_heads,
                self.head_dim,
            ).transpose(1, 2)
            safe_positions = memory.positions.masked_fill(~memory.valid, 0)
            memory_keys = self.rotary(memory_keys, position_ids=safe_positions)
            attention_keys = torch.cat((memory_keys, attention_keys), dim=-2)
            attention_values = torch.cat((memory_values, attention_values), dim=-2)
            memory_slot_count = memory.slot_count

        attn_score = (
            q @ attention_keys.transpose(-2, -1) / (self.head_dim**0.5)
        )
        query_positions = torch.arange(
            position_offset,
            position_offset + x.size(1),
            device=x.device,
        )
        local_key_positions = torch.arange(
            key_start_position,
            key_start_position + attention_keys.size(-2) - memory_slot_count,
            device=x.device,
        )
        local_causal_mask = (
            local_key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        )
        if memory_slot_count > 0:
            assert memory is not None
            memory_mask = (~memory.valid).unsqueeze(1).expand(
                -1,
                x.size(1),
                -1,
            )
            local_mask = local_causal_mask.unsqueeze(0).expand(
                x.size(0),
                -1,
                -1,
            )
            attention_mask = torch.cat(
                (memory_mask, local_mask),
                dim=-1,
            ).unsqueeze(1)
            attn_score = attn_score.masked_fill(
                attention_mask,
                float("-inf"),
            )
        else:
            attn_score = attn_score.masked_fill(
                local_causal_mask,
                float("-inf"),
            )
        attn_prob = torch.softmax(attn_score, dim=-1)
        if attention_observer is not None:
            attention_observer(
                attn_prob.detach().clone(),
                local_key_positions.clone(),
            )

        attn_prob = self.dropout(attn_prob)

        attn_output = attn_prob @ attention_values
        attn_output = attn_output.transpose(1, 2).reshape(
            x.size(0),
            x.size(1),
            self.d_model,
        )
        output = self.out_proj(attn_output)
        return output
