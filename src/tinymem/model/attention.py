"""Causal multi-head self-attention for the TinyMem decoder."""

from numbers import Real

import torch
from torch import nn

from tinymem.model.rope import RotaryEmbedding


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

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)



        q = q.reshape(x.size(0), x.size(1), self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(x.size(0), x.size(1), self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(x.size(0), x.size(1), self.n_heads, self.head_dim).transpose(1, 2)

        q = self.rotary(q, position_offset=position_offset)
        k = self.rotary(k, position_offset=position_offset)

        attn_score = q @ k.transpose(-2, -1) / (self.head_dim**0.5)
        causal_mask = torch.triu(
            torch.ones(
                attn_score.size(-2),
                attn_score.size(-1),
                dtype=torch.bool,
                device=attn_score.device,
            ),
            diagonal=1,
        )
        attn_score = attn_score.masked_fill(causal_mask, float("-inf"))
        attn_prob = torch.softmax(attn_score, dim=-1)
        attn_prob = self.dropout(attn_prob)

        attn_output = attn_prob @ v
        attn_output = attn_output.transpose(1, 2).reshape(
            x.size(0),
            x.size(1),
            self.d_model,
        )
        output = self.out_proj(attn_output)
        return output
