"""Memory-independent learned write decisions for stream segments."""

from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F


class TokenSegmentWriteGate(nn.Module):
    """Detect locally relevant token patterns without reading memory state."""

    def __init__(self, model_width: int, *, kernel_size: int = 3) -> None:
        super().__init__()
        if isinstance(model_width, bool) or not isinstance(model_width, Integral):
            raise TypeError("model_width must be an integer")
        if model_width <= 0:
            raise ValueError("model_width must be positive")
        if isinstance(kernel_size, bool) or not isinstance(kernel_size, Integral):
            raise TypeError("kernel_size must be an integer")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.model_width = int(model_width)
        self.kernel_size = int(kernel_size)
        self.patterns = nn.Conv1d(
            self.model_width,
            self.model_width,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        )
        self.score = nn.Linear(self.model_width, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.ones_(self.score.bias)

    def _pattern_features(
        self,
        token_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        radius = self.kernel_size // 2
        padded = F.pad(token_embeddings, (0, 0, radius, radius))
        token_count = token_embeddings.shape[1]
        weight = self.patterns.weight
        bias = self.patterns.bias
        taps = [
            F.linear(
                padded[:, index : index + token_count],
                weight[:, :, index],
                bias if index == 0 else None,
            )
            for index in range(self.kernel_size)
        ]
        return torch.stack(taps).sum(dim=0)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        token_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Return one write logit for each batch row."""
        if not isinstance(token_embeddings, torch.Tensor):
            raise TypeError("token_embeddings must be a torch.Tensor")
        if not isinstance(token_valid, torch.Tensor):
            raise TypeError("token_valid must be a torch.Tensor")
        if token_embeddings.ndim != 3:
            raise ValueError(
                "token_embeddings must have shape [batch, tokens, model_width]"
            )
        if token_embeddings.shape[2] != self.model_width:
            raise ValueError(
                f"token embedding width must be {self.model_width}"
            )
        if token_valid.shape != token_embeddings.shape[:2]:
            raise ValueError(
                f"token_valid must have shape {token_embeddings.shape[:2]}"
            )
        if token_valid.dtype != torch.bool:
            raise TypeError("token_valid must be a boolean tensor")
        if token_valid.device != token_embeddings.device:
            raise ValueError("token embeddings and validity must share a device")
        if not token_embeddings.is_floating_point():
            raise TypeError("token_embeddings must be floating point")
        if token_embeddings.device != self.patterns.weight.device:
            raise ValueError("token embeddings and write gate must share a device")
        if token_embeddings.dtype != self.patterns.weight.dtype:
            raise TypeError("token embeddings and write gate must share a dtype")

        masked_embeddings = token_embeddings * token_valid.unsqueeze(-1).to(
            dtype=token_embeddings.dtype
        )
        features = F.silu(self._pattern_features(masked_embeddings))
        features = features.masked_fill(
            ~token_valid.unsqueeze(-1),
            torch.finfo(features.dtype).min,
        )
        pooled = features.amax(dim=1)
        row_valid = token_valid.any(dim=1, keepdim=True)
        pooled = torch.where(row_valid, pooled, torch.zeros_like(pooled))
        return self.score(pooled)
