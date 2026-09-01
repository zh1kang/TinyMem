"""Track cumulative attention scores across a sliding key window."""

from dataclasses import dataclass

import torch

from tinymem.memory.attention_scores import attention_received


INTEGER_DTYPES = (torch.int32, torch.int64)


@dataclass(frozen=True)
class AttentionScoreState:
    """Align absolute key positions with cumulative scores.

    positions: [keys]
    scores:    [batch, keys]
    """

    positions: torch.Tensor
    scores: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.positions, torch.Tensor):
            raise TypeError("positions must be a torch.Tensor")
        if not isinstance(self.scores, torch.Tensor):
            raise TypeError("scores must be a torch.Tensor")
        if self.positions.ndim != 1:
            raise ValueError("positions must have shape [keys]")
        if self.positions.dtype not in INTEGER_DTYPES:
            raise TypeError("positions must be an integer tensor")
        if self.scores.ndim != 2:
            raise ValueError("scores must have shape [batch, keys]")
        if self.scores.shape[0] == 0:
            raise ValueError("scores must contain at least one batch row")
        if self.scores.shape[1] != self.positions.numel():
            raise ValueError("scores and positions must contain the same keys")
        if self.scores.dtype != torch.float32:
            raise TypeError("scores must use torch.float32")
        if self.positions.device != self.scores.device:
            raise ValueError("positions and scores must be on the same device")
        if (self.positions < 0).any():
            raise ValueError("positions must be nonnegative")
        if self.positions.numel() > 1 and not torch.all(
            self.positions[1:] > self.positions[:-1]
        ):
            raise ValueError("positions must be strictly increasing")
        if not torch.isfinite(self.scores).all():
            raise ValueError("scores must contain only finite values")

    @classmethod
    def empty(
        cls,
        *,
        batch_size: int,
        device: torch.device | str,
    ) -> "AttentionScoreState":
        """Create an empty score state for one stream batch."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return cls(
            positions=torch.empty(0, dtype=torch.int64, device=device),
            scores=torch.empty(
                (batch_size, 0),
                dtype=torch.float32,
                device=device,
            ),
        )


class CumulativeAttentionTracker:
    """Own cumulative scores for keys still present in the local window."""

    def __init__(self) -> None:
        self._state: AttentionScoreState | None = None

    @property
    def initialized(self) -> bool:
        """Return whether the tracker has observed an attention call."""
        return self._state is not None

    @property
    def state(self) -> AttentionScoreState:
        """Return a defensive copy of the current tracked state."""
        if self._state is None:
            raise RuntimeError("tracker has not observed attention probabilities")
        return AttentionScoreState(
            positions=self._state.positions.clone(),
            scores=self._state.scores.clone(),
        )

    def update(
        self,
        attention_prob: torch.Tensor,
        key_positions: torch.Tensor,
    ) -> AttentionScoreState:
        """Accumulate current attention and return keys that just expired."""
        if not isinstance(key_positions, torch.Tensor):
            raise TypeError("key_positions must be a torch.Tensor")
        received_scores = attention_received(attention_prob)
        tracked_key_count = key_positions.numel()
        if tracked_key_count == 0:
            raise ValueError("key_positions must contain at least one tracked key")
        if tracked_key_count > received_scores.shape[1]:
            raise ValueError("key_positions exceed the observed attention keys")
        contribution = AttentionScoreState(
            positions=key_positions.clone(),
            scores=received_scores[:, -tracked_key_count:].detach(),
        )

        if self._state is None:
            self._state = contribution
            return AttentionScoreState.empty(
                batch_size=contribution.scores.shape[0],
                device=contribution.scores.device,
            )

        if contribution.scores.shape[0] != self._state.scores.shape[0]:
            raise ValueError("batch size cannot change without resetting the tracker")
        if contribution.scores.device != self._state.scores.device:
            raise ValueError("device cannot change without resetting the tracker")
        if (
            contribution.positions[0] < self._state.positions[0]
            or contribution.positions[-1] < self._state.positions[-1]
        ):
            raise ValueError("key positions cannot move backward")

        old_state = self._state
        retained_mask = torch.isin(old_state.positions, contribution.positions)
        retained_positions = old_state.positions[retained_mask]
        retained_scores = old_state.scores[:, retained_mask]
        new_positions = contribution.positions
        new_scores = contribution.scores.clone()
        if retained_positions.numel() > 0:
            current_columns = torch.searchsorted(
                new_positions,
                retained_positions,
            )
            new_scores[:, current_columns] += retained_scores
        self._state = AttentionScoreState(
            positions=new_positions,
            scores=new_scores,
        )
        expired_mask = ~retained_mask
        return AttentionScoreState(
            positions=old_state.positions[expired_mask],
            scores=old_state.scores[:, expired_mask],
        )

    def reset(self) -> None:
        """Forget all tracked positions and cumulative scores."""
        self._state = None
