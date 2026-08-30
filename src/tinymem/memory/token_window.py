"""Bounded raw-token history aligned with the local KV-cache window."""

from dataclasses import dataclass

import torch

from tinymem.memory.state import INTEGER_DTYPES


@dataclass(frozen=True)
class RawTokenBatch:
    """Align absolute positions with batched raw token IDs.

    positions: [tokens]
    token_ids: [batch, tokens]
    """

    positions: torch.Tensor
    token_ids: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.positions, torch.Tensor):
            raise TypeError("positions must be a torch.Tensor")
        if not isinstance(self.token_ids, torch.Tensor):
            raise TypeError("token_ids must be a torch.Tensor")
        if self.positions.ndim != 1:
            raise ValueError("positions must have shape [tokens]")
        if self.token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, tokens]")
        if self.positions.dtype not in INTEGER_DTYPES:
            raise TypeError("positions must be an integer tensor")
        if self.token_ids.dtype not in INTEGER_DTYPES:
            raise TypeError("token_ids must be an integer tensor")
        if self.token_ids.shape[0] == 0:
            raise ValueError("token_ids must contain at least one batch row")
        if self.token_ids.shape[1] != self.positions.numel():
            raise ValueError("positions and token_ids must contain the same tokens")
        if self.positions.device != self.token_ids.device:
            raise ValueError("positions and token_ids must be on the same device")
        if (self.positions < 0).any():
            raise ValueError("positions must be nonnegative")
        if self.positions.numel() > 1 and not torch.all(
            self.positions[1:] > self.positions[:-1]
        ):
            raise ValueError("positions must be strictly increasing")

    @classmethod
    def empty(
        cls,
        *,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> "RawTokenBatch":
        """Create an empty raw-token batch."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if dtype not in INTEGER_DTYPES:
            raise TypeError("dtype must be an integer torch dtype")
        return cls(
            positions=torch.empty(0, dtype=torch.int64, device=device),
            token_ids=torch.empty(
                (batch_size, 0),
                dtype=dtype,
                device=device,
            ),
        )


class LocalTokenWindow:
    """Keep raw token IDs synchronized with a bounded local attention window."""

    def __init__(self, max_length: int) -> None:
        if isinstance(max_length, bool) or not isinstance(max_length, int):
            raise TypeError("max_length must be an integer")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.max_length = max_length
        self._state: RawTokenBatch | None = None

    @property
    def initialized(self) -> bool:
        """Return whether the window contains stream state."""
        return self._state is not None

    @property
    def state(self) -> RawTokenBatch:
        """Return a defensive copy of the current window."""
        if self._state is None:
            raise RuntimeError("token window has not observed any tokens")
        return RawTokenBatch(
            positions=self._state.positions.clone(),
            token_ids=self._state.token_ids.clone(),
        )

    def append(
        self,
        token_ids: torch.Tensor,
        *,
        position_offset: int,
    ) -> RawTokenBatch:
        """Append contiguous tokens and return the tokens that just expired."""
        if not isinstance(token_ids, torch.Tensor):
            raise TypeError("token_ids must be a torch.Tensor")
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, tokens]")
        if token_ids.dtype not in INTEGER_DTYPES:
            raise TypeError("token_ids must be an integer tensor")
        if token_ids.shape[0] == 0 or token_ids.shape[1] == 0:
            raise ValueError("token_ids dimensions must be nonempty")
        if isinstance(position_offset, bool) or not isinstance(position_offset, int):
            raise TypeError("position_offset must be an integer")
        if position_offset < 0:
            raise ValueError("position_offset must be nonnegative")

        if self._state is not None:
            if token_ids.shape[0] != self._state.token_ids.shape[0]:
                raise ValueError("batch size cannot change without resetting the window")
            if token_ids.device != self._state.token_ids.device:
                raise ValueError("device cannot change without resetting the window")
            if token_ids.dtype != self._state.token_ids.dtype:
                raise ValueError("dtype cannot change without resetting the window")
            expected_offset = int(self._state.positions[-1].item()) + 1
            if position_offset != expected_offset:
                raise ValueError(
                    f"position_offset must be {expected_offset}, got {position_offset}"
                )

        new_positions = torch.arange(
            position_offset,
            position_offset + token_ids.shape[1],
            device=token_ids.device,
            dtype=torch.int64,
        )
        new_tokens = RawTokenBatch(
            positions=new_positions,
            token_ids=token_ids.detach().clone(),
        )

        old_state = self._state
        if old_state is None:
            combined_positions = new_tokens.positions
            combined_token_ids = new_tokens.token_ids
        else:
            combined_positions = torch.cat(
                (old_state.positions, new_tokens.positions),
                dim=0,
            )
            combined_token_ids = torch.cat(
                (old_state.token_ids, new_tokens.token_ids),
                dim=1,
            )

        excess_length = max(combined_positions.numel() - self.max_length, 0)
        expired_positions = combined_positions[:excess_length]
        expired_token_ids = combined_token_ids[:, :excess_length]
        self._state = RawTokenBatch(
            positions=combined_positions[excess_length:].clone(),
            token_ids=combined_token_ids[:, excess_length:].clone(),
        )
        return RawTokenBatch(
            positions=expired_positions.clone(),
            token_ids=expired_token_ids.clone(),
        )

    def reset(self) -> None:
        """Forget all raw tokens and restart the stream window."""
        self._state = None
