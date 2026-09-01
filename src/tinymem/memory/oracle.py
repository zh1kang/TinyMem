"""Evaluation-only memory that retains known supporting-token positions."""

import torch

from tinymem.memory.interfaces import MemoryPolicy
from tinymem.memory.state import INTEGER_DTYPES, MemoryState


class OracleMemory(MemoryPolicy):
    """Keep candidates whose absolute positions belong to symbolic evidence."""

    def __init__(self, capacity: int, *, target_positions: torch.Tensor) -> None:
        super().__init__(capacity)
        if not isinstance(target_positions, torch.Tensor):
            raise TypeError("target_positions must be a torch.Tensor")
        if target_positions.ndim != 2:
            raise ValueError("target_positions must have shape [batch, positions]")
        if target_positions.dtype not in INTEGER_DTYPES:
            raise TypeError("target_positions must be an integer tensor")
        if (target_positions < -1).any():
            raise ValueError("target_positions must be nonnegative or -1 padding")

        for row in target_positions:
            valid_positions = row[row >= 0]
            if valid_positions.numel() > capacity:
                raise ValueError("target positions must fit within capacity")
            if torch.unique(valid_positions).numel() != valid_positions.numel():
                raise ValueError("target positions must be unique within each row")

        self._target_positions = target_positions.detach().clone()

    @property
    def target_positions(self) -> torch.Tensor:
        """Return a defensive copy of the symbolic target positions."""
        return self._target_positions.clone()

    def update(
        self,
        state: MemoryState,
        candidates: MemoryState,
        *,
        generator: torch.Generator | None = None,
    ) -> MemoryState:
        """Return only expired tokens at the configured evidence positions."""
        self._validate_update(state, candidates)
        if self._target_positions.shape[0] != state.batch_size:
            raise ValueError("target_positions batch size must match state")
        if state.token_ids is None or candidates.token_ids is None:
            raise ValueError("OracleMemory requires token_ids in state and candidates")
        if (state.scores is None) != (candidates.scores is None):
            raise ValueError("state and candidates must agree on score availability")

        combined_values = torch.cat((state.values, candidates.values), dim=1)
        combined_valid = torch.cat((state.valid, candidates.valid), dim=1)
        combined_positions = torch.cat(
            (state.positions, candidates.positions),
            dim=1,
        )
        combined_token_ids = torch.cat(
            (state.token_ids, candidates.token_ids),
            dim=1,
        )
        combined_scores = None
        if state.scores is not None and candidates.scores is not None:
            if state.scores.dtype != candidates.scores.dtype:
                raise ValueError("state and candidate scores must have the same dtype")
            combined_scores = torch.cat((state.scores, candidates.scores), dim=1)

        result = self.initialize(
            batch_size=state.batch_size,
            model_width=state.model_width,
            device=state.values.device,
            dtype=state.values.dtype,
            with_scores=combined_scores is not None,
        )
        assert result.token_ids is not None

        target_positions = self._target_positions.to(state.values.device)
        for batch_index in range(state.batch_size):
            targets = target_positions[batch_index]
            targets = targets[targets >= 0]
            selected_mask = combined_valid[batch_index] & torch.isin(
                combined_positions[batch_index],
                targets,
            )
            selected_indices = torch.nonzero(
                selected_mask,
                as_tuple=False,
            ).squeeze(-1)
            if selected_indices.numel() == 0:
                continue
            selected_positions = combined_positions[batch_index, selected_indices]
            if torch.unique(selected_positions).numel() != selected_positions.numel():
                raise ValueError("state and candidates contain duplicate target positions")
            order = torch.argsort(selected_positions, stable=True)
            selected_indices = selected_indices[order]
            selected_count = selected_indices.numel()

            result.values[batch_index, :selected_count].copy_(
                combined_values[batch_index, selected_indices]
            )
            result.valid[batch_index, :selected_count] = True
            result.positions[batch_index, :selected_count].copy_(
                combined_positions[batch_index, selected_indices]
            )
            result.token_ids[batch_index, :selected_count].copy_(
                combined_token_ids[batch_index, selected_indices]
            )
            if result.scores is not None and combined_scores is not None:
                result.scores[batch_index, :selected_count].copy_(
                    combined_scores[batch_index, selected_indices]
                )

        return result
