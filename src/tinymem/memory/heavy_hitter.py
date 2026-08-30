"""Hybrid recent-token and cumulative-importance memory baseline."""

import torch

from tinymem.memory.interfaces import MemoryPolicy
from tinymem.memory.state import MemoryState


class HeavyHitterMemory(MemoryPolicy):
    """Reserve slots for recent entries and high cumulative scores."""

    def __init__(self, capacity: int, *, recent_slots: int) -> None:
        super().__init__(capacity)
        if isinstance(recent_slots, bool) or not isinstance(recent_slots, int):
            raise TypeError("recent_slots must be an integer")
        if not 0 < recent_slots < capacity:
            raise ValueError("recent_slots must be between zero and capacity")
        self.recent_slots = recent_slots
        self.heavy_slots = capacity - recent_slots

    def update(
        self,
        state: MemoryState,
        candidates: MemoryState,
        *,
        generator: torch.Generator | None = None,
    ) -> MemoryState:
        """Return a bounded union of recent and high-score raw tokens."""
        self._validate_update(state, candidates)
        if state.scores is None or candidates.scores is None:
            raise ValueError(
                "HeavyHitterMemory requires scores in state and candidates"
            )
        if state.scores.dtype != candidates.scores.dtype:
            raise ValueError("state and candidate scores must have the same dtype")
        if state.token_ids is None or candidates.token_ids is None:
            raise ValueError(
                "HeavyHitterMemory requires token_ids in state and candidates"
            )
        if not torch.isfinite(state.scores[state.valid]).all():
            raise ValueError("valid state entries must have finite scores")
        if not torch.isfinite(candidates.scores[candidates.valid]).all():
            raise ValueError("valid candidate entries must have finite scores")

        combined_values = torch.cat((state.values, candidates.values), dim=1)
        combined_valid = torch.cat((state.valid, candidates.valid), dim=1)
        combined_token_ids = torch.cat(
            (state.token_ids, candidates.token_ids),
            dim=1,
        )
        combined_positions = torch.cat(
            (state.positions, candidates.positions),
            dim=1,
        )
        combined_scores = torch.cat(
            (state.scores, candidates.scores),
            dim=1,
        )

        result = self.initialize(
            batch_size=state.batch_size,
            model_width=state.model_width,
            device=state.values.device,
            dtype=state.values.dtype,
            with_scores=True,
        )
        assert result.token_ids is not None
        assert result.scores is not None

        for batch_index in range(state.batch_size):
            valid_indices = torch.nonzero(
                combined_valid[batch_index],
                as_tuple=False,
            ).squeeze(-1)
            if valid_indices.numel() == 0:
                continue

            selected_indices = self._select_indices(
                combined_scores[batch_index],
                combined_positions[batch_index],
                valid_indices,
            )
            selected_count = selected_indices.numel()

            result.values[batch_index, :selected_count].copy_(
                combined_values[batch_index, selected_indices]
            )
            result.valid[batch_index, :selected_count] = True
            result.token_ids[batch_index, :selected_count].copy_(
                combined_token_ids[batch_index, selected_indices]
            )
            result.positions[batch_index, :selected_count].copy_(
                combined_positions[batch_index, selected_indices]
            )
            result.scores[batch_index, :selected_count].copy_(
                combined_scores[batch_index, selected_indices]
            )

        return result

    def _select_indices(
        self,
        scores: torch.Tensor,
        positions: torch.Tensor,
        valid_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Select unique heavy and recent entries in chronological order."""
        position_rank = torch.argsort(
            positions[valid_indices],
            descending=True,
            stable=True,
        )
        newest_first = valid_indices[position_rank]
        score_rank = torch.argsort(
            scores[newest_first],
            descending=True,
            stable=True,
        )
        heavy_indices = newest_first[score_rank[: self.heavy_slots]]

        remaining_mask = ~torch.isin(valid_indices, heavy_indices)
        remaining_indices = valid_indices[remaining_mask]
        recent_rank = torch.argsort(
            positions[remaining_indices],
            descending=True,
            stable=True,
        )
        recent_indices = remaining_indices[recent_rank[: self.recent_slots]]

        selected_indices = torch.cat((heavy_indices, recent_indices))
        chronological_rank = torch.argsort(
            positions[selected_indices],
            stable=True,
        )
        return selected_indices[chronological_rank]
