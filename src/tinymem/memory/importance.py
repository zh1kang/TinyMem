"""Extractive importance-based raw-token memory baseline."""

import torch

from tinymem.memory.interfaces import MemoryPolicy
from tinymem.memory.state import MemoryState


class ExtractiveImportanceMemory(MemoryPolicy):
    """Keep the highest-scoring valid raw-token entries."""

    def update(
        self,
        state: MemoryState,
        candidates: MemoryState,
        *,
        generator: torch.Generator | None = None,
    ) -> MemoryState:
        """Return the highest-scoring entries under the fixed slot capacity.

        Scores are supplied by an upstream importance scorer. This policy only
        ranks entries, preserves aligned metadata, and returns a bounded state.
        Newer positions should win exact score ties, while the final result
        should be stored in chronological order.
        """
        self._validate_update(state, candidates)
        if state.scores is None or candidates.scores is None:
            raise ValueError(
                "ExtractiveImportanceMemory requires scores in state and candidates"
            )
        if state.scores.dtype != candidates.scores.dtype:
            raise ValueError("state and candidate scores must have the same dtype")
        if state.token_ids is None or candidates.token_ids is None:
            raise ValueError(
                "ExtractiveImportanceMemory requires token_ids in state and candidates"
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

            recency_order = torch.argsort(
                combined_positions[batch_index, valid_indices],
                descending=True,
                stable=True,
            )
            ranked_indices = valid_indices[recency_order]
            importance_order = torch.argsort(
                combined_scores[batch_index, ranked_indices],
                descending=True,
                stable=True,
            )
            selected_indices = ranked_indices[importance_order][
                : self.capacity
            ]
            chronological_order = torch.argsort(
                combined_positions[batch_index, selected_indices],
                stable=True,
            )
            selected_indices = selected_indices[chronological_order]
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
