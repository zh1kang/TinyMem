"""Random reservoir-sampling memory baseline for TinyMem."""

import torch

from tinymem.memory.interfaces import MemoryPolicy
from tinymem.memory.state import MemoryState


class RandomReservoirMemory(MemoryPolicy):
    """Keep a uniform random sample of valid raw-token candidates."""

    def update(
        self,
        state: MemoryState,
        candidates: MemoryState,
        *,
        generator: torch.Generator | None = None,
    ) -> MemoryState:
        """Return a fixed-capacity uniform sample of state and candidates."""
        self._validate_update(state, candidates)

        if state.scores is None:
            raise ValueError("RandomReservoirMemory requires scores in state")
        if state.token_ids is None or candidates.token_ids is None:
            raise ValueError(
                "RandomReservoirMemory requires token_ids in state and candidates"
            )

        candidate_scores = torch.rand(
            candidates.valid.shape,
            device=state.values.device,
            dtype=state.scores.dtype,
            generator=generator,
        ).masked_fill(~candidates.valid, float("-inf"))
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
        combined_scores = torch.cat((state.scores, candidate_scores), dim=1)

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
            selected_count = min(self.capacity, valid_indices.numel())
            if selected_count == 0:
                continue

            priority_order = torch.topk(
                combined_scores[batch_index, valid_indices],
                k=selected_count,
                largest=True,
                sorted=False,
            ).indices
            selected_indices = valid_indices[priority_order]
            position_order = torch.argsort(
                combined_positions[batch_index, selected_indices],
                stable=True,
            )
            selected_indices = selected_indices[position_order]

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
