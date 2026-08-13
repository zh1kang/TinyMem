"""Recent raw-token memory baseline."""

import torch

from tinymem.memory.interfaces import MemoryPolicy
from tinymem.memory.state import MemoryState


class RecentTokenMemory(MemoryPolicy):
    """Keep the newest expired tokens within a fixed slot capacity."""

    def update(
        self,
        state: MemoryState,
        candidates: MemoryState,
        *,
        generator: torch.Generator | None = None,
    ) -> MemoryState:
        """Return memory containing the most recent valid items."""
        self._validate_update(state, candidates)
        if state.token_ids is None or candidates.token_ids is None:
            raise ValueError("state and candidates must have token_ids")

        result = self.initialize(
            batch_size=state.batch_size,
            model_width=state.model_width,
            device=state.values.device,
            dtype=state.values.dtype,
        )
        assert result.token_ids is not None
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

        for batch_index in range(state.batch_size):
            valid_indices = torch.nonzero(
                combined_valid[batch_index],
                as_tuple=False,
            ).squeeze(-1)
            if valid_indices.numel() == 0:
                continue

            order = torch.argsort(
                combined_positions[batch_index, valid_indices]
            )
            ordered_indices = valid_indices[order]
            selected_indices = ordered_indices[-self.capacity :]
            selected_count = selected_indices.numel()

            result.values[batch_index, :selected_count].copy_(
                combined_values[batch_index, selected_indices]
            )
            result.token_ids[batch_index, :selected_count].copy_(
                combined_token_ids[batch_index, selected_indices]
            )
            result.positions[batch_index, :selected_count].copy_(
                combined_positions[batch_index, selected_indices]
            )
            result.valid[batch_index, :selected_count] = True

        return result
