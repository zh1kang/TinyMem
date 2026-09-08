"""Frozen feature extraction for the one-shot readout-interface experiment."""

import torch

from tinymem.memory.readout_interface import OneShotEncoder
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.research.pretrained import PretrainedReader


def encode_readout_history(
    reader: PretrainedReader, encoder: OneShotEncoder, history_ids: torch.Tensor,
) -> LatentSlotState:
    """Extract history-only features without gradients, then train the encoder.

    Only the returned state crosses the write/read boundary. The caller disables
    gradients for inference; training must retain gradients through the encoder.
    """
    if reader.model.training or any(module.training for module in reader.model.modules()):
        raise ValueError("feature reader must be in evaluation mode")
    if any(p.requires_grad or p.grad is not None for p in reader.model.parameters()):
        raise ValueError("feature reader must be frozen with no parameter gradients")
    if history_ids.ndim != 1 or history_ids.numel() == 0 or history_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("history_ids must be a nonempty integer vector")
    embedding = reader.model.get_input_embeddings()
    if history_ids.device != embedding.weight.device or history_ids.device != encoder.queries.device:
        raise ValueError("history, reader, and encoder must share a device")
    if embedding.embedding_dim != encoder.reader_width:
        raise ValueError("reader width must match the encoder")
    if history_ids.numel() > reader.model.config.max_position_embeddings:
        raise ValueError("history exceeds reader context; truncation is forbidden")
    if ((history_ids < 0) | (history_ids >= reader.model.config.vocab_size)).any():
        raise ValueError("history contains an out-of-vocabulary token")
    with torch.no_grad():
        positions = torch.arange(history_ids.numel(), device=history_ids.device).unsqueeze(0)
        output = reader.model.get_decoder()(
            inputs_embeds=embedding(history_ids).unsqueeze(0),
            attention_mask=torch.ones_like(positions), position_ids=positions, use_cache=False,
        )
        hidden = output.last_hidden_state.float()
    return encoder(hidden, torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device))
