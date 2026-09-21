"""Fixed history features and the single read-side LoRA adapter contract."""
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from tinymem.reader.pretrained import PretrainedReader


@dataclass(frozen=True)
class ReadoutQuery:
    case_id: str
    category: str
    answer: str
    after_ids: tuple[int, ...]
    answer_ids: tuple[int, ...]


@torch.no_grad()
def frozen_history_features(reader: PretrainedReader, history_ids: Sequence[int]) -> torch.Tensor:
    """Copy fixed history-only features to CPU before any read-side adaptation."""
    if any(m.training for m in reader.model.modules()) or any(
        p.requires_grad or p.grad is not None for p in reader.model.parameters()
    ):
        raise ValueError("feature reader must be frozen, in evaluation mode, and have no gradients")
    if not history_ids or any(type(t) is not int or not 0 <= t < reader.model.config.vocab_size for t in history_ids):
        raise ValueError("history must contain valid native token IDs")
    if len(history_ids) > reader.model.config.max_position_embeddings:
        raise ValueError("history exceeds context; no truncation allowed")
    ids = torch.tensor(history_ids, device=reader.model.device)
    positions = torch.arange(len(ids), device=ids.device).unsqueeze(0)
    output = reader.model.get_decoder()(
        inputs_embeds=reader.model.get_input_embeddings()(ids).unsqueeze(0),
        attention_mask=torch.ones_like(positions), position_ids=positions, use_cache=False,
    )
    hidden = output.last_hidden_state[0].float().detach().cpu().clone()
    if not torch.isfinite(hidden).all():
        raise ValueError("history features must be finite")
    return hidden


def configure_read_adapter(reader: PretrainedReader, *, trainable: bool) -> tuple[nn.Parameter, ...]:
    """Use the one loaded ordinary Q/V LoRA; never stack or merge adapters."""
    from peft import PeftModel
    from peft.tuners.lora import LoraLayer

    if type(trainable) is not bool or not isinstance(reader.model, PeftModel):
        raise ValueError("expected a PEFT reader and boolean trainability")
    if set(reader.model.peft_config) != {"default"}:
        raise ValueError("expected exactly one default adapter")
    config = reader.model.peft_config['default']
    if (set(config.target_modules) != {'q_proj', 'v_proj'} or config.bias != 'none'
            or config.modules_to_save or config.use_dora or config.lora_dropout != 0
            or config.lora_bias or config.target_parameters):
        raise ValueError("expected ordinary Q/V LoRA without bias, dropout, or auxiliary parameters")
    adapter = []
    for module in reader.model.modules():
        if isinstance(module, LoraLayer):
            if module.merged or module.disable_adapters:
                raise ValueError("adapter must be active and unmerged")
            adapter.extend(module.lora_A['default'].parameters())
            adapter.extend(module.lora_B['default'].parameters())
    if not adapter or len({id(p) for p in adapter}) != len(adapter):
        raise ValueError("expected distinct adapter parameters")
    reader.model.zero_grad(set_to_none=True)
    reader.model.requires_grad_(False)
    reader.model.set_adapter('default', inference_mode=not trainable)
    selected = tuple(adapter) if trainable else ()
    if {id(p) for p in reader.model.parameters() if p.requires_grad} != {id(p) for p in selected}:
        raise ValueError("unexpected trainable reader parameters")
    # Keep dropout disabled in both arms; gradients do not require training mode.
    reader.model.eval()
    return selected
