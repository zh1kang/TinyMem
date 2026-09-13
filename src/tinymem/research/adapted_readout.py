"""Separate read-side adaptation from fixed history features and closed studies."""
from collections.abc import Sequence

import torch
from torch import nn

from tinymem.memory.readout_interface import OneShotEncoder, ReadoutBridge, STATE_BYTES
from tinymem.research.prefix_reader import prefix_answer_loss
from tinymem.research.pretrained import PretrainedReader
from tinymem.research.readout_runner import ReadoutQuery


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


def train_adapted_step(
    reader: PretrainedReader, encoder: OneShotEncoder, bridge: ReadoutBridge,
    hidden: torch.Tensor, before_ids: tuple[int, ...], queries: Sequence[ReadoutQuery],
    optimizer: torch.optim.Optimizer, *, adapter_parameters: tuple[nn.Parameter, ...],
) -> dict[str, float | int]:
    """One fixed-feature write and mean answer CE; optimize only declared owners."""
    if any(m.training for m in reader.model.modules()):
        raise ValueError("read-side model must be in evaluation mode")
    reader_owned = {id(p) for p in adapter_parameters}
    if len(reader_owned) != len(adapter_parameters) or reader_owned != {
        id(p) for p in reader.model.parameters() if p.requires_grad
    }:
        raise ValueError("reader adapter ownership changed")
    parameters = list(encoder.parameters()) + list(bridge.parameters()) + list(adapter_parameters)
    owned = {id(p) for p in parameters}
    optimized = [p for group in optimizer.param_groups for p in group['params']]
    if (len(owned) != len(parameters) or len(optimized) != len(owned)
            or {id(p) for p in optimized} != owned or any(not p.requires_grad for p in parameters)):
        raise ValueError("optimizer must own exactly encoder, bridge, and declared read adapter")
    if (hidden.ndim != 2 or hidden.shape[0] == 0 or hidden.shape[1] != encoder.reader_width
            or hidden.dtype != torch.float32 or hidden.device.type != 'cpu' or hidden.requires_grad):
        raise ValueError("expected detached CPU FP32 history features")
    if not queries:
        raise ValueError("queries are required")
    if any(p.grad is not None for p in reader.model.parameters() if id(p) not in reader_owned):
        raise ValueError("frozen reader has unexpected gradients")
    optimizer.zero_grad(set_to_none=True)
    device = reader.model.device
    features = hidden.to(device).unsqueeze(0)
    state = encoder(features, torch.ones(features.shape[:2], device=device, dtype=torch.bool))
    memory = bridge(state)
    before = torch.tensor(before_ids, device=device)
    loss = torch.stack([
        prefix_answer_loss(reader, before, memory, torch.tensor(q.after_ids, device=device),
                           torch.tensor(q.answer_ids, device=device)) for q in queries
    ]).mean()
    if not torch.isfinite(loss):
        raise ValueError("nonfinite answer loss")
    loss.backward()
    if any(p.grad is None for p in parameters):
        raise ValueError("all declared parameters must receive gradients")
    if any(p.grad is not None for p in reader.model.parameters() if id(p) not in reader_owned):
        raise ValueError("frozen reader gradient ownership violation")
    gradient_norms = {}
    for name, group in (('encoder', tuple(encoder.parameters())),
                        ('bridge', tuple(bridge.parameters())), ('adapter', adapter_parameters)):
        value = torch.stack([p.grad.float().square().sum() for p in group]).sum().sqrt() if group else loss.new_zeros(())
        if group and (not torch.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must receive finite nonzero gradients")
        gradient_norms[name + '_gradient_norm'] = float(value)
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    result = {'answer_ce': float(loss.detach()), 'gradient_norm': float(norm),
              'persistent_bytes': STATE_BYTES, 'write_states': 1,
              'supervised_tokens': sum(len(q.answer_ids) for q in queries), **gradient_norms}
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in parameters):
        raise ValueError("optimizer produced nonfinite parameters")
    return result
