"""Low-rank reader adaptation with answer-only, memory-efficient logits."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from tinymem.data.reader_gate import ReaderCase
from tinymem.evaluation.reader_gate import reader_messages
from tinymem.research.pretrained import PretrainedReader


@dataclass(frozen=True)
class ReaderAnswerTokens:
    case_id: str
    prompt_ids: tuple[int, ...]
    answer_ids: tuple[int, ...]


def encode_reader_answer(reader: PretrainedReader, case: ReaderCase) -> ReaderAnswerTokens:
    prompt = reader.tokenizer.apply_chat_template(
        reader_messages(case, condition="full_context"), tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    prompt_ids = tuple(reader.tokenizer.encode(prompt, add_special_tokens=False))
    answer_ids = (*reader.tokenizer.encode(case.answer, add_special_tokens=False), reader.tokenizer.eos_token_id)
    if not prompt_ids or len(answer_ids) < 2 or answer_ids[-1] is None:
        raise ValueError("native prompt, answer, and end-of-turn token are required")
    if len(prompt_ids) + len(answer_ids) > reader.model.config.max_position_embeddings:
        raise ValueError("training example exceeds reader context; truncation is forbidden")
    return ReaderAnswerTokens(case.case_id, prompt_ids, answer_ids)


def attach_reader_lora(reader: PretrainedReader, *, rank: int = 8, checkpointing: bool = True) -> None:
    from peft import LoraConfig, get_peft_model

    if rank <= 0:
        raise ValueError("LoRA rank must be positive")
    reader.model.requires_grad_(False)
    reader.model = get_peft_model(reader.model, LoraConfig(
        task_type="CAUSAL_LM", r=rank, lora_alpha=2 * rank,
        target_modules=["q_proj", "v_proj"], lora_dropout=0.0, bias="none",
    ))
    if checkpointing:
        reader.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reader.model.train()


def reader_answer_loss(reader: PretrainedReader, example: ReaderAnswerTokens) -> torch.Tensor:
    """Compute only answer-position logits; shift once, outside the HF loss."""
    if not example.prompt_ids or not example.answer_ids:
        raise ValueError("prompt and answer tokens must be nonempty")
    input_ids = torch.tensor([(*example.prompt_ids, *example.answer_ids[:-1])], device=reader.model.device)
    target = torch.tensor(example.answer_ids, device=reader.model.device)
    output = reader.model(
        input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
        use_cache=False, logits_to_keep=len(example.answer_ids),
    )
    return F.cross_entropy(output.logits[0].float(), target)
