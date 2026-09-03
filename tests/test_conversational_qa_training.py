import math

import torch

from tinymem.data.schema import ReasoningExample
from tinymem.memory.continuous import MeanPoolMemoryCompressor
from tinymem.memory.recurrent_memory import RecurrentMemoryBank
from tinymem.model.config import ModelConfig
from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.tokenization.byte_tokenizer import ByteTokenizer
from tinymem.training.conversational_qa import (
    collate_conversational_qa,
    encode_conversational_qa_example,
    format_conversational_qa_prompt,
    train_conversational_qa,
)


def make_reasoning_example(answer: str = "blue") -> ReasoningExample:
    context = "The square was red.\nThe square is now blue."
    return ReasoningExample(
        dataset="babi",
        task_id="qa1",
        split="train",
        context=context,
        question="Which color is current?",
        answer=answer,
        supporting_fact_ids=(2,),
        source_length=len(context),
        source_example_id=f"example-{answer}",
        context_fact_ids=(1, 2),
    )


def make_decoder() -> SegmentedContinuousDecoder:
    config = ModelConfig(
        vocab_size=260,
        d_model=8,
        n_layers=1,
        n_heads=2,
        d_ff=16,
        max_local_tokens=8,
        dropout=0.0,
    )
    return SegmentedContinuousDecoder(
        DecoderOnlyTransformer(config),
        MeanPoolMemoryCompressor(8),
        RecurrentMemoryBank(capacity=2, model_width=8),
        segment_length=8,
    )


def test_conversational_prompt_matches_external_answer_boundary() -> None:
    example = make_reasoning_example()

    prompt = format_conversational_qa_prompt(example)

    assert prompt.startswith("[session babi:qa1 | undated]\n")
    assert "User: The square is now blue.\n" in prompt
    assert prompt.endswith("User: Which color is current?\nAssistant:")
    assert not prompt.endswith(example.answer)


def test_collation_supervises_only_answer_and_newline() -> None:
    tokenizer = ByteTokenizer()
    encoded = encode_conversational_qa_example(
        make_reasoning_example(),
        tokenizer,
    )

    input_ids, target_ids, token_valid = collate_conversational_qa(
        [encoded],
        pad_id=tokenizer.special_tokens["<pad>"],
        device="cpu",
    )

    prompt_length = len(encoded.prompt_ids)
    assert (target_ids[0, :prompt_length] == -100).all()
    assert target_ids[0, prompt_length:-1].tolist() == list(encoded.answer_ids)
    assert target_ids[0, -1].item() == ord("\n")
    assert token_valid.all()
    assert input_ids.shape == target_ids.shape == token_valid.shape


def test_conversational_training_updates_decoder_with_lm_anchor() -> None:
    torch.manual_seed(3)
    tokenizer = ByteTokenizer()
    decoder = make_decoder()
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=0.01)
    examples = [
        encode_conversational_qa_example(make_reasoning_example("blue"), tokenizer),
        encode_conversational_qa_example(make_reasoning_example("green"), tokenizer),
    ]
    before = decoder.model.lm_head.weight.detach().clone()

    history = train_conversational_qa(
        decoder,
        optimizer,
        examples,
        steps=2,
        batch_size=2,
        gradient_clip_norm=1.0,
        pad_id=tokenizer.special_tokens["<pad>"],
        device="cpu",
        seed=5,
        language_model_token_ids=torch.arange(128) % 256,
        language_model_loss_weight=0.25,
        language_model_sequence_length=16,
    )

    assert len(history.total_losses) == 2
    assert len(history.answer_losses) == 2
    assert history.language_model_losses is not None
    assert len(history.language_model_losses) == 2
    assert all(math.isfinite(loss) for loss in history.total_losses)
    assert not torch.equal(before, decoder.model.lm_head.weight)
