"""Answer-supervised rollout through fixed-size history chunks."""

from collections.abc import Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from tinymem.data.replacement_qa import ReplacementQAExample
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.model.recurrent_slot_decoder import RecurrentSlotDecoder
from tinymem.tokenization.byte_tokenizer import ByteTokenizer
from tinymem.training.losses import next_token_cross_entropy
from tinymem.training.replacement_qa import collate_replacement_query


def write_replacement_histories(
    model: RecurrentSlotDecoder, examples: Sequence[ReplacementQAExample],
    *, omit_correction: bool = False,
) -> LatentSlotState:
    """Write history only; event labels and later questions are not model inputs."""
    if not examples or not all(isinstance(row, ReplacementQAExample) for row in examples):
        raise ValueError("examples must contain ReplacementQAExample values")
    device = model.writer.queries.device
    histories = [
        tuple(token for fact in row.initial_fact_ids for token in fact)
        + (() if omit_correction else row.correction_ids)
        for row in examples
    ]
    sequences = [torch.tensor(history, dtype=torch.long, device=device) for history in histories]
    ids = pad_sequence(sequences, batch_first=True, padding_value=ByteTokenizer.special_tokens["<pad>"])
    lengths = torch.tensor([len(history) for history in histories], device=device)
    valid = torch.arange(ids.shape[1], device=device).unsqueeze(0) < lengths.unsqueeze(1)
    state = model.writer.empty(len(examples))
    for start in range(0, ids.shape[1], model.segment_length):
        state = model.write(
            ids[:, start:start + model.segment_length],
            valid[:, start:start + model.segment_length], state,
        )
    return state


def recurrent_slot_answer_loss(
    model: RecurrentSlotDecoder, examples: Sequence[ReplacementQAExample],
) -> torch.Tensor:
    state = write_replacement_histories(model, examples)
    ids, targets, _ = collate_replacement_query(
        examples, pad_id=ByteTokenizer.special_tokens["<pad>"],
        device=model.writer.queries.device,
    )
    return next_token_cross_entropy(model(ids, state), targets)
