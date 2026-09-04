"""Frozen causal interventions for the narrow recurrent-slot diagnostic."""

from collections.abc import Sequence

import torch

from tinymem.data.replacement_qa import ReplacementQAExample
from tinymem.evaluation.replacement_qa import mismatched_history_indices
from tinymem.memory.recurrent_slots import LatentSlotState
from tinymem.model.recurrent_slot_decoder import RecurrentSlotDecoder
from tinymem.tokenization.byte_tokenizer import ByteTokenizer
from tinymem.training.recurrent_slots import write_replacement_histories


@torch.no_grad()
def evaluate_recurrent_slots(
    model: RecurrentSlotDecoder, examples: Sequence[ReplacementQAExample],
    *, max_new_tokens: int = 16,
) -> dict[str, object]:
    if not examples:
        raise ValueError("examples must be nonempty")
    if max_new_tokens <= 0 or any(len(row.query_ids) + max_new_tokens > model.segment_length for row in examples):
        raise ValueError("generation must fit inside the query segment")
    was_training = model.training
    model.eval()
    try:
        state = write_replacement_histories(model, examples)
        indices = torch.tensor(mismatched_history_indices(examples), device=state.values.device)
        conditions = {
            "normal": state,
            "drop": LatentSlotState(state.values, torch.zeros_like(state.valid)),
            "zero": LatentSlotState(torch.zeros_like(state.values), state.valid),
            "shuffle": LatentSlotState(state.values[indices], state.valid[indices]),
            "without_correction": write_replacement_histories(model, examples, omit_correction=True),
        }
        results = {}
        tokenizer = ByteTokenizer()
        for name, memory in conditions.items():
            predictions = []
            for index, example in enumerate(examples):
                row_state = LatentSlotState(memory.values[index:index + 1], memory.valid[index:index + 1])
                generated = []
                for _ in range(max_new_tokens):
                    prefix = example.query_ids + tuple(generated)
                    ids = torch.tensor([prefix], dtype=torch.long, device=state.values.device)
                    next_id = int(model(ids, row_state)[0, -1].argmax())
                    if next_id == ord("\n") or not 0 <= next_id < 256:
                        break
                    generated.append(next_id)
                answer = bytes(generated).decode("utf-8", errors="replace").strip()
                reference = tokenizer.decode(example.answer_ids)
                predictions.append({
                    "source_example_id": example.source_example_id,
                    "prediction": answer, "reference": reference,
                    "exact_match": answer.casefold() == reference.casefold(),
                    "query_requires_correction": example.query_requires_correction,
                })
            results[name] = {
                "count": len(predictions),
                "exact_accuracy": sum(row["exact_match"] for row in predictions) / len(predictions),
                "predictions": predictions,
            }
        return results
    finally:
        model.train(was_training)
