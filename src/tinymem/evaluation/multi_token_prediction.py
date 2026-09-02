"""Validation losses for auxiliary future-token prediction."""

from collections.abc import Sequence

import torch

from tinymem.model.continuous_decoder import SegmentedContinuousDecoder
from tinymem.model.multi_token_prediction import MultiTokenPredictionHeads
from tinymem.model.transformer import DecoderOnlyTransformer
from tinymem.training.continuous import collate_segmented_answer_supervision
from tinymem.training.controlled_qa import EncodedQAExample, collate_answer_supervision
from tinymem.training.multi_token_prediction import multi_token_cross_entropy


def _validate_inputs(
    examples: Sequence[EncodedQAExample],
    *,
    batch_size: int,
) -> None:
    if not examples:
        raise ValueError("examples must be nonempty")
    if not all(isinstance(example, EncodedQAExample) for example in examples):
        raise TypeError("examples must contain EncodedQAExample values")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")


def _update_totals(
    totals: dict[int, float],
    counts: dict[int, int],
    losses: dict[int, torch.Tensor],
    token_valid: torch.Tensor,
) -> None:
    for horizon, loss in losses.items():
        count = int(
            (token_valid[:, :-horizon] & token_valid[:, horizon:]).sum()
        )
        totals[horizon] = totals.get(horizon, 0.0) + float(loss) * count
        counts[horizon] = counts.get(horizon, 0) + count


def _means(totals: dict[int, float], counts: dict[int, int]) -> dict[int, float]:
    return {
        horizon: totals[horizon] / counts[horizon]
        for horizon in sorted(totals)
    }


@torch.no_grad()
def evaluate_base_mtp_loss(
    model: DecoderOnlyTransformer,
    heads: MultiTokenPredictionHeads,
    examples: Sequence[EncodedQAExample],
    *,
    batch_size: int,
    pad_id: int,
    device: torch.device | str,
) -> dict[int, float]:
    """Measure each auxiliary loss on a fixed encoded split."""
    if not isinstance(model, DecoderOnlyTransformer):
        raise TypeError("model must be a DecoderOnlyTransformer")
    if not isinstance(heads, MultiTokenPredictionHeads):
        raise TypeError("heads must be MultiTokenPredictionHeads")
    _validate_inputs(examples, batch_size=batch_size)
    totals: dict[int, float] = {}
    counts: dict[int, int] = {}
    model_was_training = model.training
    heads_were_training = heads.training
    model.eval()
    heads.eval()
    try:
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            input_ids, _ = collate_answer_supervision(
                batch,
                pad_id=pad_id,
                device=device,
            )
            token_valid = torch.zeros_like(input_ids, dtype=torch.bool)
            for row, example in enumerate(batch):
                token_valid[row, : len(example.input_ids)] = True
            losses = multi_token_cross_entropy(
                heads(model.forward_hidden(input_ids)),
                input_ids,
                token_valid,
            ).by_horizon
            _update_totals(totals, counts, losses, token_valid)
    finally:
        model.train(model_was_training)
        heads.train(heads_were_training)
    return _means(totals, counts)


@torch.no_grad()
def evaluate_continuous_mtp_loss(
    decoder: SegmentedContinuousDecoder,
    examples: Sequence[EncodedQAExample],
    *,
    batch_size: int,
    pad_id: int,
    device: torch.device | str,
) -> dict[int, float]:
    """Measure each auxiliary loss through segmented recurrent memory."""
    if not isinstance(decoder, SegmentedContinuousDecoder):
        raise TypeError("decoder must be a SegmentedContinuousDecoder")
    if decoder.mtp_heads is None:
        raise ValueError("decoder must contain MTP heads")
    _validate_inputs(examples, batch_size=batch_size)
    totals: dict[int, float] = {}
    counts: dict[int, int] = {}
    was_training = decoder.training
    decoder.eval()
    try:
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            input_ids, _, token_valid = collate_segmented_answer_supervision(
                batch,
                pad_id=pad_id,
                device=device,
            )
            output = decoder(input_ids, token_valid)
            if output.mtp_logits is None:
                raise RuntimeError("decoder did not return MTP logits")
            losses = multi_token_cross_entropy(
                output.mtp_logits,
                input_ids,
                token_valid,
            ).by_horizon
            _update_totals(totals, counts, losses, token_valid)
    finally:
        decoder.train(was_training)
    return _means(totals, counts)
