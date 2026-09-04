#!/usr/bin/env python3
"""Run the query-blind narrow-slot byte diagnostic, not a full frontier claim."""

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from tinymem.data.replacement_manifest import replacement_manifest
from tinymem.data.replacement_qa import generate_replacement_qa_examples
from tinymem.evaluation.recurrent_slots import evaluate_recurrent_slots
from tinymem.evaluation.replacement_qa import mismatched_history_indices
from tinymem.evaluation.wikitext_checkpoint import load_wikitext_checkpoint
from tinymem.model.recurrent_slot_decoder import RecurrentSlotDecoder
from tinymem.tokenization.byte_tokenizer import ByteTokenizer
from tinymem.training.checkpointing import save_checkpoint
from tinymem.training.recurrent_slots import recurrent_slot_answer_loss
from tinymem.utils.device import select_device
from tinymem.utils.experiment import create_run_directory, current_git_source_state
from tinymem.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--memory-width", type=int, default=8)
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--facts", type=int, default=2)
    parser.add_argument("--segment-length", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-examples", type=int, default=1000)
    parser.add_argument("--validation-examples", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--validation-seed", type=int, default=10000)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    parser.add_argument("--freeze-reader", action="store_true")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/predictions/recurrent_slot_qa"))
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise ValueError("training steps, batch size, and learning rate must be positive")
    device = select_device(args.device)
    seed_everything(args.seed)
    loaded = load_wikitext_checkpoint(args.checkpoint, device=device)
    model = RecurrentSlotDecoder(
        loaded.decoder.model, memory_width=args.memory_width,
        slots=args.slots, segment_length=args.segment_length,
    ).to(device)
    if args.freeze_reader:
        model.reader.requires_grad_(False)
    tokenizer = ByteTokenizer()
    train = generate_replacement_qa_examples(tokenizer, split="train", count=args.train_examples, memory_capacity=args.facts, segment_length=args.segment_length, base_seed=args.data_seed)
    validation = generate_replacement_qa_examples(tokenizer, split="validation", count=args.validation_examples, memory_capacity=args.facts, segment_length=args.segment_length, base_seed=args.validation_seed)
    if args.max_new_tokens <= 0 or any(len(row.query_ids) + args.max_new_tokens > args.segment_length for row in validation):
        raise ValueError("generation must fit inside the query segment")
    manifest = replacement_manifest(train, validation, protocol="history_disjoint_v2", data_seed=args.data_seed, validation_seed=args.validation_seed)
    mismatched_history_indices(validation)
    config = replace(
        loaded.config, seed=args.seed,
        stream=replace(loaded.config.stream, segment_length=args.segment_length),
        memory=replace(loaded.config.memory, n_slots=args.slots, codes_per_write=args.slots),
        training=replace(loaded.config.training, max_steps=args.steps, warmup_steps=0, batch_size=args.batch_size, learning_rate=args.learning_rate, weight_decay=0.01, gradient_clip_norm=1.0),
    )
    source = current_git_source_state(Path(__file__).resolve().parents[1])
    run = create_run_directory(args.artifact_root, config, git_commit=source.commit, source_state=source)
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (run / "data_manifest.json").write_text(manifest_text, encoding="utf-8")
    settings = {
        **{name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        "architecture": "narrow_recurrent_slots_v1",
        "experiment_config_scope": "reader_interface_and_common_training",
        "persistent_state_shape": [args.slots, args.memory_width],
        "reader_interface_width": model.reader.config.d_model,
        "local_input_limit": args.segment_length,
        "source_state": source.to_dict(),
        "data_manifest_sha256": hashlib.sha256(manifest_text.encode()).hexdigest(),
        "parent_checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "objective": "unweighted_answer_cross_entropy",
        "slot_supervision": False,
        "history_chunking": "fixed_bytes_without_event_boundaries",
        "reader_adaptation": "frozen" if args.freeze_reader else "full",
        "persistent_history_state_bytes_per_stream": model.writer.empty(1).nbytes,
        "history_state_measurement": "batch_one_segment_boundary_excludes_query_buffer",
        "shared_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "torch_version": str(torch.__version__),
        "resolved_device": str(device),
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
        "dtype": str(next(model.parameters()).dtype),
    }
    (run / "protocol.json").write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=0.01)
    generator = torch.Generator().manual_seed(args.seed)
    losses = []
    started = time.perf_counter()
    model.train()
    for step in range(args.steps):
        indices = torch.randint(len(train), (args.batch_size,), generator=generator).tolist()
        batch = [train[index] for index in indices]
        optimizer.zero_grad(set_to_none=True)
        loss = recurrent_slot_answer_loss(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 500 == 0 or step + 1 == args.steps:
            print(json.dumps({"step": step + 1, "loss": losses[-1], "seconds": time.perf_counter() - started}), flush=True)
    training_seconds = time.perf_counter() - started
    save_checkpoint(run / "checkpoint.pt", model=model, optimizer=optimizer, step=args.steps, config=config, extra=settings)
    evaluation = evaluate_recurrent_slots(model, validation, max_new_tokens=args.max_new_tokens)
    result = {"status": "development_mechanistic_diagnostic", "protocol": settings, "training_seconds": training_seconds, "losses": losses, "evaluation": evaluation}
    (run / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({name: row["exact_accuracy"] for name, row in evaluation.items()}, indent=2))
    print(f"artifacts: {run}")


if __name__ == "__main__":
    main()
