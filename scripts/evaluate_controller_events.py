#!/usr/bin/env python3
"""Evaluate adaptive write timing on delayed facts and corrections."""

import argparse
import hashlib
import json
from pathlib import Path

from tinymem.data.babi import load_babi_file
from tinymem.data.wikitext import load_wikitext_parquet
from tinymem.evaluation.continuous_checkpoint import load_continuous_checkpoint
from tinymem.evaluation.controller import evaluate_controller_policies
from tinymem.training.continuous import encode_qa_with_distributed_facts
from tinymem.utils.device import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--delay", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--periodic-interval", type=int, default=4)
    parser.add_argument("--random-seed", type=int, required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.delay < 0:
        raise ValueError("delay must be nonnegative")
    if args.batch_size <= 0 or args.periodic_interval <= 0:
        raise ValueError("batch size and periodic interval must be positive")
    if args.random_seed < 0:
        raise ValueError("random seed must be nonnegative")

    repository_root = Path(__file__).resolve().parents[1]
    checkpoint_path = args.checkpoint.resolve()
    device = select_device(args.device)
    loaded = load_continuous_checkpoint(checkpoint_path, device=device)
    validation_examples = load_babi_file(
        repository_root / "data/raw/tasks_1-20_v1-2/en-valid-10k/qa1_valid.txt",
        task_id="qa1",
        split="validation",
    )
    filler_ids = loaded.vocabulary.encode(
        load_wikitext_parquet(
            repository_root / "data/raw/wikitext2/validation.parquet",
            split="validation",
        ).text
    )
    if len(filler_ids) < args.delay:
        raise ValueError("WikiText validation filler is shorter than the delay")
    encoded = []
    for index, example in enumerate(validation_examples):
        start = (index * loaded.decoder.segment_length) % (
            len(filler_ids) - args.delay + 1
        )
        encoded.append(
            encode_qa_with_distributed_facts(
                example,
                loaded.vocabulary,
                distractor_ids=filler_ids[start : start + args.delay],
                segment_length=loaded.decoder.segment_length,
                gap_rotation=index,
            )
        )

    comparison = evaluate_controller_policies(
        loaded.decoder,
        encoded,
        batch_size=args.batch_size,
        pad_id=loaded.vocabulary.token_to_id["<pad>"],
        device=device,
        periodic_interval=args.periodic_interval,
        random_seed=args.random_seed,
    )
    result = {
        "status": "development_validation",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "seed": args.random_seed,
        "delay": args.delay,
        "examples": len(encoded),
        "segment_length": loaded.decoder.segment_length,
        "comparison": comparison.to_dict(),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output / "controller_trace.jsonl").write_text(
        "".join(
            json.dumps(trace.to_dict(), sort_keys=True, allow_nan=False) + "\n"
            for trace in comparison.traces
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    print(f"artifacts: {args.output}")


if __name__ == "__main__":
    main()
