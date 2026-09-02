#!/usr/bin/env python3
"""Compose a trained decoder with a separately trained write gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tinymem.data.babilong import load_babilong_file
from tinymem.evaluation.continuous_checkpoint import (
    TOKEN_GATED_MULTISLOT_ARCHITECTURE,
    load_continuous_checkpoint,
)
from tinymem.evaluation.continuous_memory import calibrate_write_threshold
from tinymem.memory.recurrent_memory import GatedRecurrentMemoryBank
from tinymem.memory.write_gate import TokenSegmentWriteGate
from tinymem.training.checkpointing import save_checkpoint
from tinymem.training.continuous import encode_qa_with_evidence_write_targets
from tinymem.utils.device import select_device
from tinymem.utils.experiment import create_run_directory, current_git_commit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--write-gate-checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-file", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-false-positive-rate", type=float, default=0.0)
    parser.add_argument("--minimum-write-recall", type=float, default=0.8)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/predictions/continuous_memory_composed"),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if not 0 <= args.max_false_positive_rate < 1:
        raise ValueError("maximum false-positive rate must be in [0, 1)")
    if not 0 <= args.minimum_write_recall <= 1:
        raise ValueError("minimum write recall must be in [0, 1]")

    repository_root = Path(__file__).resolve().parents[1]
    device = select_device(args.device)
    decoder_path = args.decoder_checkpoint.resolve()
    gate_path = args.write_gate_checkpoint.resolve()
    calibration_path = args.calibration_file.resolve()
    loaded = load_continuous_checkpoint(decoder_path, device=device)
    donor = load_continuous_checkpoint(gate_path, device=device)
    if loaded.config != donor.config:
        raise ValueError("decoder and write-gate checkpoints must share a config")
    if loaded.vocabulary.id_to_token != donor.vocabulary.id_to_token:
        raise ValueError("decoder and write-gate checkpoints must share a vocabulary")
    if not isinstance(donor.decoder.write_gate, TokenSegmentWriteGate):
        raise ValueError("write-gate checkpoint has no token write gate")
    if not isinstance(loaded.decoder.bank, GatedRecurrentMemoryBank):
        raise ValueError("decoder checkpoint has no gated memory bank")
    if (
        loaded.decoder.memory_position_mode
        != donor.decoder.memory_position_mode
    ):
        raise ValueError("decoder and write gate must share a memory position mode")

    loaded.decoder.write_gate = donor.decoder.write_gate
    calibration_examples = [
        encode_qa_with_evidence_write_targets(
            example,
            loaded.vocabulary,
            segment_length=loaded.decoder.segment_length,
        )
        for example in load_babilong_file(
            calibration_path,
            task_id="qa1",
            split="validation",
        )
    ]
    calibration = calibrate_write_threshold(
        loaded.decoder,
        calibration_examples,
        batch_size=args.batch_size,
        pad_id=loaded.vocabulary.token_to_id["<pad>"],
        device=device,
        max_false_positive_rate=args.max_false_positive_rate,
    )
    if calibration.writes.recall < args.minimum_write_recall:
        raise RuntimeError("calibrated write recall is below the required minimum")
    loaded.decoder.bank.set_write_threshold(calibration.threshold)

    commit = current_git_commit(repository_root)
    run_directory = create_run_directory(
        repository_root / args.artifact_root,
        loaded.config,
        git_commit=commit,
    )
    result_document = {
        "status": "development_composed_checkpoint",
        "git_commit": commit,
        "decoder_checkpoint": str(decoder_path),
        "decoder_checkpoint_sha256": _sha256(decoder_path),
        "write_gate_checkpoint": str(gate_path),
        "write_gate_checkpoint_sha256": _sha256(gate_path),
        "calibration_file": str(calibration_path),
        "calibration_file_sha256": _sha256(calibration_path),
        "calibration": calibration.to_dict(),
    }
    (run_directory / "results.json").write_text(
        json.dumps(result_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    gate = loaded.decoder.write_gate
    assert isinstance(gate, TokenSegmentWriteGate)
    save_checkpoint(
        run_directory / "checkpoint.pt",
        model=loaded.decoder,
        step=loaded.step,
        config=loaded.config,
        extra={
            "architecture": TOKEN_GATED_MULTISLOT_ARCHITECTURE,
            "vocabulary": list(loaded.vocabulary.id_to_token),
            "write_threshold": calibration.threshold,
            "memory_position_mode": loaded.decoder.memory_position_mode,
            "write_gate_kernel_size": gate.kernel_size,
            "decoder_checkpoint_sha256": _sha256(decoder_path),
            "write_gate_checkpoint_sha256": _sha256(gate_path),
            "calibration_file_sha256": _sha256(calibration_path),
        },
    )
    print(json.dumps(result_document, indent=2, sort_keys=True))
    print(f"artifacts: {run_directory}")


if __name__ == "__main__":
    main()
