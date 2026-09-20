"""Prepare, train, seal, and score the privileged correct-state diagnostic."""

import argparse
import json
from pathlib import Path

import torch

from tinymem.research.delta_fact_profile import write_json
from tinymem.research.delta_fact_protocol import cell_identity
from tinymem.research.oracle_fact_fit import train_cell
from tinymem.research.oracle_fact_protocol import (
    prepare_study, require_training_seal, seal_training, settings, verify_study,
)
from tinymem.research.oracle_fact_scoring import aggregate_study, score_cell
from tinymem.research.pretrained import load_qwen_reader, verify_qwen_snapshot
from tinymem.research.study_runtime import REPOSITORY, prepare_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--snapshot", required=True, type=Path)
    prepare.add_argument("--device", required=True, choices=("cpu", "mps", "cuda"))
    prepare.add_argument("--smoke", action="store_true")
    for stage in ("train", "seal", "evaluate", "report"):
        command = commands.add_parser(stage)
        command.add_argument("--study", required=True, type=Path)
        if stage in ("train", "evaluate"):
            command.add_argument("--snapshot", required=True, type=Path)
            command.add_argument("--cell", required=True, type=int)
    args = parser.parse_args()
    if args.stage == "prepare":
        protocol = prepare_study(REPOSITORY, args.output, verify_qwen_snapshot(args.snapshot),
                                 settings(device=args.device, smoke=args.smoke))
        (args.output / "source/logs").mkdir()
        print(json.dumps({"status": "prepared", "purpose": protocol["settings"]["purpose"],
                          "cells": len(protocol["cells"]), "study": str(args.output)}))
        return
    protocol, dataset = verify_study(args.study, REPOSITORY)
    if args.stage == "seal":
        seal_training(args.study, protocol)
        print(json.dumps({"status": "all_training_sealed"}))
        return
    if args.stage == "report":
        aggregate_study(args.study, protocol)
        print(json.dumps({"status": "report_complete", "path": str(args.study / "report.json")}))
        return
    cell_identity(args.study, protocol, args.cell, args.stage)
    if args.stage == "evaluate":
        require_training_seal(args.study, protocol)
    if verify_qwen_snapshot(args.snapshot) != protocol["snapshot"]:
        raise ValueError("reader snapshot differs from declaration")
    directory = args.study / ("training" if args.stage == "train" else "evaluation") / str(args.cell)
    if directory.exists():
        raise FileExistsError(f"preserve the existing attempt: {directory}")
    device = prepare_device(protocol["settings"]["device"])
    torch.set_num_threads(4)
    reader = load_qwen_reader(args.snapshot, device=device, dtype=torch.bfloat16)
    try:
        result = (train_cell if args.stage == "train" else score_cell)(
            reader, args.study, protocol, dataset, args.cell)
        verify_study(args.study, REPOSITORY)
    except Exception as error:
        if directory.exists() and not (directory / "complete.json").exists():
            write_json(directory / "failure.json", {"stage": args.stage, "error_type": type(error).__name__,
                       "error": str(error), "retry": "none; preserve the failed attempt"})
        raise
    print(json.dumps({"status": "complete", "stage": args.stage, "result": result}, allow_nan=False))


if __name__ == "__main__":
    main()
