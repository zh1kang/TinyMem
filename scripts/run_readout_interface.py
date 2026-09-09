"""Development-only readout study: check, disposable profile, explicit run, report."""

import argparse
import json
import math
from pathlib import Path

from tinymem.research.readout_experiment import profile_arm, run_arm
from tinymem.research.readout_paired_report import build_paired_report, write_paired_report
from tinymem.research.readout_runner import encode_before
from tinymem.research.study_runtime import check_repository, prepare_device, repository_path
from tinymem.research.update_protocol import (
    file_sha256, load_development_data, load_shared_reader, shared_reader_identity,
)


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def finite_nonnegative(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "profile", "run"):
        command = commands.add_parser(name)
        command.add_argument("--data", type=repository_path, required=True)
        if name == "check":
            continue
        command.add_argument("--device", choices=("cpu", "mps", "cuda"), required=True)
        command.add_argument("--output", type=repository_path, required=True)
        command.add_argument("--arm", choices=("affine", "gelu"), required=True)
        command.add_argument("--seed", type=nonnegative, required=True)
        command.add_argument("--steps", type=positive, required=True)
        command.add_argument("--learning-rate", type=finite_nonnegative, required=True)
        command.add_argument("--weight-decay", type=finite_nonnegative, required=True)
        command.add_argument("--max-new-tokens", type=positive, required=True)
        if name == "profile":
            command.add_argument("--histories", type=positive, required=True,
                                 help="training histories selected by length, including the longest")
    report = commands.add_parser("report")
    report.add_argument("--runs", type=repository_path, nargs="+", required=True)
    report.add_argument("--output", type=repository_path, required=True)
    report.add_argument("--resamples", type=positive, default=2000)
    report.add_argument("--bootstrap-seed", type=nonnegative, default=0)
    args = parser.parse_args(argv)
    check_repository()
    if args.command != "check" and args.output.exists():
        parser.error("output must be fresh; partial outputs cannot be resumed")
    if args.command == "report":
        result = build_paired_report(args.runs, resamples=args.resamples, bootstrap_seed=args.bootstrap_seed)
        write_paired_report(result, args.output)
        print(json.dumps({"report": str(args.output), "seeds": result["seeds"]}))
        return
    if args.command != "check" and args.learning_rate == 0:
        parser.error("learning rate must be positive")
    if args.command == "profile" and (args.histories < 2 or args.steps < args.histories):
        parser.error("profile requires at least two histories and one step per selected history")
    data = load_development_data(args.data)
    identity = shared_reader_identity()
    if data.protocol["design"]["status"] != "data_and_measurement_design_not_a_training_launch_protocol":
        parser.error("the production CLI does not run synthetic fixture datasets")
    if args.command == "check":
        print(json.dumps({"input_check": "passed", "training": len(data.train),
                          "development": len(data.development), "reader_adapter": identity["adapter"],
                          "confirmation_opened": False, "model_loaded": False}))
        return
    if args.command == "profile" and args.histories > len(data.train):
        parser.error("profile histories exceed the training set")
    reader = load_shared_reader(identity, prepare_device(args.device))
    train = tuple(encode_before(reader, row) for row in data.train)
    provenance = {"data_protocol_sha256": data.protocol_sha256, "reader": identity,
                  "cli_sha256": file_sha256(Path(__file__)), "confirmation_opened": False}
    options = dict(kind=args.arm, seed=args.seed, steps=args.steps, learning_rate=args.learning_rate,
                   weight_decay=args.weight_decay, max_new_tokens=args.max_new_tokens,
                   input_identity=provenance)
    if args.command == "profile":
        # Deterministic length-spaced coverage, not answer- or accuracy-based selection.
        ordered = sorted(train, key=lambda row: (len(row.history_ids), row.history_id))
        selected = tuple(ordered[i * (len(ordered) - 1) // (args.histories - 1)]
                         for i in range(args.histories))
        result = profile_arm(reader, selected, args.output, **options)
    else:
        development = tuple(encode_before(reader, row) for row in data.development)
        result = run_arm(reader, {"train": train, "development": development}, args.output, **options)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
