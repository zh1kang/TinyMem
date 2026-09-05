"""Run the separate fixed-byte paired-update study; never overwrite old runs."""
from __future__ import annotations

import argparse
import json

from tinymem.research.study_runtime import check_repository, prepare_device, repository_path
from tinymem.research.update_protocol import load_development_data, load_shared_reader, shared_reader_identity
from tinymem.research.update_experiment import SEEDS, evaluate, freeze_launch, profile, qualify, train
from tinymem.research.update_runner import BASELINES, CONTROLS, NEURAL_METHODS


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=repository_path, required=True)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="input and adapter verification; no model or confirmation loading")
    profiling = commands.add_parser("profile", help="ten training-only compute steps; no reusable checkpoint")
    profiling.add_argument("--output", type=repository_path, required=True)
    profiling.add_argument("--method", choices=NEURAL_METHODS, required=True)
    qualification = commands.add_parser("qualify", help="new development full-context reader gate")
    qualification.add_argument("--output", type=repository_path, required=True)
    freeze = commands.add_parser("freeze", help="freeze new launch after successful reader qualification")
    freeze.add_argument("--qualification", type=repository_path, required=True)
    freeze.add_argument("--output", type=repository_path, required=True)
    freeze.add_argument("--steps", type=int, required=True, help="explicit fixed schedule; no checkpoint selection")
    training = commands.add_parser("train")
    training.add_argument("--launch", type=repository_path, required=True)
    training.add_argument("--method", choices=NEURAL_METHODS, required=True)
    training.add_argument("--seed", type=int, choices=SEEDS, required=True)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--launch", type=repository_path, required=True)
    evaluation.add_argument("--method", choices=(*NEURAL_METHODS, *BASELINES, *CONTROLS), required=True)
    evaluation.add_argument("--seed", type=int, choices=SEEDS)
    evaluation.add_argument("--split", choices=("development", "confirmation"), required=True)
    args = parser.parse_args(argv)
    check_repository()
    data = load_development_data(args.data)
    identity = shared_reader_identity()
    if data.protocol["design"]["status"] != "data_and_measurement_design_not_a_training_launch_protocol":
        parser.error("the production CLI does not run synthetic fixture datasets")
    if args.command == "check":
        print(json.dumps({"input_check": "passed", "training": len(data.train), "development": len(data.development),
                          "reader_adapter": identity["adapter"], "confirmation_opened": False, "model_loaded": False}))
        return
    if args.command in ("qualify", "freeze", "profile") and args.output.exists():
        parser.error("output must be fresh")
    if args.command == "train" and (args.launch / f"runs/{args.method}_seed_{args.seed}").exists():
        parser.error("training output already exists; partial runs are not resumable")
    if args.command == "evaluate":
        if (args.method in NEURAL_METHODS) != (args.seed is not None):
            parser.error("only learned methods require a seed")
        name = args.method if args.seed is None else f"{args.method}_seed_{args.seed}"
        if (args.launch / "evaluations" / args.split / name).exists():
            parser.error("evaluation output already exists")
    device = prepare_device(args.device)
    reader = load_shared_reader(identity, device)
    if args.command == "profile":
        result = profile(reader, data, identity, args.output, args.method)
    elif args.command == "qualify":
        result = qualify(reader, data, identity, args.output)
    elif args.command == "freeze":
        result = freeze_launch(reader, data, identity, args.qualification, args.output, steps=args.steps)
    elif args.command == "train":
        result = train(reader, data, identity, args.launch, args.method, args.seed)
    else:
        result = evaluate(reader, data, identity, args.launch, args.method, seed=args.seed, split=args.split)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
