"""Verify and report one complete paired-update launch without model loading."""
import argparse
import json

from tinymem.evaluation.update_report import report_updates
from tinymem.research.study_runtime import check_repository, repository_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=repository_path, required=True)
    parser.add_argument("--output", type=repository_path, required=True)
    parser.add_argument("--split", choices=("development", "confirmation"), required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    args = parser.parse_args(argv)
    check_repository()
    report = report_updates(args.launch, args.output, split=args.split, resamples=args.resamples)
    print(json.dumps({"output": str(args.output), "evidence_kind": report["evidence_kind"],
                      "histories": report["aggregate"]["histories"], "split": args.split}))


if __name__ == "__main__":
    main()
