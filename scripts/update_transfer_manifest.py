"""Print NUL-delimited update-study input paths for rsync --files-from=-.

Transfer committed code separately. This list includes the frozen data inputs,
qualified reader, snapshot, and old checkpoints needed for preserved diagnostics;
it excludes predictions from old confirmation runs and local partial results.
"""
from pathlib import Path
import sys

from tinymem.research.study_runtime import REPOSITORY, check_repository, repository_path
from tinymem.research.update_protocol import OLD_STUDY, load_development_data, read_json, shared_reader_identity


def transfer_paths(directory: Path) -> list[str]:
    data = load_development_data(directory)
    identity = shared_reader_identity()
    paths = {str(repository_path(directory))}
    paths.update(str(repository_path(name)) for name in data.protocol["input_sha256"])
    old = read_json(REPOSITORY / OLD_STUDY)
    paths.update(str(repository_path(name)) for name in old["source_sha256"] if name.startswith("artifacts/") or "/artifacts/" in name)
    paths.update(str(repository_path(name)) for name in old["runs"])
    paths.update((identity["adapter"], str(repository_path(old["reader_gate"])),
                  "data/raw/pretrained/qwen3-1.7b", old["data"],
                  "artifacts/predictions/raw_capacity_audit_20260905/opaque_train_vocabulary.json"))
    for name in paths:
        if not (REPOSITORY / name).exists():
            raise FileNotFoundError(f"missing transfer input: {name}")
    return sorted(name for name in paths if not any(name != parent and name.startswith(parent.rstrip("/") + "/") for parent in paths))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=repository_path, default=Path("artifacts/predictions/memory_update_data_20260905_v2"))
    args = parser.parse_args()
    check_repository()
    sys.stdout.buffer.write(b"".join(path.encode() + b"\0" for path in transfer_paths(args.data)))


if __name__ == "__main__":
    main()
