#!/usr/bin/env python3
"""Download and verify the dataset subsets declared in data/manifest.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tarfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


CHUNK_BYTES = 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 120
USER_AGENT = "TinyMem dataset installer/0.1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def expand_dataset_files(
    dataset_name: str, dataset: dict[str, Any]
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []

    for file_spec in dataset.get("files", []):
        source_path = file_spec.get("source_path")
        if "url" in file_spec:
            url = file_spec["url"]
        elif source_path and "base_url" in dataset:
            url = dataset["base_url"] + urllib.parse.quote(source_path)
        else:
            raise ValueError(f"{dataset_name} has a file without a resolvable URL")

        expanded.append(
            {
                **file_spec,
                "dataset": dataset_name,
                "url": url,
            }
        )

    matrix = dataset.get("matrix")
    if matrix:
        for task in matrix["tasks"]:
            for length in matrix["lengths"]:
                values = {"task": task, "length": length}
                source_path = matrix["source_path"].format(**values)
                destination_path = matrix["destination_path"].format(**values)
                expanded.append(
                    {
                        "dataset": dataset_name,
                        "source_path": source_path,
                        "path": destination_path,
                        "url": dataset["base_url"]
                        + urllib.parse.quote(source_path),
                    }
                )

    if not expanded:
        raise ValueError(f"{dataset_name} does not declare any files")
    return expanded


def selected_files(
    manifest: dict[str, Any], group: str
) -> list[dict[str, Any]]:
    try:
        dataset_names = manifest["groups"][group]
    except KeyError as error:
        available = ", ".join(sorted(manifest.get("groups", {})))
        raise ValueError(
            f"unknown group {group!r}; available groups: {available}"
        ) from error

    files: list[dict[str, Any]] = []
    for dataset_name in dataset_names:
        dataset = manifest["datasets"][dataset_name]
        files.extend(expand_dataset_files(dataset_name, dataset))
    return files


def lock_key(file_spec: dict[str, Any]) -> str:
    return f"{file_spec['dataset']}:{file_spec['path']}"


def validate_file(
    path: Path,
    *,
    expected_bytes: int | None,
    expected_sha256: str | None,
) -> tuple[int, str]:
    actual_bytes = path.stat().st_size
    if expected_bytes is not None and actual_bytes != expected_bytes:
        raise ValueError(
            f"{path} has {actual_bytes} bytes; expected {expected_bytes}"
        )

    actual_sha256 = sha256_file(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"{path} has SHA-256 {actual_sha256}; expected {expected_sha256}"
        )
    return actual_bytes, actual_sha256


def download_file(
    url: str,
    destination: Path,
    *,
    expected_bytes: int | None,
    expected_sha256: str | None,
) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.parent / f".{destination.name}.part"
    partial.unlink(missing_ok=True)

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    print(f"download {url}")
    try:
        with urllib.request.urlopen(
            request, timeout=DOWNLOAD_TIMEOUT_SECONDS
        ) as response, partial.open("wb") as output:
            response_bytes = response.headers.get("Content-Length")
            for chunk in iter(lambda: response.read(CHUNK_BYTES), b""):
                output.write(chunk)

        if response_bytes is not None:
            received = partial.stat().st_size
            if received != int(response_bytes):
                raise ValueError(
                    f"partial response for {url}: received {received} of "
                    f"{response_bytes} bytes"
                )

        actual = validate_file(
            partial,
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
        )
        os.replace(partial, destination)
        return actual
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def safe_destination(root: Path, member_name: str) -> Path:
    member = Path(member_name)
    if member.is_absolute():
        raise ValueError(f"archive contains absolute path: {member_name}")

    destination = (root / member).resolve()
    resolved_root = root.resolve()
    if destination != resolved_root and resolved_root not in destination.parents:
        raise ValueError(f"archive path escapes destination: {member_name}")
    return destination


def extract_tar(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target = safe_destination(destination, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"cannot read archive member: {member.name}")
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
            else:
                raise ValueError(
                    f"unsupported tar member type for safety: {member.name}"
                )


def extract_zip(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = safe_destination(destination, member.filename)
            file_type = (member.external_attr >> 16) & 0o170000
            if file_type == stat.S_IFLNK:
                raise ValueError(
                    f"zip archive contains a symbolic link: {member.filename}"
                )
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)


def extract_if_needed(
    archive_path: Path,
    extraction: dict[str, Any],
    raw_root: Path,
    archive_sha256: str,
) -> None:
    destination = raw_root / extraction["destination"]
    marker_name = hashlib.sha256(
        (
            str(archive_path.relative_to(raw_root))
            + ":"
            + archive_sha256
            + ":"
            + extraction["destination"]
        ).encode()
    ).hexdigest()
    marker = raw_root / ".state" / f"{marker_name}.extracted"
    if marker.exists():
        print(f"verified extraction {destination}")
        return

    destination.mkdir(parents=True, exist_ok=True)
    archive_format = extraction["format"]
    print(f"extract {archive_path} -> {destination}")
    if archive_format == "tar.gz":
        extract_tar(archive_path, destination)
    elif archive_format == "zip":
        extract_zip(archive_path, destination)
    else:
        raise ValueError(f"unsupported archive format: {archive_format}")

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{archive_sha256}\n", encoding="utf-8")


def write_lock(
    lock_path: Path,
    manifest_sha256: str,
    records: dict[str, dict[str, Any]],
) -> None:
    value = {
        "schema_version": 1,
        "manifest_sha256": manifest_sha256,
        "files": dict(sorted(records.items())),
    }
    temporary = lock_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, lock_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        default="initial",
        help="manifest group to install (default: initial)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list selected files without downloading them",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    manifest_path = repository_root / "data" / "manifest.json"
    lock_path = repository_root / "data" / "installed.lock.json"
    raw_root = repository_root / "data" / "raw"

    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    files = selected_files(manifest, args.group)

    known_bytes = sum(
        file_spec["bytes"]
        for file_spec in files
        if file_spec.get("bytes") is not None
    )
    if args.list:
        for file_spec in files:
            print(
                f"{file_spec['dataset']:<12} "
                f"{file_spec['path']:<55} "
                f"{file_spec.get('bytes', 'size recorded after download')}"
            )
        print(f"{len(files)} files; at least {known_bytes:,} known bytes")
        return

    previous_records: dict[str, dict[str, Any]] = {}
    if lock_path.exists():
        previous_records = load_json(lock_path).get("files", {})

    installed_records = dict(previous_records)
    raw_root.mkdir(parents=True, exist_ok=True)

    for file_spec in files:
        key = lock_key(file_spec)
        previous = previous_records.get(key, {})
        expected_bytes = file_spec.get("bytes", previous.get("bytes"))
        expected_sha256 = file_spec.get("sha256", previous.get("sha256"))
        destination = raw_root / file_spec["path"]

        if destination.exists():
            actual_bytes, actual_sha256 = validate_file(
                destination,
                expected_bytes=expected_bytes,
                expected_sha256=expected_sha256,
            )
            print(f"verified {destination.relative_to(repository_root)}")
        else:
            actual_bytes, actual_sha256 = download_file(
                file_spec["url"],
                destination,
                expected_bytes=expected_bytes,
                expected_sha256=expected_sha256,
            )

        installed_records[key] = {
            "bytes": actual_bytes,
            "path": file_spec["path"],
            "sha256": actual_sha256,
            "source_url": file_spec["url"],
        }

        extraction = file_spec.get("extract")
        if extraction:
            extract_if_needed(
                destination,
                extraction,
                raw_root,
                actual_sha256,
            )

    write_lock(
        lock_path,
        hashlib.sha256(manifest_bytes).hexdigest(),
        installed_records,
    )
    installed_bytes = sum(
        record["bytes"] for record in installed_records.values()
    )
    print(
        f"installed {len(installed_records)} files "
        f"({installed_bytes:,} downloaded bytes)"
    )


if __name__ == "__main__":
    main()
