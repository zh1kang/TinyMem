#!/usr/bin/env python3
"""Download the pinned Qwen3-1.7B snapshot and verify it against Hub hashes.

Writes ``upstream.json`` (the Hub model record for the pinned revision) and the
model files that ``tinymem.reader.pretrained`` requires, then runs the same
verification the study runners perform before loading the reader.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from tinymem.reader.pretrained import QWEN_FILES, QWEN_MODEL_ID, QWEN_REVISION, verify_qwen_snapshot  # noqa: E402

HUB = "https://huggingface.co"
CHUNK_BYTES = 1024 * 1024
TIMEOUT_SECONDS = 120
USER_AGENT = "TinyMem model installer/0.1"


def fetch(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.part")
    partial.unlink(missing_ok=True)
    headers = {"User-Agent": USER_AGENT}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    print(f"download {url}", flush=True)
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response, partial.open("wb") as output:
            for chunk in iter(lambda: response.read(CHUNK_BYTES), b""):
                output.write(chunk)
        os.replace(partial, destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=REPOSITORY / "data/raw/pretrained/qwen3-1.7b")
    args = parser.parse_args()
    destination = args.destination
    upstream = destination / "upstream.json"
    if not upstream.exists():
        fetch(f"{HUB}/api/models/{QWEN_MODEL_ID}/revision/{QWEN_REVISION}?blobs=true", upstream)
    record = json.loads(upstream.read_text(encoding="utf-8"))
    if record.get("sha") != QWEN_REVISION:
        raise ValueError("Hub returned a different revision than the pinned one")
    sizes = {item["rfilename"]: item["size"] for item in record["siblings"]}
    for filename in QWEN_FILES:
        path = destination / filename
        if path.exists() and path.stat().st_size == sizes[filename]:
            continue
        fetch(f"{HUB}/{QWEN_MODEL_ID}/resolve/{QWEN_REVISION}/{filename}", path)
    manifest = verify_qwen_snapshot(destination)
    print(json.dumps({"model_id": manifest["model_id"], "revision": manifest["revision"],
                      "files": len(manifest["files"]), "destination": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
