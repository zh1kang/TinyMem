"""Append-only JSON Lines metric logging."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO


class JSONLMetricLogger:
    """Write one self-contained metric record per line."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: TextIO | None = self.path.open("a", encoding="utf-8")

    def log(self, metrics: Mapping[str, object], *, step: int | None = None) -> None:
        if self._handle is None:
            raise RuntimeError("cannot log after the logger is closed")
        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a mapping")
        if step is not None:
            if isinstance(step, bool) or not isinstance(step, int):
                raise TypeError("step must be an integer or None")
            if step < 0:
                raise ValueError("step must be nonnegative")

        reserved = {"timestamp", "step"}.intersection(metrics)
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ValueError(f"metrics cannot contain reserved fields: {names}")

        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "step": step,
            **metrics,
        }
        line = json.dumps(record, sort_keys=True, allow_nan=False)
        self._handle.write(line + "\n")
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> JSONLMetricLogger:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
